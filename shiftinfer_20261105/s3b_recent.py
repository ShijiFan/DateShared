"""EXPLORE_S3B_RECENT: SFUDA (RSE 2024) and RAINCOAT (ICML 2023) re-implemented on the parcel sequences of db_run.data.
Usage: python s3b_recent.py <grp> [n_transfers for smoke] -> s3b_recent_<grp>.csv (resumable)"""
import os, sys, itertools, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
HERE = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(HERE, '..', 'multiyear_20261017', 'deep_baseline_2023')
GRP = sys.argv[1]; NSMOKE = int(sys.argv[2]) if len(sys.argv) > 2 else 0; sys.argv = [os.path.join(DB, 'db_run.py'), GRP]; sys.path.insert(0, DB)
import db_run as D
sys.path.insert(0, os.path.join(HERE, '..', 'multiyear_20261017')); from xyexcl import drop as xydrop
DEV = D.DEV; K = D.K; DEVY = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]
OUT = os.path.join(HERE, f's3b_recent_{GRP}.csv' if not NSMOKE else f's3b_smoke_{GRP}.csv')


# ---------------- SFUDA ----------------
MIN_STEPS = 100


def tsallis(p, a=5.0, dim=-1): return (1 - torch.pow(p + 1e-12, a).sum(dim)) / (a - 1)


def sfuda(net0, Xt, seed, lr=1e-3, bs=1024):
    state = {k: v.clone() for k, v in net0.state_dict().items()}; probs, confs = [], []
    for beta in [i / 10 for i in range(11)]:
        net = D.TempCNN(Xt.shape[1], Xt.shape[2]).to(DEV); net.load_state_dict(state)
        bb = list(net.conv.parameters()) + list(net.feat.parameters()); opt = torch.optim.SGD([{'params': bb, 'lr': lr * 0.1}, {'params': net.out.parameters(), 'lr': lr}])
        net.eval(); g = torch.Generator().manual_seed(seed)
        nep = max(1, math.ceil(MIN_STEPS / math.ceil(len(Xt) / bs)))          # change 4: keep >= 100 updates per beta
        for _ in range(nep):
            perm = torch.randperm(len(Xt), generator=g).to(DEV)
            for i in range(0, len(Xt), bs):
                q = net(Xt[perm[i:i + bs]]).softmax(1)
                loss = -beta * tsallis(q.mean(0)) + tsallis(q, dim=1).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): p = torch.cat([net(Xt[i:i + 4096]).softmax(1) for i in range(0, len(Xt), 4096)])
        probs.append(p); confs.append(float(p.max(1).values.mean()))
    c = np.array(confs); w = (c - c.min()) / (c.max() - c.min()) * 0.9 + 0.1 if c.max() > c.min() else np.ones_like(c)
    P = sum(wi * pi for wi, pi in zip(w, probs)) / w.sum(); return P.argmax(1).cpu().numpy()


# ---------------- RAINCOAT ----------------
class Spectral(nn.Module):
    def __init__(self, c, m):
        super().__init__(); self.m = m; self.w = nn.Parameter((1 / (c * c)) * torch.rand(c, c, m, dtype=torch.cfloat))
    def forward(self, x):
        xf = torch.fft.rfft(torch.cos(x), norm='ortho'); o = torch.zeros(x.shape[0], self.w.shape[1], x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        o[:, :, :self.m] = torch.einsum('bix,iox->box', xf[:, :, :self.m], self.w)
        return torch.cat([o[:, :, :self.m].abs(), o[:, :, :self.m].angle()], -1), o


class Enc(nn.Module):
    def __init__(self, c, L, m=8, mid=64, fin=128, k=5, drop=0.2):
        super().__init__(); self.m = m
        self.freq = Spectral(c, m); self.bn_f = nn.BatchNorm1d(2 * m); self.avg = nn.Conv1d(c, 1, 3, padding=1, bias=False)
        self.b1 = nn.Sequential(nn.Conv1d(c, mid, k, padding=k // 2, bias=False), nn.BatchNorm1d(mid), nn.ReLU(), nn.MaxPool1d(2, 2, 1), nn.Dropout(drop))
        self.b3 = nn.Sequential(nn.Conv1d(mid, fin, 8, padding=4, bias=False), nn.BatchNorm1d(fin), nn.ReLU(), nn.MaxPool1d(2, 2, 1))
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        ef, o = self.freq(x); ef = Fn.relu(self.bn_f(self.avg(ef).squeeze(1)))
        et = self.pool(self.b3(self.b1(x))).flatten(1)
        return Fn.normalize(torch.cat([ef, et], -1)), o


class Dec(nn.Module):
    def __init__(self, c, L, m=8, fin=128):
        super().__init__(); self.L, self.m = L, m; self.bn1 = nn.BatchNorm1d(c); self.bn2 = nn.BatchNorm1d(c); self.convT = nn.ConvTranspose1d(fin, L, c)
    def forward(self, f, o):
        xl = self.bn1(torch.fft.irfft(o, n=self.L)); et = f[:, 2 * self.m:]
        return xl + Fn.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0, 2, 1)))


def sinkhorn(x, y, eps=1e-3, it=1000):
    C = torch.cdist(x, y, p=2) ** 2; n, m = C.shape
    mu = torch.full((n,), 1.0 / n, device=x.device); nu = torch.full((m,), 1.0 / m, device=x.device); u = torch.zeros_like(mu); v = torch.zeros_like(nu)
    M = lambda u, v: (-C + u[:, None] + v[None, :]) / eps
    for _ in range(it):
        u1 = u; u = eps * (torch.log(mu + 1e-8) - torch.logsumexp(M(u, v), 1)) + u
        v = eps * (torch.log(nu + 1e-8) - torch.logsumexp(M(u, v).T, 1)) + v
        if (u - u1).abs().sum().item() < 1e-1: break
    return (torch.exp(M(u, v)) * C).sum()


def raincoat(Xs, ys, Xt, seed, ep=25, bs=256, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); c, L = Xs.shape[1], Xs.shape[2]
    enc, dec = Enc(c, L).to(DEV), Dec(c, L).to(DEV); clf = nn.Linear(2 * 8 + 128, K, bias=False).to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + list(clf.parameters()), lr=lr, weight_decay=wd)
    copt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=wd)
    XS, YS = Xs, torch.tensor(ys, device=DEV); nb = max(1, len(XS) // bs); l1 = nn.L1Loss(reduction='sum')
    for phase in ('align', 'correct'):
        for _ in range(ep):
            enc.train(); dec.train(); clf.train(); perm = torch.randperm(len(XS), generator=g).to(DEV)
            for i in range(nb):
                j = perm[i * bs:(i + 1) * bs]; jt = torch.randint(0, len(Xt), (len(j),), generator=g).to(DEV)
                fs, os_ = enc(XS[j]); ft, ot = enc(Xt[jt])
                rec = 1e-4 * (l1(dec(fs, os_), XS[j]) + l1(dec(ft, ot), Xt[jt]))
                if phase == 'align':
                    loss = rec + sinkhorn(fs, ft) + Fn.cross_entropy(clf(fs) / 0.1, YS[j]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                else:
                    copt.zero_grad(set_to_none=True); rec.backward(); copt.step()
    enc.eval(); clf.eval()
    with torch.no_grad(): return torch.cat([(clf(enc(Xt[i:i + 4096])[0]) / 0.1).argmax(1) for i in range(0, len(Xt), 4096)]).cpu().numpy()


DEV = D.DEV
done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set()
years = [y for y in D.YEARS if y in DEVY]; t0 = time.time(); todo = list(itertools.permutations(years, 2))
for ya, yb in (todo[:NSMOKE] if NSMOKE else todo):
    if (ya, yb) in done: continue
    S = tuple(sorted(set(D.M[f'{ya}_T{D.TRK}']['slots']) & set(D.M[f'{yb}_T{D.TRK}']['slots'])))
    Xa, Ya, Fa = D.data(ya, S); Xb, Yb, Fb = D.data(yb, S)
    out = {v: np.full(len(Yb), -1) for v in ('sfuda', 'raincoat')}
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~xydrop(GRP, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        mu = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
        norm = lambda X: np.nan_to_num((X - mu) / sd).astype(np.float32)
        Xs, Xt = norm(Xa[tr]), norm(Xb[te])
        net, XT = D.train(Xs, Ya[tr], Xt, False, D.SEED + k)            # identical source model to S3a dl_raw
        out['sfuda'][te] = sfuda(net, XT, D.SEED + k)
        out['raincoat'][te] = raincoat(torch.tensor(Xs, device=DEV), Ya[tr], XT, D.SEED + k)
    row = dict(group=GRP, src=ya, tgt=yb, **{v: D.bal_acc(Yb[p >= 0], p[p >= 0]) for v, p in out.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(GRP, ya, '->', yb, {v: round(row[v], 3) for v in out}, round(time.time() - t0), 's', flush=True)
print('DONE', GRP)
