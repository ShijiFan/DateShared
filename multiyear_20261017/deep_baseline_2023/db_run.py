"""DB deep baselines (NOTE_DB_deep_baseline.md): TempCNN dl_raw / dl_adabn / dl_dann on the same transfers, folds and
target-data usage as eval_prereg.py. 2023 is never read. Usage: python db_run.py <A88|B88|A37> [shard/nshards]
Resumable: appends one block of rows per transfer to db_<group>[_s<i>].csv."""
import os, sys, json, time, math, itertools
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); MY = os.path.dirname(HERE); ROOT = r'D:\sar_data\multiyear'
GRP = sys.argv[1]; SITE, TRK = GRP[0], int(GRP[1:])
SH, NSH = (int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else '0/1').split('/'))
K, BS, LR, WD = 7, 256, 1e-3, 1e-4
SEED = int(os.environ.get('DB_SEED', 20261019))          # NOTE section 7(b): extra seeds 20261020 / 20261021
INDUCT = os.environ.get('DB_INDUCT', '0') == '1'          # NOTE section 7(c): adapt on target fold-k TRAIN parcels, evaluate on TEST
EPOCHS = int(os.environ.get('DB_EPOCHS', 25))   # env override only for smoke tests
SUFFIX = ('' if SEED == 20261019 else f'_seed{SEED}') + ('_induct' if INDUCT else '')
DEV = torch.device('cuda')
Z = np.load(os.path.join(ROOT, f'features_{SITE}.npz'), allow_pickle=True); M = json.load(open(os.path.join(ROOT, f'features_{SITE}_meta.json')))
YEARS = sorted(int(t.split('_')[0]) for t in M if t.endswith(f'_T{TRK}'))


def data(yr, S):
    m = M[f'{yr}_T{TRK}']; pos = [m['slots'].index(s) for s in S]; dc = [c for p in pos for c in (2 * p, 2 * p + 1)]
    g = lambda l: Z[f'{yr}_T{TRK}_{l}'].astype(np.float32)
    vv, vh, co = g('VV')[:, dc], g('VH')[:, dc], g('COH')[:, pos]
    seq = np.stack([vv, vh, vh - vv, np.repeat(co, 2, axis=1)], axis=1)          # (n, 4, 2S)
    return seq, Z[f'{yr}_T{TRK}_y'], Z[f'{yr}_T{TRK}_folds']


class TempCNN(nn.Module):
    def __init__(self, c_in, length, n_cls=K):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(0.2))
        self.conv = nn.Sequential(blk(c_in, 64), blk(64, 64), blk(64, 64))
        self.feat = nn.Sequential(nn.Flatten(), nn.Linear(64 * length, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2))
        self.out = nn.Linear(256, n_cls)

    def forward(self, x, return_feat=False):
        f = self.feat(self.conv(x)); return (self.out(f), f) if return_feat else self.out(f)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None


def bal_acc(y, p):
    cm = np.zeros((K, K)); np.add.at(cm, (y, p), 1); s = cm.sum(1)
    return float(np.mean(np.divide(np.diag(cm), s, out=np.zeros(K), where=s > 0)[s > 0]))


def train(Xs, ys, Xt, dann, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = TempCNN(Xs.shape[1], Xs.shape[2]).to(DEV); params = list(net.parameters()); disc = None
    if dann:
        disc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)).to(DEV); params += list(disc.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    cnt = np.bincount(ys, minlength=K).astype(float)
    w = torch.tensor(np.where(cnt > 0, cnt.sum() / (K * np.maximum(cnt, 1)), 0.0), dtype=torch.float32, device=DEV)
    XS, YS, XT = torch.tensor(Xs, device=DEV), torch.tensor(ys, device=DEV), torch.tensor(Xt, device=DEV)
    nb = max(1, len(XS) // BS); total = EPOCHS * nb; step = 0
    for _ in range(EPOCHS):
        net.train(); perm = torch.randperm(len(XS), generator=g).to(DEV)
        for i in range(nb):
            j = perm[i * BS:(i + 1) * BS]
            if dann:
                jt = torch.randint(0, len(XT), (len(j),), generator=g).to(DEV)
                lo, fs = net(XS[j], True); _, ft = net(XT[jt], True)
                lam = 0.3 * (2 / (1 + math.exp(-10 * step / total)) - 1)
                d = disc(GRL.apply(torch.cat([fs, ft]), lam))[:, 0]
                dy = torch.cat([torch.ones(len(j), device=DEV), torch.zeros(len(j), device=DEV)])
                loss = F.cross_entropy(lo, YS[j], weight=w) + F.binary_cross_entropy_with_logits(d, dy)
            else:
                loss = F.cross_entropy(net(XS[j]), YS[j], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
    return net, XT


def predict(net, X):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(X[i:i + 4096]).argmax(1) for i in range(0, len(X), 4096)]).cpu().numpy()


def adabn_predict(net, X):
    """re-estimate BN running statistics on unlabeled target data, then predict"""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d): m.reset_running_stats(); m.momentum = None   # cumulative average
    net.train()
    with torch.no_grad():
        for i in range(0, len(X), 1024): net.feat(net.conv(X[i:i + 1024]))
    return predict(net, X)


def adabn_predict2(net, Xadapt, Xeval):
    """inductive AdaBN: BN statistics from the adaptation set, prediction on a disjoint evaluation set"""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm1d): m.reset_running_stats(); m.momentum = None
    net.train()
    with torch.no_grad():
        for i in range(0, len(Xadapt), 1024): net.feat(net.conv(Xadapt[i:i + 1024]))
    return predict(net, Xeval)


def eval_refs():
    fs = [os.path.join(MY, f'eval_{GRP}.csv')] + [os.path.join(MY, f'eval_{GRP}_s{i}.csv') for i in range(16)]
    E = pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)])
    E = E[E.variant == 'raw']
    return {(int(r.src), int(r.tgt)): (float(r.ba_within), float(r.ba_raw)) for r in E.itertuples()}


def main():
    REF = eval_refs()
    OUTP = os.path.join(HERE, f'db_{GRP}{SUFFIX}.csv' if NSH == 1 else f'db_{GRP}{SUFFIX}_s{SH}.csv'); done = set()
    for f in [os.path.join(HERE, f'db_{GRP}{SUFFIX}.csv')] + [os.path.join(HERE, f'db_{GRP}{SUFFIX}_s{i}.csv') for i in range(16)]:
        if os.path.exists(f): done |= {(r.src, r.tgt) for r in pd.read_csv(f).itertuples()}
    todo = [p for p in itertools.permutations(YEARS, 2) if p not in done and p in REF and 2023 in p]   # FREEZE C5
    t0 = time.time()
    for ya, yb in todo[SH::NSH]:
        S = tuple(sorted(set(M[f'{ya}_T{TRK}']['slots']) & set(M[f'{yb}_T{TRK}']['slots'])))
        Xa, Ya, Fa = data(ya, S); Xb, Yb, Fb = data(yb, S)
        out = {v: np.full(len(Yb), -1) for v in ('dl_raw', 'dl_adabn', 'dl_dann')}
        for k in range(6):
            tr, te = Fa[:, k] == 'TRAIN', Fb[:, k] == 'TEST'
            if not te.sum(): continue
            mu = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
            norm = lambda X: np.nan_to_num((X - mu) / sd).astype(np.float32)
            Xs, Xt = norm(Xa[tr]), norm(Xb[te])
            if INDUCT:   # adaptation data = target fold-k TRAIN parcels (unlabeled); evaluation on fold-k TEST parcels
                Xad = norm(Xb[Fb[:, k] == 'TRAIN'])
                net, XAD = train(Xs, Ya[tr], Xad, False, SEED + k); XT = torch.tensor(Xt, device=DEV)
                out['dl_raw'][te] = predict(net, XT)
                out['dl_adabn'][te] = adabn_predict2(net, XAD, XT)
                net2, _ = train(Xs, Ya[tr], Xad, True, SEED + k)
                out['dl_dann'][te] = predict(net2, XT)
            else:
                net, XT = train(Xs, Ya[tr], Xt, False, SEED + k)
                out['dl_raw'][te] = predict(net, XT)
                out['dl_adabn'][te] = adabn_predict(net, XT)
                net2, _ = train(Xs, Ya[tr], Xt, True, SEED + k)
                out['dl_dann'][te] = predict(net2, XT)
        bw, br = REF[(ya, yb)]
        rows = []
        for v, p in out.items():
            b = bal_acc(Yb[p >= 0], p[p >= 0])
            rows.append(dict(group=GRP, src=ya, tgt=yb, n_slots=len(S), seed=SEED, induct=int(INDUCT), variant=v, ba=b, ba_within=bw, ba_raw=br, drop=bw - br,
                             recovery=(b - br) / (bw - br) if bw - br > 0.01 else np.nan, reference_only=int({ya, yb} == {2024, 2025})))
        pd.DataFrame(rows).to_csv(OUTP, mode='a', header=not os.path.exists(OUTP), index=False)
        print(GRP, ya, '->', yb, {r['variant']: round(r['ba'], 3) for r in rows}, 'raw_hgb', round(br, 3), round(time.time() - t0), 's', flush=True)
    print('DONE', GRP, SH, NSH)


if __name__ == '__main__':
    main()
