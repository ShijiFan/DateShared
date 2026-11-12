"""EXPLORE_S3C_OFFICIAL: official RAINCOAT code (minimal patches) + SFUDA (S3b implementation) with uniform small lr tuning.
Usage: python s3c_official.py tune            -> s3c_tune.csv
       python s3c_official.py run <grp> <lr_raincoat> <lr_sfuda>  -> s3c_official_<grp>.csv (todo[0::2], resumable)"""
import os, sys, types, itertools, time, math
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(HERE, '..', 'multiyear_20261017', 'deep_baseline_2023')
RC = r'E:\research\_ext\s3b_src\raincoat\Raincoat-main'
MODE = sys.argv[1]; GRP = sys.argv[2] if MODE == 'run' else 'B88'
ARGS = list(sys.argv)                                         # sys.argv is overwritten below for the db_run import
# --- official RAINCOAT modules with the two documented patches ---
pml = types.ModuleType('pytorch_metric_learning'); pml.losses = types.SimpleNamespace(ContrastiveLoss=lambda **k: None)
sys.modules['pytorch_metric_learning'] = pml; sys.modules['pytorch_metric_learning.losses'] = pml.losses
sys.path.insert(0, RC)
import models.loss, models.models                                       # official, unmodified
src = open(os.path.join(RC, 'algorithms', 'RAINCOAT.py'), encoding='utf-8').read()
assert src.count('torch.fft.irfft(out_ft, n=128)') == 1
src = src.replace('torch.fft.irfft(out_ft, n=128)', 'torch.fft.irfft(out_ft, n=self.sequence_len)')       # patch 1
RCmod = types.ModuleType('raincoat_official'); exec(compile(src, 'RAINCOAT.py(patched)', 'exec'), RCmod.__dict__)
# --- data / TempCNN / SFUDA from the frozen db_run + S3b ---
sys.argv = [os.path.join(DB, 'db_run.py'), GRP]; sys.path.insert(0, DB); import db_run as D
sys.path.insert(0, os.path.join(HERE, '..', 'multiyear_20261017')); from xyexcl import drop as xydrop
sys.path.insert(0, HERE)
import importlib
DEV = D.DEV; K = D.K; DEVY = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]; MIN_STEPS = 100


# --- SFUDA: identical to s3b_recent.sfuda (S3b changes 1-4), lr exposed for the tuning grid ---
def tsallis(p, a=5.0, dim=-1): return (1 - torch.pow(p + 1e-12, a).sum(dim)) / (a - 1)


def sfuda(net0, Xt, seed, lr=1e-3, bs=1024):
    state = {k: v.clone() for k, v in net0.state_dict().items()}; probs, confs = [], []
    for beta in [i / 10 for i in range(11)]:
        net = D.TempCNN(Xt.shape[1], Xt.shape[2]).to(DEV); net.load_state_dict(state)
        bb = list(net.conv.parameters()) + list(net.feat.parameters()); opt = torch.optim.SGD([{'params': bb, 'lr': lr * 0.1}, {'params': net.out.parameters(), 'lr': lr}])
        net.eval(); g = torch.Generator().manual_seed(seed); nep = max(1, math.ceil(MIN_STEPS / math.ceil(len(Xt) / bs)))
        for _ in range(nep):
            perm = torch.randperm(len(Xt), generator=g).to(DEV)
            for i in range(0, len(Xt), bs):
                q = net(Xt[perm[i:i + bs]]).softmax(1); loss = -beta * tsallis(q.mean(0)) + tsallis(q, dim=1).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): p = torch.cat([net(Xt[i:i + 4096]).softmax(1) for i in range(0, len(Xt), 4096)])
        probs.append(p); confs.append(float(p.max(1).values.mean()))
    c = np.array(confs); w = (c - c.min()) / (c.max() - c.min()) * 0.9 + 0.1 if c.max() > c.min() else np.ones_like(c)
    return (sum(wi * pi for wi, pi in zip(w, probs)) / w.sum()).argmax(1).cpu().numpy()


s3b = types.SimpleNamespace(sfuda=sfuda)


class Cfg:
    def __init__(s, L):
        s.input_channels, s.sequence_len, s.kernel_size, s.stride, s.dropout, s.num_classes = 4, L, 5, 1, 0.5, K
        s.fourier_modes = L // 2; s.mid_channels, s.final_out_channels, s.features_len = 64, 128, 1; s.out_dim = 2 * s.fourier_modes + 128


def raincoat_official(Xs, ys, Xt, seed, lr, ep=40, bs=128):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); cfg = Cfg(Xs.shape[2])
    alg = RCmod.RAINCOAT(cfg, dict(learning_rate=lr, weight_decay=1e-4), DEV)
    XS = torch.tensor(Xs, device=DEV); YS = torch.tensor(ys, device=DEV); nb = max(1, len(XS) // bs)
    for phase in ('update', 'correct'):
        for _ in range(ep):
            alg.train(); perm = torch.randperm(len(XS), generator=g).to(DEV)
            for i in range(nb):
                j = perm[i * bs:(i + 1) * bs]; jt = torch.randint(0, len(Xt), (len(j),), generator=g).to(DEV)
                getattr(alg, phase)(XS[j], YS[j], Xt[jt])
    alg.eval()
    with torch.no_grad():
        return torch.cat([alg.classifier(alg.feature_extractor(Xt[i:i + 4096])[0]).argmax(1) for i in range(0, len(Xt), 4096)]).cpu().numpy()


def transfer(grp_D, ya, yb, lr_rc, lr_sf):
    S = tuple(sorted(set(grp_D.M[f'{ya}_T{grp_D.TRK}']['slots']) & set(grp_D.M[f'{yb}_T{grp_D.TRK}']['slots'])))
    Xa, Ya, Fa = grp_D.data(ya, S); Xb, Yb, Fb = grp_D.data(yb, S); out = {}
    for v in [f'rc_{x}' for x in lr_rc] + [f'sf_{x}' for x in lr_sf]: out[v] = np.full(len(Yb), -1)
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~xydrop(grp_D.GRP, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        mu = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
        norm = lambda X: np.nan_to_num((X - mu) / sd).astype(np.float32); Xs, Xt = norm(Xa[tr]), norm(Xb[te])
        net, XT = grp_D.train(Xs, Ya[tr], Xt, False, grp_D.SEED + k)
        for x in lr_sf: out[f'sf_{x}'][te] = s3b.sfuda(net, XT, grp_D.SEED + k, lr=x)
        for x in lr_rc: out[f'rc_{x}'][te] = raincoat_official(Xs, Ya[tr], XT, grp_D.SEED + k, x)
    return {v: grp_D.bal_acc(Yb[p >= 0], p[p >= 0]) for v, p in out.items()}


if MODE == 'tune':
    TUNE = {'B88': [(2017, 2018), (2019, 2021)], 'A88': [(2018, 2020), (2022, 2024)]}; rows = []; t0 = time.time()
    for g, pairs in TUNE.items():
        sys.argv = [os.path.join(DB, 'db_run.py'), g]; Dg = importlib.reload(D)
        for ya, yb in pairs:
            r = transfer(Dg, ya, yb, (5e-4, 1e-3, 2e-3), (1e-3, 1e-2, 5e-2)); rows.append(dict(group=g, src=ya, tgt=yb, **r))
            print(g, ya, yb, {k: round(v, 3) for k, v in r.items()}, round(time.time() - t0), 's', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(HERE, 's3c_tune.csv'), index=False)
    T = pd.DataFrame(rows); print('mean BA', T.drop(columns=['group', 'src', 'tgt']).mean().round(4).to_dict())
else:
    lr_rc, lr_sf = float(ARGS[3]), float(ARGS[4]); OUT = os.path.join(HERE, f's3c_official_{GRP}.csv')
    done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set()
    years = [y for y in D.YEARS if y in DEVY]; t0 = time.time()
    for ya, yb in list(itertools.permutations(years, 2))[0::2]:
        if (ya, yb) in done: continue
        r = transfer(D, ya, yb, (lr_rc,), (lr_sf,))
        row = dict(group=GRP, src=ya, tgt=yb, raincoat_off=r[f'rc_{lr_rc}'], sfuda_tuned=r[f'sf_{lr_sf}'], lr_rc=lr_rc, lr_sf=lr_sf)
        pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
        print(GRP, ya, '->', yb, round(row['raincoat_off'], 3), round(row['sfuda_tuned'], 3), round(time.time() - t0), 's', flush=True)
    print('DONE', GRP)
