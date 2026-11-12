"""EXPLORE_S3D_Q1: recent Q1-journal baselines. RAINCOAT-SRS (ISPRS JPRS 2025, re-implemented on the official RAINCOAT code per paper
Table 4) with three pre-registered checkpoint rules, and SFUDA (RSE 2024, S3c implementation, lr 1e-3) on France.
Usage: python s3d_q1.py tune                 -> s3d_tune.csv
       python s3d_q1.py run <B88|A37|A88|FR> <lr_srs>   -> s3d_<grp>.csv, s3d_trace_<grp>.csv (resumable)
       python s3d_q1.py smoke                -> one fold of one transfer, timing only"""
import os, sys, types, itertools, time, math, json
import numpy as np, pandas as pd, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(HERE, '..', 'multiyear_20261017', 'deep_baseline_2023')
RC = r'E:\research\_ext\s3b_src\raincoat\Raincoat-main'
ARGS = list(sys.argv); MODE = ARGS[1]; GRP = ARGS[2] if MODE == 'run' else 'B88'
# --- official RAINCOAT modules with the two S3c patches ---
pml = types.ModuleType('pytorch_metric_learning'); pml.losses = types.SimpleNamespace(ContrastiveLoss=lambda **k: None)
sys.modules['pytorch_metric_learning'] = pml; sys.modules['pytorch_metric_learning.losses'] = pml.losses
sys.path.insert(0, RC)
import models.loss, models.models                                       # official, unmodified
src = open(os.path.join(RC, 'algorithms', 'RAINCOAT.py'), encoding='utf-8').read()
assert src.count('torch.fft.irfft(out_ft, n=128)') == 1
src = src.replace('torch.fft.irfft(out_ft, n=128)', 'torch.fft.irfft(out_ft, n=self.sequence_len)')
RCmod = types.ModuleType('raincoat_official'); exec(compile(src, 'RAINCOAT.py(patched)', 'exec'), RCmod.__dict__)
# --- frozen db_run (TempCNN, train, bal_acc); NL data via db_run, FR data inlined from newregion/fr_deep.py ---
sys.argv = [os.path.join(DB, 'db_run.py'), 'B88' if GRP == 'FR' else GRP]; sys.path.insert(0, DB); import db_run as D
sys.path.insert(0, os.path.join(HERE, '..', 'multiyear_20261017')); from xyexcl import drop as nl_xydrop
import importlib
DEV = D.DEV; K = D.K; DEVY = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]; MIN_STEPS = 100


class NLData:
    def __init__(s, g):
        sys.argv = [os.path.join(DB, 'db_run.py'), g]; s.D = importlib.reload(D); s.g = g
        s.years = [y for y in s.D.YEARS if y in DEVY]; s.pairs = list(itertools.permutations(s.years, 2))[0::2]
    def slots(s, y): return s.D.M[f'{y}_T{s.D.TRK}']['slots']
    def data(s, y, S): return s.D.data(y, S)
    def drop(s, ya, yb, k, n): return nl_xydrop(s.g, ya, yb, k, n)


class FRData:
    ROOT = r'D:\sar_data\fr_a'
    def __init__(s):
        s.Z = np.load(os.path.join(s.ROOT, 'features_FR.npz'), allow_pickle=True); s.M = json.load(open(os.path.join(s.ROOT, 'features_FR_meta.json')))
        s.X = np.load(os.path.join(s.ROOT, 'xyexcl_FR.npz')); s.TAG = {int(t.split('_')[0]): t for t in s.M}
        s.years = sorted(s.TAG); s.pairs = list(itertools.permutations(s.years, 2)); s.g = 'FR'
    def slots(s, y): return s.M[s.TAG[y]]['slots']
    def data(s, y, S):          # identical to fr_deep.data
        m = s.M[s.TAG[y]]; pos = [m['slots'].index(x) for x in S]; dc = [c for p in pos for c in (2 * p, 2 * p + 1)]
        g = lambda l: s.Z[f'{s.TAG[y]}_{l}'].astype(np.float32); vv, vh, co = g('VV')[:, dc], g('VH')[:, dc], g('COH')[:, pos]
        return np.stack([vv, vh, vh - vv, np.repeat(co, 2, axis=1)], axis=1), s.Z[f'{s.TAG[y]}_y'], s.Z[f'{s.TAG[y]}_folds']
    def drop(s, ya, yb, k, n):  # identical to fr_deep.xydrop
        m = np.zeros(n, bool)
        if int(ya) != int(yb): m[s.X[f'{int(ya)}_{int(yb)}_{int(k)}']] = True
        return m


# --- RAINCOAT-SRS time branch (ISPRS JPRS 2025, Table 4) ---
class PosEnc(nn.Module):
    def __init__(s, d, dropout=0.1, max_len=512):
        super().__init__(); s.drop = nn.Dropout(dropout); pe = torch.zeros(max_len, d); pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d)); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        s.register_buffer('pe', pe.unsqueeze(0))
    def forward(s, x): return s.drop(x + s.pe[:, :x.size(1)])


class SRSTime(nn.Module):
    def __init__(s, c_in, d=164, h1=82):
        super().__init__(); s.mlp = nn.Sequential(nn.Linear(c_in, h1), nn.ReLU(), nn.Linear(h1, d)); s.pe = PosEnc(d, 0.1)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 1024, 0.1, batch_first=True), 2)
    def forward(s, x): return s.enc(s.pe(s.mlp(x.permute(0, 2, 1)))).mean(1)      # (B, C, T) -> (B, 164)


class Cfg:
    def __init__(s, L):
        s.input_channels, s.sequence_len, s.kernel_size, s.stride, s.dropout, s.num_classes = 4, L, 5, 1, 0.5, K
        s.fourier_modes = L // 2; s.mid_channels, s.final_out_channels, s.features_len = 64, 164, 1; s.out_dim = 2 * s.fourier_modes + 164


def srs_model(L, lr):
    cfg = Cfg(L); alg = RCmod.RAINCOAT(cfg, dict(learning_rate=lr, weight_decay=1e-4), DEV)
    alg.feature_extractor.cnn = SRSTime(4).to(DEV)                       # the only architectural change: ATT encoder + default decoder
    ps = lambda: list(alg.feature_extractor.parameters()) + list(alg.decoder.parameters())
    alg.optimizer = torch.optim.Adam(ps() + list(alg.classifier.parameters()), lr=lr, weight_decay=1e-4)
    alg.coptimizer = torch.optim.Adam(ps(), lr=lr, weight_decay=1e-4); return alg


def chunks(n, size=4096):      # equal-size chunks, never of length 1 (official encoder squeezes the batch dim)
    k = max(1, math.ceil(n / size)); return [c for c in torch.tensor_split(torch.arange(n), k) if len(c)]


@torch.no_grad()
def predict(alg, X):
    alg.eval(); one = len(X) == 1; X = torch.cat([X, X]) if one else X
    out = torch.cat([alg.classifier(alg.feature_extractor(X[c.to(X.device)])[0]).argmax(1) for c in chunks(len(X))]).cpu().numpy()
    return out[:1] if one else out


def raincoat_srs(Xs, ys, Xv, yv, XT, XTv, yTv, seed, lr, ep=40, bs=128, yT=None):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); alg = srs_model(Xs.shape[2], lr)
    XS = torch.tensor(Xs, device=DEV); YS = torch.tensor(ys, device=DEV); XV = torch.tensor(Xv, device=DEV); nb = max(1, len(XS) // bs)
    best = {'A': (-1, None, -1), 'C': (-1, None, -1), 'D': (-1, None, -1)}; trace = []; e = 0
    for phase in ('update', 'correct'):
        for _ in range(ep):
            alg.train(); perm = torch.randperm(len(XS), generator=g).to(DEV)
            for i in range(nb):
                j = perm[i * bs:(i + 1) * bs]; jt = torch.randint(0, len(XT), (len(j),), generator=g).to(DEV)
                getattr(alg, phase)(XS[j], YS[j], XT[jt])
            e += 1; pt = predict(alg, XT); bv = D.bal_acc(yv, predict(alg, XV)); bt = D.bal_acc(yTv, predict(alg, XTv))
            bT = D.bal_acc(yT, pt); trace.append(dict(epoch=e, phase=phase, src_val=bv, tgt_val=bt, tgt_test=bT))
            if bv >= best['A'][0]: best['A'] = (bv, pt, e)
            if bt >= best['C'][0]: best['C'] = (bt, pt, e)
            if bT >= best['D'][0]: best['D'] = (bT, pt, e)     # official trainer: best epoch on target test labels
    return {'A': best['A'][1], 'B': pt, 'C': best['C'][1], 'D': best['D'][1]}, {'A': best['A'][2], 'C': best['C'][2], 'D': best['D'][2]}, trace


# --- SFUDA: identical to s3c_official.sfuda (S3b changes 1-4) ---
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


def transfer(G, ya, yb, lrs, do_sfuda, folds=range(6)):
    S = tuple(sorted(set(G.slots(ya)) & set(G.slots(yb)))); Xa, Ya, Fa = G.data(ya, S); Xb, Yb, Fb = G.data(yb, S)
    out = {f'srs{r}_{x}': np.full(len(Yb), -1) for x in lrs for r in 'ABCD'}; traces = []; sel = []
    if do_sfuda: out['sfuda'] = np.full(len(Yb), -1)
    for k in folds:
        tr = (Fa[:, k] == 'TRAIN') & ~G.drop(ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        rng = np.random.default_rng(D.SEED + k); itr = np.flatnonzero(tr); hold = np.zeros(len(Fa), bool); hold[rng.choice(itr, len(itr) // 5, replace=False)] = True
        tv_pool = np.flatnonzero(Fb[:, k] == 'TRAIN'); tv = rng.choice(tv_pool, min(2000, len(tv_pool)), replace=False)
        mu = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
        norm = lambda X: np.nan_to_num((X - mu) / sd).astype(np.float32); T = lambda X: torch.tensor(norm(X), device=DEV)
        XT, XTv = T(Xb[te]), T(Xb[tv])
        if do_sfuda:                                       # same source model / normalisation as S3c (all TRAIN, no hold-out)
            net, XT0 = D.train(norm(Xa[tr]), Ya[tr], norm(Xb[te]), False, D.SEED + k); out['sfuda'][te] = sfuda(net, XT0, D.SEED + k, lr=1e-3)
        fit = tr & ~hold
        for x in lrs:
            p, ep, tc = raincoat_srs(norm(Xa[fit]), Ya[fit], norm(Xa[hold]), Ya[hold], XT, XTv, Yb[tv], D.SEED + k, x, yT=Yb[te])
            for r in 'ABCD': out[f'srs{r}_{x}'][te] = p[r]
            sel.append(dict(fold=k, lr=x, ep_A=ep['A'], ep_C=ep['C'], ep_D=ep['D'])); traces += [dict(src=ya, tgt=yb, fold=k, lr=x, **t) for t in tc]
    return {v: D.bal_acc(Yb[q >= 0], q[q >= 0]) for v, q in out.items()}, sel, traces


if MODE == 'smoke':
    G = NLData('B88'); t0 = time.time(); r, sel, tr = transfer(G, 2019, 2020, (1e-3,), True, folds=[0])
    print({k: round(v, 3) for k, v in r.items()}, sel, round(time.time() - t0), 's'); print(pd.DataFrame(tr)[['epoch', 'phase', 'src_val', 'tgt_val']].iloc[::8].round(3))
elif MODE == 'tune1':        # one (transfer, lr) cell of the same tuning grid, for parallel tuning; appends to s3d_tune_parts.csv
    g, ya, yb, x = ARGS[2], int(ARGS[3]), int(ARGS[4]), float(ARGS[5]); G = NLData(g); r, _, _ = transfer(G, ya, yb, (x,), False)
    row = pd.DataFrame([dict(group=g, src=ya, tgt=yb, lr=x, srsA=r[f'srsA_{x}'], srsB=r[f'srsB_{x}'], srsC=r[f'srsC_{x}'], srsD=r[f'srsD_{x}'])])
    P = os.path.join(HERE, 's3d_tune_parts.csv'); time.sleep(np.random.default_rng().uniform(0, 5))
    row.to_csv(P, mode='a', header=not os.path.exists(P), index=False); print(row.to_dict('records'), flush=True)
elif MODE == 'select':       # same rule as tune: best mean rule-A BA over the 4 tuning transfers
    T = pd.read_csv(os.path.join(HERE, 's3d_tune.csv')); rows = []
    for t in T.itertuples():
        for x in (5e-4, 1e-3, 2e-3): rows.append(dict(group=t.group, src=t.src, tgt=t.tgt, lr=x, srsA=getattr(t, f'srsA_{x}'.replace('.', '_')) if False else T.loc[t.Index, f'srsA_{x}']))
    A = pd.concat([pd.DataFrame(rows), pd.read_csv(os.path.join(HERE, 's3d_tune_parts.csv'))[['group', 'src', 'tgt', 'lr', 'srsA']]])
    A = A.drop_duplicates(['group', 'src', 'tgt', 'lr']); n = A.groupby('lr').size(); assert (n == 4).all(), n
    m = A.groupby('lr').srsA.mean(); print('mean BA (rule A)', m.round(4).to_dict()); best = m.idxmax()
    open(os.path.join(HERE, 's3d_selected.txt'), 'w').write(repr(float(best))); print('SELECTED', best)
elif MODE == 'tune':
    TUNE = {'B88': [(2017, 2018), (2019, 2021)], 'A88': [(2018, 2020), (2022, 2024)]}; rows = []; t0 = time.time()
    for g, pairs in TUNE.items():
        G = NLData(g)
        for ya, yb in pairs:
            r, sel, _ = transfer(G, ya, yb, (5e-4, 1e-3, 2e-3), False); rows.append(dict(group=g, src=ya, tgt=yb, **r))
            print(g, ya, yb, {k: round(v, 3) for k, v in r.items()}, round(time.time() - t0), 's', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(HERE, 's3d_tune.csv'), index=False)
    T = pd.DataFrame(rows); m = T[[c for c in T.columns if c.startswith('srsA_')]].mean(); print('mean BA (rule A)', m.round(4).to_dict())
    best = m.idxmax()[5:]; open(os.path.join(HERE, 's3d_selected.txt'), 'w').write(best); print('SELECTED', best)
else:
    import psutil                                         # memory guard: extra workers bow out instead of exhausting RAM
    if psutil.virtual_memory().available < 3 * 2 ** 30: print('SKIP worker: available RAM < 3 GB', flush=True); sys.exit(0)
    lr = float(ARGS[3]); G = FRData() if GRP == 'FR' else NLData(GRP); OUT = os.path.join(HERE, f's3d_{GRP}.csv'); TR = os.path.join(HERE, f's3d_trace_{GRP}.csv')
    done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set(); t0 = time.time()
    CL = os.path.join(HERE, 's3d_claims'); os.makedirs(CL, exist_ok=True)
    def claim(name):                                      # lets several worker processes share one group without duplicates
        try: os.close(os.open(os.path.join(CL, name), os.O_CREAT | os.O_EXCL)); return True
        except FileExistsError: return False
    def locked_append(df, path):
        lk = path + '.lock'
        while not claim(os.path.basename(lk)): time.sleep(0.5)
        try: df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        finally: os.remove(os.path.join(CL, os.path.basename(lk)))
    for ya, yb in G.pairs:
        if (ya, yb) in done or not claim(f'{GRP}_{ya}_{yb}'): continue
        r, sel, tr = transfer(G, ya, yb, (lr,), GRP == 'FR')
        row = dict(group=GRP, src=ya, tgt=yb, srs_A=r[f'srsA_{lr}'], srs_B=r[f'srsB_{lr}'], srs_C=r[f'srsC_{lr}'], srs_D=r[f'srsD_{lr}'], lr=lr,
                   ep_A=';'.join(str(s['ep_A']) for s in sel), ep_C=';'.join(str(s['ep_C']) for s in sel), ep_D=';'.join(str(s['ep_D']) for s in sel))
        if GRP == 'FR': row['sfuda'] = r['sfuda']
        locked_append(pd.DataFrame([row]), OUT); locked_append(pd.DataFrame(tr), TR)
        print(GRP, ya, '->', yb, {k: round(v, 3) for k, v in row.items() if k.startswith(('srs', 'sfuda'))}, round(time.time() - t0), 's', flush=True)
    print('DONE', GRP)
