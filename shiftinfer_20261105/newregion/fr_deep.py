"""FR-A deep baselines (report only; prereg §9 2026-11-07): frozen db_run TempCNN raw / AdaBN / DANN and frozen-method resynthesis + TempCNN / DANN
(as P11d), on the 56 FR transfers with the FR cross-year 400 m exclusion. Usage: python fr_deep.py -> FR_DEEP_rows.csv (resumable)"""
import os, sys, json, itertools, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE); DB = os.path.join(UP, '..', 'multiyear_20261017', 'deep_baseline_2023')
sys.argv = [os.path.join(DB, 'db_run.py'), 'B88']; sys.path.insert(0, DB)
import db_run as D                                            # functions only; its module-level NL data are not used
for p in (UP, os.path.join(UP, 'frozen'), os.path.join(HERE, 'pipeline')): sys.path.insert(0, p)
import hybridD_lw as F
from p9_lib import gen_em2
_X = np.load(r'D:\sar_data\fr_a\xyexcl_FR.npz')


def xydrop(s, t, k, n):      # identical to fr_xyexcl.drop (inlined: the GPU env has no geopandas)
    m = np.zeros(n, bool)
    if int(s) != int(t): m[_X[f'{int(s)}_{int(t)}_{int(k)}']] = True
    return m
ROOT = r'D:\sar_data\fr_a'; Z = np.load(os.path.join(ROOT, 'features_FR.npz'), allow_pickle=True); M = json.load(open(os.path.join(ROOT, 'features_FR_meta.json')))
TAG = {int(t.split('_')[0]): t for t in M}; YEARS = sorted(TAG); AR = F.Archive.load(os.path.join(UP, 'frozen', 'archive_NL_all.npz'))
OUT = os.path.join(HERE, 'FR_DEEP_rows.csv')


def data(y, S):      # same layout as db_run.data: (n, 4, 2S) = VV, VH, VH-VV, COH repeated
    m = M[TAG[y]]; pos = [m['slots'].index(s) for s in S]; dc = [c for p in pos for c in (2 * p, 2 * p + 1)]
    g = lambda l: Z[f'{TAG[y]}_{l}'].astype(np.float32); vv, vh, co = g('VV')[:, dc], g('VH')[:, dc], g('COH')[:, pos]
    return np.stack([vv, vh, vh - vv, np.repeat(co, 2, axis=1)], axis=1), Z[f'{TAG[y]}_y'], Z[f'{TAG[y]}_folds']


def tk_doy(y, S):
    m = M[TAG[y]]; dt = pd.to_datetime(m['dates']); tk, doy = [], []
    for ty in (0, 1):
        for s in S:
            p = m['slots'].index(s)
            for u in (0, 1): tk.append((ty, s, u)); doy.append(dt[2 * p + u].dayofyear)
    for s in S:
        p = m['slots'].index(s); tk.append((2, s, 0)); doy.append((dt[2 * p] + (dt[2 * p + 1] - dt[2 * p]) / 2).dayofyear)
    return tk, np.array(doy)


flat = lambda X: np.hstack([X[:, 0], X[:, 1], X[:, 3][:, ::2]]).astype(float)


def shift_seq(X, y, delta, S):
    X = X.copy(); n2 = 2 * len(S)
    for c, d in delta.items():
        m = y == c; X[m, 0] += d[:n2]; X[m, 1] += d[n2:2 * n2]; X[m, 3] += np.repeat(d[2 * n2:], 2)
    X[:, 2] = X[:, 1] - X[:, 0]; return X.astype(np.float32)


done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set(); t0 = time.time()
for ya, yb in itertools.permutations(YEARS, 2):
    if (ya, yb) in done: continue
    S = tuple(sorted(set(M[TAG[ya]]['slots']) & set(M[TAG[yb]]['slots']))); Xa, Ya, Fa = data(ya, S); Xb, Yb, Fb = data(yb, S); tk, doy = tk_doy(ya, S)
    Ea_all = flat(Xa); Ea = F.impute(Ea_all, Ea_all); Eb = F.impute(flat(Xb), Ea_all)
    out = {v: np.full(len(Yb), -1) for v in ('dl_raw', 'dl_adabn', 'dl_dann', 'tcn_em', 'tcn_lw', 'dann_lw')}
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~xydrop(ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        mu0 = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd0 = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
        n0 = lambda X: np.nan_to_num((X - mu0) / sd0).astype(np.float32)
        net, XT = D.train(n0(Xa[tr]), Ya[tr], n0(Xb[te]), False, D.SEED + k); out['dl_raw'][te] = D.predict(net, XT); out['dl_adabn'][te] = D.adabn_predict(net, XT)
        net2, _ = D.train(n0(Xa[tr]), Ya[tr], n0(Xb[te]), True, D.SEED + k); out['dl_dann'][te] = D.predict(net2, XT)
        mu, prec, logdet, cls = F.gauss_fit(Ea[tr], Ya[tr]); m, Ps = AR.prior(cls, mu, tk, doy)
        _, d_lw = F.map_em(Eb[te], mu, prec, logdet, cls, m, Ps); _, d_em = gen_em2(Eb[te], mu, prec, logdet, cls, None, 0.1)
        for v, dl, dann in (('tcn_em', d_em, False), ('tcn_lw', d_lw, False), ('dann_lw', d_lw, True)):
            Xs = shift_seq(np.nan_to_num(Xa[tr]), Ya[tr], dl, S); m_ = Xs.mean(axis=(0, 2), keepdims=True); s_ = Xs.std(axis=(0, 2), keepdims=True) + 1e-6
            nn_ = lambda X: np.nan_to_num((X - m_) / s_).astype(np.float32)
            net3, XT3 = D.train(nn_(Xs), Ya[tr], nn_(Xb[te]), dann, D.SEED + k); out[v][te] = D.predict(net3, XT3)
    row = dict(src=ya, tgt=yb, **{v: D.bal_acc(Yb[p >= 0], p[p >= 0]) for v, p in out.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(ya, '->', yb, round(time.time() - t0), 's', flush=True)
print('DONE')
