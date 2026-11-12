"""EXPLORE_P11D_DEEP: frozen-method resynthesis + TempCNN / DANN (db_run.train unchanged). Usage: python p11d_deep.py <grp> (resumable)"""
import os, sys, itertools, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(HERE, '..', 'multiyear_20261017', 'deep_baseline_2023')
GRP = sys.argv[1]; sys.argv = [os.path.join(DB, 'db_run.py'), GRP]; sys.path.insert(0, DB)
import db_run as D
sys.path.insert(0, os.path.join(HERE, '..', 'multiyear_20261017')); from xyexcl import drop as xydrop
sys.path.insert(0, os.path.join(HERE, 'frozen')); import hybridD_lw as F
sys.path.insert(0, HERE); from p9_lib import gen_em2
DEVY = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]
OUT = os.path.join(HERE, f'p11d_deep_{GRP}.csv')
ar = F.Archive.load(os.path.join(HERE, 'frozen', f"archive_NL_{'A' if GRP[0] == 'B' else 'B'}.npz"))


def tk_doy(ya, S):
    m = D.M[f'{ya}_T{D.TRK}']; dt = pd.to_datetime(m['dates']); tk, doy = [], []
    for ty in (0, 1):
        for s in S:
            p = m['slots'].index(s)
            for u in (0, 1): tk.append((ty, s, u)); doy.append(dt[2 * p + u].dayofyear)
    for s in S:
        p = m['slots'].index(s); tk.append((2, s, 0)); doy.append((dt[2 * p] + (dt[2 * p + 1] - dt[2 * p]) / 2).dayofyear)
    return tk, np.array(doy)


def flat(X):    # db_run seq (n, 4, 2S) -> E = [VV(2S), VH(2S), COH(S)] (COH repeated twice in seq)
    return np.hstack([X[:, 0], X[:, 1], X[:, 3][:, ::2]]).astype(float)


def shift_seq(X, y, delta, S):
    X = X.copy(); n2 = 2 * len(S)
    for c, d in delta.items():
        m = y == c; X[m, 0] += d[:n2]; X[m, 1] += d[n2:2 * n2]; X[m, 3] += np.repeat(d[2 * n2:], 2)
    X[:, 2] = X[:, 1] - X[:, 0]; return X.astype(np.float32)


done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set()
years = [y for y in D.YEARS if y in DEVY]; t0 = time.time()
for ya, yb in itertools.permutations(years, 2):
    if (ya, yb) in done: continue
    S = tuple(sorted(set(D.M[f'{ya}_T{D.TRK}']['slots']) & set(D.M[f'{yb}_T{D.TRK}']['slots'])))
    Xa, Ya, Fa = D.data(ya, S); Xb, Yb, Fb = D.data(yb, S); tk, doy = tk_doy(ya, S)
    Ea_all = flat(Xa); Ea = F.impute(Ea_all, Ea_all); Eb = F.impute(flat(Xb), Ea_all)
    out = {v: np.full(len(Yb), -1) for v in ('tcn_em', 'tcn_lw', 'dann_lw')}
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~xydrop(GRP, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        mu, prec, logdet, cls = F.gauss_fit(Ea[tr], Ya[tr]); m, Psi = ar.prior(cls, mu, tk, doy)
        _, d_lw = F.map_em(Eb[te], mu, prec, logdet, cls, m, Psi); _, d_em = gen_em2(Eb[te], mu, prec, logdet, cls, None, 0.1)
        for v, dl, dann in (('tcn_em', d_em, False), ('tcn_lw', d_lw, False), ('dann_lw', d_lw, True)):
            Xs_raw = shift_seq(np.nan_to_num(Xa[tr]), Ya[tr], dl, S)
            mu_ = Xs_raw.mean(axis=(0, 2), keepdims=True); sd_ = Xs_raw.std(axis=(0, 2), keepdims=True) + 1e-6
            norm = lambda X: np.nan_to_num((X - mu_) / sd_).astype(np.float32)
            net, XT = D.train(norm(Xs_raw), Ya[tr], norm(Xb[te]), dann, D.SEED + k); out[v][te] = D.predict(net, XT)
    row = dict(group=GRP, src=ya, tgt=yb, **{v: D.bal_acc(Yb[p >= 0], p[p >= 0]) for v, p in out.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(GRP, ya, '->', yb, {v: round(row[v], 3) for v in out}, round(time.time() - t0), 's', flush=True)
print('DONE', GRP)
