"""EXPLORE_S3A_DEEP: frozen TempCNN raw / AdaBN / DANN (db_run.py functions, unchanged) on all dev ordered year pairs, 2023 not read,
source TRAIN with xydrop as in P9-P11. Usage: python s3a_deep.py <grp> -> s3a_deep_<grp>.csv (resumable)."""
import os, sys, itertools, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(HERE, '..', 'multiyear_20261017', 'deep_baseline_2023')
GRP = sys.argv[1]; sys.argv = [os.path.join(DB, 'db_run.py'), GRP]; sys.path.insert(0, DB)
import db_run as D                                              # module-level: loads features for GRP; functions unchanged
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'multiyear_20261017'))
from xyexcl import drop as xydrop
DEVY = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]
OUT = os.path.join(HERE, f's3a_deep_{GRP}.csv')
done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set()
years = [y for y in D.YEARS if y in DEVY]; t0 = time.time()
for ya, yb in itertools.permutations(years, 2):
    if (ya, yb) in done: continue
    S = tuple(sorted(set(D.M[f'{ya}_T{D.TRK}']['slots']) & set(D.M[f'{yb}_T{D.TRK}']['slots'])))
    Xa, Ya, Fa = D.data(ya, S); Xb, Yb, Fb = D.data(yb, S)
    out = {v: np.full(len(Yb), -1) for v in ('dl_raw', 'dl_adabn', 'dl_dann')}
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~xydrop(GRP, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        mu = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
        norm = lambda X: np.nan_to_num((X - mu) / sd).astype(np.float32)
        Xs, Xt = norm(Xa[tr]), norm(Xb[te])
        net, XT = D.train(Xs, Ya[tr], Xt, False, D.SEED + k)
        out['dl_raw'][te] = D.predict(net, XT); out['dl_adabn'][te] = D.adabn_predict(net, XT)
        net2, _ = D.train(Xs, Ya[tr], Xt, True, D.SEED + k); out['dl_dann'][te] = D.predict(net2, XT)
    row = dict(group=GRP, src=ya, tgt=yb, n_slots=len(S), seed=D.SEED, **{v: D.bal_acc(Yb[p >= 0], p[p >= 0]) for v, p in out.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(GRP, ya, '->', yb, {v: round(row[v], 3) for v in out}, round(time.time() - t0), 's', flush=True)
print('DONE', GRP)
