"""S3g: per-parcel predictions for the classification-map figure (report only; protocol EXPLORE_S3D_Q1_ZH.md, S3g).
Transfer per region fixed by the Fig. 1 rule, independent of any result: among seasons with a parcel raster, the ordered pair with the
largest increase in mean 1 Mar-30 Jun antecedent precipitation index (target - source). Pipelines identical to p11d_deep.py (NL) and
newregion/fr_deep.py (France): TempCNN without adaptation, DANN, TempCNN on the season resynthesised with the frozen-method shifts, and
the Gaussian classifier with the frozen-method shifts. Every eligible target parcel is predicted once, in the fold where it is TEST.
Usage: python s3g_mappred.py <B88|A88|FR>  -> s3g_pred_<grp>.csv, s3g_pick.json (appended)"""
import os, sys, json, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(HERE, '..', 'multiyear_20261017', 'deep_baseline_2023')
GRP = sys.argv[1]; ARGV_EXTRA = [int(v) for v in sys.argv[2:4]]; sys.argv = [os.path.join(DB, 'db_run.py'), 'B88' if GRP == 'FR' else GRP]; sys.path.insert(0, DB)
import db_run as D
sys.path.insert(0, os.path.join(HERE, '..', 'multiyear_20261017')); from xyexcl import drop as nl_xydrop
sys.path.insert(0, os.path.join(HERE, 'frozen')); import hybridD_lw as F
sys.path.insert(0, HERE); import p9_lib as P9


def api_season_fr(yr):
    RR = pd.read_csv(os.path.join(HERE, 'toy_B', 'full', 'frozen_j2', 'fr_rr_daily.csv'), parse_dates=['date']).set_index('date').RR.clip(lower=0)
    out = []
    for t in pd.date_range(f'{yr}-03-01', f'{yr}-06-30'):
        p = RR.loc[t - pd.Timedelta(days=59):t].to_numpy()[::-1]; out.append(float((p * 0.9 ** np.arange(len(p))).sum()))
    return float(np.mean(out))


if GRP == 'FR':
    import types; R = types.ModuleType('fr_deep_defs'); R.__file__ = os.path.join(HERE, 'newregion', 'fr_deep.py')   # definitions only, no run loop
    src = open(R.__file__, encoding='utf-8').read(); src = src[:src.index(chr(10) + 'done = {')]; exec(compile(src, R.__file__, 'exec'), R.__dict__)
    years = [y for y in R.YEARS if os.path.exists(rf'D:\sar_data\fr_a\FR\labels\parcel_raster_{y}.tif')]
    SA = {y: api_season_fr(y) for y in years}; data, tk_doy, drop, ar = R.data, R.tk_doy, R.xydrop, R.AR
    rowid = lambda y: R.Z[f'{R.TAG[y]}_row']
else:
    site = GRP[0]; ar = F.Archive.load(os.path.join(HERE, 'frozen', f"archive_NL_{'A' if site == 'B' else 'B'}.npz"))
    years = [y for y in D.YEARS if y in (2017, 2018, 2019, 2020, 2021, 2022) and os.path.exists(rf'D:\sar_data\multiyear\{site}\labels\parcel_raster_{y}.tif')]
    SA = {y: P9.season_api(site, y) for y in years}; data = D.data; drop = lambda s, t, k, n: nl_xydrop(GRP, s, t, k, n)
    rowid = lambda y: D.Z[f'{y}_T{D.TRK}_row']

    def tk_doy(ya, S):        # = p11d_deep.tk_doy
        m = D.M[f'{ya}_T{D.TRK}']; dt = pd.to_datetime(m['dates']); tk, doy = [], []
        for ty in (0, 1):
            for s in S:
                p = m['slots'].index(s)
                for u in (0, 1): tk.append((ty, s, u)); doy.append(dt[2 * p + u].dayofyear)
        for s in S:
            p = m['slots'].index(s); tk.append((2, s, 0)); doy.append((dt[2 * p] + (dt[2 * p + 1] - dt[2 * p]) / 2).dayofyear)
        return tk, np.array(doy)
flat = lambda X: np.hstack([X[:, 0], X[:, 1], X[:, 3][:, ::2]]).astype(float)


def shift_seq(X, y, delta, S):    # = p11d_deep.shift_seq
    X = X.copy(); n2 = 2 * len(S)
    for c, d in delta.items():
        m = y == c; X[m, 0] += d[:n2]; X[m, 1] += d[n2:2 * n2]; X[m, 3] += np.repeat(d[2 * n2:], 2)
    X[:, 2] = X[:, 1] - X[:, 0]; return X.astype(np.float32)


ya, yb = max(itertools.permutations(years, 2), key=lambda p: SA[p[1]] - SA[p[0]])
TAGOUT = ''
if len(ARGV_EXTRA) == 2: ya, yb = ARGV_EXTRA; TAGOUT = f'_{ya}_{yb}'          # S3g amendment: representative transfer given explicitly
slots = (lambda y: R.M[R.TAG[y]]['slots']) if GRP == 'FR' else (lambda y: D.M[f'{y}_T{D.TRK}']['slots'])
S = tuple(sorted(set(slots(ya)) & set(slots(yb)))); Xa, Ya, Fa = data(ya, S); Xb, Yb, Fb = data(yb, S); tk, doy = tk_doy(ya, S)
Ea_all = flat(Xa); Ea = F.impute(Ea_all, Ea_all); Eb = F.impute(flat(Xb), Ea_all)
out = {v: np.full(len(Yb), -1) for v in ('none', 'dann', 'prop_tcn', 'prop_gauss')}; fold = np.full(len(Yb), -1)
for k in range(6):
    tr = (Fa[:, k] == 'TRAIN') & ~drop(ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
    if not te.sum(): continue
    fold[te] = k
    mu, prec, logdet, cls = F.gauss_fit(Ea[tr], Ya[tr]); m, Psi = ar.prior(cls, mu, tk, doy)
    lab, d_lw = F.map_em(Eb[te], mu, prec, logdet, cls, m, Psi); out['prop_gauss'][te] = lab
    mu0 = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd0 = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
    n0 = lambda X: np.nan_to_num((X - mu0) / sd0).astype(np.float32)
    for v, dann in (('none', False), ('dann', True)):
        net, XT = D.train(n0(Xa[tr]), Ya[tr], n0(Xb[te]), dann, D.SEED + k); out[v][te] = D.predict(net, XT)
    Xs = shift_seq(np.nan_to_num(Xa[tr]), Ya[tr], d_lw, S); mu_ = Xs.mean(axis=(0, 2), keepdims=True); sd_ = Xs.std(axis=(0, 2), keepdims=True) + 1e-6
    n1 = lambda X: np.nan_to_num((X - mu_) / sd_).astype(np.float32)
    net, XT = D.train(n1(Xs), Ya[tr], n1(Xb[te]), False, D.SEED + k); out['prop_tcn'][te] = D.predict(net, XT)
P = pd.DataFrame(dict(row=rowid(yb), y=Yb, fold=fold, **out)); P.to_csv(os.path.join(HERE, f's3g_pred_{GRP}{TAGOUT}.csv'), index=False)
ok = P.fold >= 0; ba = {v: D.bal_acc(P.y[ok], P[v][ok]) for v in out}
rec = dict(group=GRP, rule='api' if not TAGOUT else 'median_gain', src=int(ya), tgt=int(yb), season_api={int(k): round(v, 2) for k, v in SA.items()}, n_pred=int(ok.sum()), ba=ba)
with open(os.path.join(HERE, 's3g_pick.json'), 'a') as f: f.write(json.dumps(rec) + '\n')
print(json.dumps(rec))
