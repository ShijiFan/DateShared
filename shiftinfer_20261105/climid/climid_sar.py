"""ClimID-UDA (Wang et al., J. Remote Sensing 2025, DOI 10.34133/remotesensing.0439) ported to Sentinel-1 parcels; report-only baseline
(EXPLORE_S3D_Q1_ZH.md addendum). Paper configuration D4_C2_I11:
  - 4-day accumulation (sum over the 4 days before each acquisition date);
  - DBA barycenter (DTW barycenter averaging, EM); 6 variables (Tmean, Tmax, dew point, precipitation, solar radiation, surface pressure);
  - ClimVb(v) = barycenter over grid cells (3x3 points per AOI); ClimI = barycenter over the 6 min-max-normalised ClimVb;
  - additive ClimID cid(t) = ClimI_src(t) - ClimI_tgt(t), target series corrected ts'(t) = ts(t) + cid(t) (Eq. 3, 5).
SAR interface (declared): correction applied in source-standardised units (per feature for HGB, per channel for TempCNN), the analogue
of the paper's [0,1] reflectance; min-max normalisation of each variable over the union of source and target series.
Classifiers: frozen HGB (p1_tasks.H) and frozen TempCNN (db_run), each raw vs +ClimID, same folds / 400 m exclusion / transfers as S3c, S3d.
Usage: python climid_sar.py <A37|A88|B88|FR> -> climid_<grp>.csv (resumable)"""
import os, sys, json, itertools, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE); DB = os.path.join(UP, '..', 'multiyear_20261017', 'deep_baseline_2023')
GRP = sys.argv[1]
sys.argv = [os.path.join(DB, 'db_run.py'), 'B88']; sys.path.insert(0, DB); import db_run as D       # TempCNN / train / bal_acc only
sys.path.insert(0, UP); import p1_tasks as P1
sys.path.insert(0, os.path.join(UP, '..', 'multiyear_20261017')); from xyexcl import drop as nl_xydrop
hgb, assemble, impute = P1.H['hgb'], P1.H['assemble'], P1.impute
VARS = ['temperature_2m_mean', 'temperature_2m_max', 'dew_point_2m_mean', 'precipitation_sum', 'shortwave_radiation_sum', 'surface_pressure_mean']
DEVY = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]; ACC = 4


# ---------------- data ----------------
if GRP == 'FR':
    ROOT = r'D:\sar_data\fr_a'; Z = np.load(os.path.join(ROOT, 'features_FR.npz'), allow_pickle=True); M = json.load(open(os.path.join(ROOT, 'features_FR_meta.json')))
    XE = np.load(os.path.join(ROOT, 'xyexcl_FR.npz')); TAG = {int(t.split('_')[0]): t for t in M}; YEARS = sorted(TAG); PAIRS = list(itertools.permutations(YEARS, 2)); SITE = 'FR'
    def drop(ya, yb, k, n):
        m = np.zeros(n, bool)
        if int(ya) != int(yb): m[XE[f'{int(ya)}_{int(yb)}_{int(k)}']] = True
        return m
else:
    SITE, TRK, Z, M = P1.load(GRP); TAG = {y: f'{y}_T{TRK}' for y in DEVY if f'{y}_T{TRK}' in M}; YEARS = sorted(TAG)
    PAIRS = list(itertools.permutations(YEARS, 2))[0::2]; drop = lambda ya, yb, k, n: nl_xydrop(GRP, ya, yb, k, n)


def raw(y, S):
    m = M[TAG[y]]; pos = [m['slots'].index(s) for s in S]; dc = [c for p in pos for c in (2 * p, 2 * p + 1)]
    g = lambda l: Z[f'{TAG[y]}_{l}'].astype(float); dates = pd.to_datetime([m['dates'][c] for c in dc])
    return g('VV')[:, dc], g('VH')[:, dc], g('COH')[:, pos], Z[f'{TAG[y]}_y'], Z[f'{TAG[y]}_folds'], dates


# ---------------- ClimI / ClimID ----------------
W = pd.read_csv(os.path.join(HERE, f'era5land_{SITE}.csv'), parse_dates=['time'])
CELLS = {c: g.set_index('time')[VARS] for c, g in W.groupby('cell')}


def dtw_path(a, b):
    n, m = len(a), len(b); C = np.full((n + 1, m + 1), np.inf); C[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1): C[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    i, j, P = n, m, []
    while i > 0 and j > 0:
        P.append((i - 1, j - 1)); k = np.argmin([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]]); i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    return P


def dba(series, n_iter=10):          # DTW barycenter averaging, EM (paper's algorithm C2)
    S = np.asarray(series, float); b = S.mean(0)
    for _ in range(n_iter):
        acc = [[] for _ in range(len(b))]
        for s in S:
            for i, j in dtw_path(b, s): acc[i].append(s[j])
        nb = np.array([np.mean(a) for a in acc])
        if np.allclose(nb, b): break
        b = nb
    return b


def climvb(dates):                   # per variable: 4-day accumulation before each date, barycenter over cells
    out = {}
    for v in VARS:
        ser = [np.array([CELLS[c][v].loc[d - pd.Timedelta(days=ACC):d - pd.Timedelta(days=1)].sum() for d in dates]) for c in CELLS]
        out[v] = dba(ser)
    return out


def climid(dates_s, dates_t):
    Vs, Vt = climvb(dates_s), climvb(dates_t); ns, nt = [], []
    for v in VARS:
        lo, hi = min(Vs[v].min(), Vt[v].min()), max(Vs[v].max(), Vt[v].max()); r = hi - lo if hi > lo else 1.0
        ns.append((Vs[v] - lo) / r); nt.append((Vt[v] - lo) / r)
    return dba(ns) - dba(nt)         # cid_a(t) = ClimI_src(t) - ClimI_tgt(t)


# ---------------- run ----------------
OUT = os.path.join(HERE, f'climid_{GRP}.csv'); done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set(); t0 = time.time()
for ya, yb in PAIRS:
    if (ya, yb) in done: continue
    S = tuple(sorted(set(M[TAG[ya]]['slots']) & set(M[TAG[yb]]['slots'])))
    vva, vha, coa, Ya, Fa, da = raw(ya, S); vvb, vhb, cob, Yb, Fb, db = raw(yb, S); nd = vva.shape[1]
    cid = climid(da, db); cid_slot = cid.reshape(-1, 2).mean(1)
    Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
    split = lambda E: (E[:, :nd], E[:, nd:2 * nd], E[:, 2 * nd:]); cfull = np.concatenate([cid, cid, cid_slot])
    cube = lambda vv, vh, co: np.stack([vv, vh, vh - vv, np.repeat(co, 2, axis=1)], axis=1).astype(np.float32)
    Xa, Xb = cube(vva, vha, coa), cube(vvb, vhb, cob)
    out = {v: np.full(len(Yb), -1) for v in ('hgb_raw', 'hgb_climid', 'tcn_raw', 'tcn_climid')}
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~drop(ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        sd = Ea[tr].std(0) + 1e-6; m0 = hgb().fit(assemble(*split(Ea[tr])), Ya[tr])
        out['hgb_raw'][te] = m0.predict(assemble(*split(Eb[te]))); out['hgb_climid'][te] = m0.predict(assemble(*split(Eb[te] + cfull * sd)))
        mu = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); s0 = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
        n0 = lambda X: np.nan_to_num((X - mu) / s0).astype(np.float32); XT = n0(Xb[te])
        net, XTt = D.train(n0(Xa[tr]), Ya[tr], XT, False, D.SEED + k); out['tcn_raw'][te] = D.predict(net, XTt)
        out['tcn_climid'][te] = D.predict(net, D.torch.tensor(XT + cid[None, None, :].astype(np.float32), device=D.DEV))
    row = dict(group=GRP, src=ya, tgt=yb, cid_mean=float(cid.mean()), cid_absmax=float(np.abs(cid).max()),
               **{v: D.bal_acc(Yb[p >= 0], p[p >= 0]) for v, p in out.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(GRP, ya, '->', yb, {k: round(v, 3) for k, v in row.items() if k not in ('group', 'src', 'tgt')}, round(time.time() - t0), 's', flush=True)
print('DONE', GRP)
