"""Toy B (CHARTER_LINES_ZH.md §2.5): archive shift table for equation discovery.
One row per (group, src, tgt, cls, dtype, slot, sub): oracle shift (p1_rows target, mean over folds), source class mean
mu_src and its time derivative, acquisition doy in both years, KNMI weather features at both dates. 2023 not read.
Usage: python build_table.py -> shift_table.csv"""
import os, sys, json, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
sys.path.insert(0, UP)
from p1_tasks import load, DEV
KNMI = os.path.join(UP, '..', 'multiyear_20261017', 'knmi_daily_my.csv'); STN = {'A': 283, 'B': 280}

w = pd.read_csv(KNMI, parse_dates=['date'])
w['P'] = w.RH.clip(lower=0) / 10.0; w['E'] = w.EV24 / 10.0          # mm/day (RH = -1 means < 0.05 mm)


def wfeat(stn, dates):
    d = w[w.station_code == stn].set_index('date').sort_index(); P, E = d.P, d.E; out = []
    for t in pd.to_datetime(dates):
        r = {}
        for n in (1, 3, 7, 14, 30): r[f'P{n}'] = float(P.loc[t - pd.Timedelta(days=n - 1):t].sum())
        for n in (14, 30): r[f'E{n}'] = float(E.loc[t - pd.Timedelta(days=n - 1):t].sum())
        p60 = P.loc[t - pd.Timedelta(days=59):t].to_numpy()[::-1]; r['API'] = float((p60 * 0.9 ** np.arange(len(p60))).sum())
        out.append(r)
    return pd.DataFrame(out)


rows = pd.concat([pd.read_parquet(os.path.join(UP, f'p1_rows_{g}.parquet')) for g in ('A37', 'A88', 'B88')], ignore_index=True)
orc = rows.groupby(['group', 'src', 'tgt', 'cls', 'dtype', 'slot', 'sub'], as_index=False).agg(delta=('target', 'mean'), nfold=('target', 'count'))
out = []
for grp in ('A37', 'A88', 'B88'):
    site, trk, Z, M = load(grp); stats = {}
    for y in DEV:
        k = f'{y}_T{trk}'
        if k not in M: continue
        m = M[k]; y_ = Z[f'{k}_y']; dates = pd.to_datetime(m['dates'])
        recs = []
        for p, sl in enumerate(m['slots']):
            for ty, nm in ((0, 'VV'), (1, 'VH')):
                for sub in (0, 1): recs.append((ty, sl, sub, nm, 2 * p + sub, dates[2 * p + sub]))
            recs.append((2, sl, 0, 'COH', p, dates[2 * p] + (dates[2 * p + 1] - dates[2 * p]) / 2))
        X = {nm: Z[f'{k}_{nm}'].astype(float) for nm in ('VV', 'VH', 'COH')}
        W = wfeat(STN[site], [r[5] for r in recs])
        for c in sorted(set(y_.tolist())):
            for i, (ty, sl, sub, nm, col, dt) in enumerate(recs):
                stats[(y, c, ty, sl, sub)] = dict(mu=float(np.nanmean(X[nm][y_ == c, col])), doy=int(dt.dayofyear), n=int((y_ == c).sum()), **W.iloc[i].to_dict())
    S = pd.DataFrame([dict(year=k[0], cls=k[1], dtype=k[2], slot=k[3], sub=k[4], **v) for k, v in stats.items()])
    # d mu / dt within (year, cls, dtype), sorted by doy (dB/day; coherence/day)
    S = S.sort_values(['year', 'cls', 'dtype', 'doy']); S['dmu'] = np.nan
    for _, ix in S.groupby(['year', 'cls', 'dtype']).groups.items():
        g = S.loc[ix]; S.loc[ix, 'dmu'] = np.gradient(g.mu.to_numpy(), g.doy.to_numpy()) if len(g) > 1 else 0.0
    o = orc[orc.group == grp]
    a = o.merge(S.add_suffix('_s').rename(columns={'year_s': 'src', 'cls_s': 'cls', 'dtype_s': 'dtype', 'slot_s': 'slot', 'sub_s': 'sub'}), on=['src', 'cls', 'dtype', 'slot', 'sub'])
    a = a.merge(S.add_suffix('_t').rename(columns={'year_t': 'tgt', 'cls_t': 'cls', 'dtype_t': 'dtype', 'slot_t': 'slot', 'sub_t': 'sub'}), on=['tgt', 'cls', 'dtype', 'slot', 'sub'])
    a['site'] = site; out.append(a)
T = pd.concat(out, ignore_index=True)
T = T.drop(columns=['mu_t'])            # target-year class mean is the label; never an input
T.to_csv(os.path.join(HERE, 'shift_table.csv'), index=False)
print(T.shape); print(T.groupby(['group', 'dtype']).delta.describe().round(3)); print(T[['mu_s', 'dmu_s', 'doy_s', 'doy_t', 'API_s', 'API_t']].describe().round(3))
