"""S3i summary (EXPLORE_S3D_Q1_ZH.md, S3i). Merges s3i_curve_*.csv with the S3h end points (K=0: g_em, hgb_raw from s3h_cpu_*.csv;
K=all: g_lw_excl, hgb_pool_excl, hgb_poolsh_excl from s3h_cpu / s3h_poolx_cpu). Per region and K: median BA, paired comparisons
(median difference, share positive, year-block bootstrap 90% interval, 2000 draws, as s3h_summary.py). -> s3i_summary.csv, s3i_medians.csv"""
import os, glob
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(H, pat))) if 'smoke' not in f], ignore_index=True)
K3 = ['group', 'src', 'tgt']
C = rd('s3i_curve_*.csv').drop_duplicates(K3)
cpu = rd('s3h_cpu_*.csv').drop_duplicates(K3)[K3 + ['g_em', 'hgb_raw', 'g_lw_excl']]
px = rd('s3h_poolx_cpu_*.csv').drop_duplicates(K3)[K3 + ['hgb_pool_excl', 'hgb_poolsh_excl']]
T = C.merge(cpu, on=K3, how='left').merge(px, on=K3, how='left')
try:                                                     # amendment 2: TempCNN at K = 2; K = all end point from s3h_poolx_deep
    T = T.merge(rd('s3i_deep_*.csv').drop_duplicates(K3)[K3 + ['tcn_pool_k2', 'tcn_poolsh_k2']], on=K3, how='left')
    T = T.merge(rd('s3h_poolx_deep_*.csv').drop_duplicates(K3)[K3 + ['tcn_pool_excl', 'tcn_poolsh_excl']], on=K3, how='left')
except ValueError: pass
T = T.rename(columns={'g_em': 'g_lw_k0', 'hgb_raw': 'hgb_pool_k0', 'g_lw_excl': 'g_lw_kall', 'hgb_pool_excl': 'hgb_pool_kall',
                      'hgb_poolsh_excl': 'hgb_poolsh_kall', 'tcn_pool_excl': 'tcn_pool_kall', 'tcn_poolsh_excl': 'tcn_poolsh_kall'})
REG = {'NL site B': ['B88'], 'NL site A': ['A37', 'A88'], 'France': ['FR']}
KS = ['k0', 'k1', 'k2', 'k4', 'kall']
PAIRS = [(f'hgb_poolsh_{k}', f'hgb_pool_{k}') for k in ('k2', 'k4', 'kall')] + [(f'g_lw_{k}', f'hgb_pool_{k}') for k in ('k0', 'k2', 'k4', 'kall')] +         [(f'tcn_poolsh_{k}', f'tcn_pool_{k}') for k in ('k2', 'kall')]
rng = np.random.default_rng(20261112)


def pair(R, a, b):
    dd = (R[a] - R[b]); ok = dd.notna(); d = dd[ok].values
    if not len(d): return None
    Y = sorted(set(R.src) | set(R.tgt)); boot = []
    for _ in range(2000):
        ys = set(rng.choice(Y, len(Y), replace=True).tolist()); m = R.src.isin(ys) & R.tgt.isin(ys) & ok
        if m.sum() >= 5: boot.append(float(np.median(dd[m])))
    return dict(n=len(d), median=float(np.median(d)), mean=float(np.mean(d)), frac_pos=float((d > 0).mean()),
                lo90=float(np.percentile(boot, 5)) if boot else np.nan, hi90=float(np.percentile(boot, 95)) if boot else np.nan)


rows, med = [], []
for reg, gs in REG.items():
    R = T[T.group.isin(gs)]
    if not len(R): continue
    for k in KS:
        med.append(dict(region=reg, K=k, n=len(R), **{m: (R[f'{m}_{k}'].median() if f'{m}_{k}' in R else np.nan)
                                                      for m in ('hgb_pool', 'g_lw', 'hgb_poolsh', 'tcn_pool', 'tcn_poolsh')}))
    for a, b in PAIRS:
        if a in R and b in R:
            p = pair(R, a, b)
            if p: rows.append(dict(region=reg, a=a, b=b, **p))
S = pd.DataFrame(rows); S.to_csv(os.path.join(H, 's3i_summary.csv'), index=False)
M = pd.DataFrame(med); M.to_csv(os.path.join(H, 's3i_medians.csv'), index=False)
pd.set_option('display.width', 250)
print(M.round(4).to_string(index=False)); print(S.round(4).to_string(index=False))
