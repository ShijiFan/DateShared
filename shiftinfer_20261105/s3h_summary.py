"""S3h summary (EXPLORE_S3D_Q1_ZH.md, S3h). Merges s3h_cpu_*.csv / s3h_deep_*.csv (all parts) and reports per region (NL site B = B88,
NL site A = A37 + A88, France) median BA and paired comparisons: median difference, share positive, year-block bootstrap 90% interval
(2000 draws, resample years and keep transfers whose two seasons were both drawn; as newregion/confirm_fr_summary.py).
Also checks the recomputed tcn_lw against p11d_deep_<g>.csv / newregion/FR_DEEP_rows.csv. -> s3h_summary.csv, s3h_medians.csv"""
import os, glob
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
rd = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(H, pat))) if 'smoke' not in f], ignore_index=True)
cpu = rd('s3h_cpu_*.csv').drop_duplicates(['group', 'src', 'tgt'])
try:
    deep = rd('s3h_deep_*.csv').drop_duplicates(['group', 'src', 'tgt'])
    T = cpu.merge(deep.drop(columns=['n_pool_rows']), on=['group', 'src', 'tgt'], how='outer')
except ValueError:
    deep = None; T = cpu
for kind in ('cpu', 'deep'):                             # S3h amendment (2026-11-11 12:10): pooled training without the transfer's seasons
    try:
        px = rd(f's3h_poolx_{kind}_*.csv').drop_duplicates(['group', 'src', 'tgt']).drop(columns=['n_pool_rows', 'n_pool_seasons'])
        T = T.merge(px, on=['group', 'src', 'tgt'], how='left')
    except ValueError: pass
REG = {'NL site B': ['B88'], 'NL site A': ['A37', 'A88'], 'France': ['FR']}
PAIRS = [('g_lw', 'g_em'), ('g_lw', 'g_lw_excl'), ('g_lw', 'g_notau'), ('g_lw', 'g_nodate'), ('g_lw', 'g_shuffle'),
         ('g_lw', 'hgb_pool'), ('hgb_lw', 'hgb_pool'), ('hgb_poolsh', 'hgb_pool'), ('hgb_pool', 'hgb_raw'), ('hgb_lw', 'hgb_raw'),
         ('tcn_lw', 'tcn_pool'), ('tcn_poolsh', 'tcn_pool'), ('tcn_pool', 'hgb_pool'), ('g_lw', 'tcn_pool'),
         ('g_lw_excl', 'hgb_pool_excl'), ('hgb_lw', 'hgb_pool_excl'), ('hgb_poolsh_excl', 'hgb_pool_excl'), ('hgb_pool', 'hgb_pool_excl'),
         ('tcn_lw', 'tcn_pool_excl'), ('tcn_poolsh_excl', 'tcn_pool_excl'), ('tcn_pool', 'tcn_pool_excl'), ('g_lw_excl', 'tcn_pool_excl')]
rng = np.random.default_rng(20261111)


def pair(R, a, b):
    dd = (R[a] - R[b]); ok = dd.notna(); d = dd[ok].values
    if not len(d): return None
    Y = sorted(set(R.src) | set(R.tgt)); boot = []
    for _ in range(2000):
        ys = set(rng.choice(Y, len(Y), replace=True).tolist()); m = R.src.isin(ys) & R.tgt.isin(ys) & ok
        if m.sum() >= 5: boot.append(float(np.median(dd[m])))
    return dict(n=len(d), median=float(np.median(d)), mean=float(np.mean(d)), frac_pos=float((d > 0).mean()),
                lo90=float(np.percentile(boot, 5)), hi90=float(np.percentile(boot, 95)))


rows, med = [], []
for reg, gs in REG.items():
    R = T[T.group.isin(gs)]
    if not len(R): continue
    med.append(dict(region=reg, n=len(R), **{c: R[c].dropna().median() for c in R.columns if c.startswith(('g_', 'hgb_', 'tcn_'))}))
    for a, b in PAIRS:
        if a in R and b in R:
            p = pair(R, a, b)
            if p: rows.append(dict(region=reg, a=a, b=b, **p))
S = pd.DataFrame(rows); S.to_csv(os.path.join(H, 's3h_summary.csv'), index=False)
M = pd.DataFrame(med); M.to_csv(os.path.join(H, 's3h_medians.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(M.round(4).to_string(index=False)); print(S.round(4).to_string(index=False))
if deep is not None:                                     # internal check: recomputed tcn_lw vs stored
    ref = pd.concat([pd.read_csv(os.path.join(H, f'p11d_deep_{g}.csv')) for g in ('B88', 'A37', 'A88')] +
                    [pd.read_csv(os.path.join(H, 'newregion', 'FR_DEEP_rows.csv')).assign(group='FR')])[['group', 'src', 'tgt', 'tcn_lw']]
    c = deep.merge(ref, on=['group', 'src', 'tgt'], suffixes=('', '_ref'))
    print('tcn_lw recomputed vs stored: n', len(c), 'max |diff|', float((c.tcn_lw - c.tcn_lw_ref).abs().max()),
          'median diff', float((c.tcn_lw - c.tcn_lw_ref).median()))
