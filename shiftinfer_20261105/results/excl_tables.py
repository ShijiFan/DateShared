"""S3j (EXPLORE_S3D_Q1_ZH.md): Dutch single-source results with the archive WITHOUT the transfer's seasons -> results/EXCL.json.
Replaces the archive-based entries of CONSOLIDATED.json / BASELINES.json / P11_SUMMARY.json / P11D_SUMMARY.json used by tab_nl, tab_plugin
and fig5_severity; archive-free columns are read unchanged from the original files (p9_basis em_cov01/mllr_cov01, s3a_deep dl_*,
p11_c2/p11_c2b hgb_raw/hgb_em/rf_raw/rf_em, p11d tcn_em). Severity = terciles of dl_raw per test direction (as compare_baselines.py)."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE); G = ['A37', 'A88', 'B88']; KEY = ['group', 'src', 'tgt']
DIRS = {'testB88_priorA': ['B88'], 'testA_priorB': ['A37', 'A88']}
rd = lambda pat: pd.concat([pd.read_csv(os.path.join(UP, pat.format(g))) for g in G if os.path.exists(os.path.join(UP, pat.format(g)))], ignore_index=True)


def pair(df, a, b):
    d = (df[a] - df[b]).dropna(); return dict(n=int(len(d)), median=round(float(d.median()), 4), frac_pos=round(float((d > 0).mean()), 3))


X = rd('s3j_cpu_{}.csv').drop_duplicates(KEY); XD = rd('s3j_deep_{}.csv').drop_duplicates(KEY)
T = (rd('p9_basis_{}.csv')[KEY + ['em_cov01', 'mllr_cov01']].merge(X, on=KEY).merge(rd('s3a_deep_{}.csv')[KEY + ['dl_raw', 'dl_adabn', 'dl_dann']], on=KEY, how='left'))
c2 = rd('p11_c2_{}.csv')[KEY + ['hgb_raw', 'hgb_em']]
rf = pd.concat([rd('p11_c2_{}.csv')[KEY + ['rf_raw', 'rf_em']].dropna(), rd('p11_c2b_{}.csv')[KEY + ['rf_raw', 'rf_em']].dropna()]).drop_duplicates(KEY)
P = (X.merge(c2, on=KEY).merge(rf, on=KEY, how='left').merge(rd('p11d_deep_{}.csv')[KEY + ['tcn_em']], on=KEY)
      .merge(XD, on=KEY, how='left').merge(rd('s3a_deep_{}.csv')[KEY + ['dl_raw', 'dl_dann']], on=KEY, how='left'))
out = {'n': {'tab_nl': int(len(T)), 'plugin': int(len(P)), 'deep': int(P.tcn_lw_x.notna().sum()), 'rf': int(P.rf_lw_x.notna().sum())}}
for dn, gs in DIRS.items():
    d = T[T.group.isin(gs)]; p = P[P.group.isin(gs)]
    o = {'tab_nl': {m: dict(median_ba=round(float(d[m].median()), 4), diff=(None if m == 'hybridD_lw_x' else pair(d, 'hybridD_lw_x', m)))
                    for m in ['dl_raw', 'dl_adabn', 'dl_dann', 'em_cov01', 'mllr_cov01', 'hpj_cov01_x', 'hybridD_r*_x', 'hybridD_lw_x']}}
    o['plugin'] = {m: round(float(p[m].median()), 4) for m in ['hgb_raw', 'hgb_em', 'hgb_lw_x', 'rf_raw', 'rf_em', 'rf_lw_x', 'dl_raw', 'tcn_em', 'tcn_lw_x', 'dl_dann', 'dann_lw_x']}
    o['plugin_pairs'] = {f'{a} - {b}': pair(p, a, b) for a, b in [('hgb_lw_x', 'hgb_em'), ('rf_lw_x', 'rf_em'), ('tcn_lw_x', 'tcn_em'), ('tcn_lw_x', 'dl_dann'),
                                                                   ('dann_lw_x', 'dl_dann')]}
    q = pd.qcut(T[T.group.isin(gs)].set_index(KEY)['dl_raw'], 3, labels=['severe', 'mid', 'mild'])
    pp = p.set_index(KEY).join(q.rename('sev')); dd = d.set_index(KEY).join(q.rename('sev'))
    o['by_severity'] = {lv: {'lw_x - dl_dann': pair(dd[dd.sev == lv], 'hybridD_lw_x', 'dl_dann'), 'dann_lw_x - dl_dann': pair(pp[pp.sev == lv], 'dann_lw_x', 'dl_dann')}
                        for lv in ('severe', 'mid', 'mild')}
    out[dn] = o
json.dump(out, open(os.path.join(HERE, 'EXCL.json'), 'w'), indent=1)
print(json.dumps(out, indent=1)[:3000])
