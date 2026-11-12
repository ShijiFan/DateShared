"""EXPLORE_P11_C2 summary -> P11_SUMMARY.json (criteria C2a, C2b, C2c, C3-HGB as written; report-only comparisons)."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); G = ['A37', 'A88', 'B88']
DIRS = {'testB88_priorA': ['B88'], 'testA_priorB': ['A37', 'A88']}
KEY = ['group', 'src', 'tgt']
rdf = lambda pat: pd.concat([pd.read_csv(os.path.join(HERE, pat.format(g))) for g in G if os.path.exists(os.path.join(HERE, pat.format(g)))], ignore_index=True)
R = rdf('p11_c2_{}.csv').set_index(KEY)
if any(os.path.exists(os.path.join(HERE, f'p11_c2b_{g}.csv')) for g in G): R = R.combine_first(rdf('p11_c2b_{}.csv').set_index(KEY))   # phase 2 columns
R = R.reset_index()


def pair(df, a, b):
    d = (df[a] - df[b]).dropna()
    if not len(d): return dict(n=0, median=float('nan'), mean=float('nan'), frac_pos=float('nan'))
    return dict(n=int(len(d)), median=round(float(d.median()), 4), mean=round(float(d.mean()), 4), frac_pos=round(float((d > 0).mean()), 3))


def bydir(a, b): return {dn: pair(R[R.group.isin(gs)], a, b) for dn, gs in DIRS.items()}


for m in ('hpj', 'lw'):
    for i in (0, 1):
        if f'hgb_{m}_k3_{i}' not in R: R[f'hgb_{m}_k3_{i}'] = np.nan
    R[f'hgb_{m}_k3'] = R[[f'hgb_{m}_k3_0', f'hgb_{m}_k3_1']].mean(axis=1, skipna=False)
for c in ('rf_raw', 'rf_em', 'rf_lw'):
    if c not in R: R[c] = np.nan
c2a, c2b, c2c, c3 = bydir('hgb_lw', 'hgb_em'), bydir('hgb_lw', 'hgb_hpj'), bydir('rf_lw', 'rf_em'), bydir('hgb_lw_k3', 'hgb_hpj_k3')
out = dict(n=R.groupby('group').size().to_dict(), criteria={
    'C2a hgb_lw vs hgb_em': dict(res=c2a, pass_=all(x['median'] >= 0.01 and x['frac_pos'] >= 2 / 3 for x in c2a.values())),
    'C2b hgb_lw vs hgb_hpj': dict(res=c2b, pass_=all(x['median'] >= -0.003 for x in c2b.values())),
    'C2c rf_lw vs rf_em': dict(res=c2c, pass_=all(x['median'] >= 0.005 and x['frac_pos'] >= 0.6 for x in c2c.values())),
    'C3-HGB k3 hgb_lw vs hgb_hpj': dict(res=c3, pass_=all(x['median'] >= 0.003 and x['frac_pos'] >= 0.6 for x in c3.values()))},
    report={'hgb_em vs hgb_raw': bydir('hgb_em', 'hgb_raw'), 'hgb_lw vs hgb_raw': bydir('hgb_lw', 'hgb_raw'), 'hgb_oracle vs hgb_raw': bydir('hgb_oracle', 'hgb_raw'),
            'rf_lw vs rf_raw': bydir('rf_lw', 'rf_raw'), 'hgb_lw_k3 vs hgb_lw (k8)': bydir('hgb_lw_k3', 'hgb_lw'), 'hgb_lw_k3 vs hgb_hpj (k8)': bydir('hgb_lw_k3', 'hgb_hpj')},
    median_ba={dn: {c: round(float(R[R.group.isin(gs)][c].median()), 4) for c in R.columns if c.startswith(('hgb', 'rf'))} for dn, gs in DIRS.items()})
for dn in DIRS:
    mb = out['median_ba'][dn]; gap = mb['hgb_oracle'] - mb['hgb_raw']
    out['median_ba'][dn]['recovery_em'] = round((mb['hgb_em'] - mb['hgb_raw']) / gap, 3); out['median_ba'][dn]['recovery_lw'] = round((mb['hgb_lw'] - mb['hgb_raw']) / gap, 3)
json.dump(out, open(os.path.join(HERE, 'P11_SUMMARY.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
