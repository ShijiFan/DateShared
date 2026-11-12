"""Summarise the archive-length curve (curve_b.py) together with p9_curve.py (hybridD_r*, hpj): per archive length L in
{3, 5, 8} (k3 = mean of k3_0/k3_1 per transfer, k5 likewise), per test direction, median BA and paired differences
vs hpj_cov01 at the same L (median, share positive). H2-style check as in EXPLORE_P9: gain at L=3 > 0 and > gain at L=8."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE))
V = ['hpj_cov01', 'j2_hpj_wc_cond', 'k1_decay_cond', 'hybridD_r*']
L = {'3': ('k3_0', 'k3_1'), '5': ('k5_0', 'k5_1'), '8': ('k8_all',)}

rows = []
for g in ('A37', 'A88', 'B88'):
    a = pd.read_csv(os.path.join(HERE, 'runs', f'curve_{g}.csv')); b = pd.read_csv(os.path.join(UP, f'p9_curve_{g}.csv'))
    m = a.merge(b[['src', 'tgt'] + [c for c in b.columns if c.endswith('|hybridD_r*')]], on=['src', 'tgt'])
    for _, r in m.iterrows():
        for l, subs in L.items():
            rows.append(dict(group=g, dir='testA' if g[0] == 'A' else 'testB', src=r.src, tgt=r.tgt, L=l, **{v: np.mean([r[f'{s}|{v}'] for s in subs]) for v in V}))
D = pd.DataFrame(rows); out = []
for (dr, l), d in D.groupby(['dir', 'L']):
    o = dict(dir=dr, L=l, n=len(d), **{f'BA_{v}': round(d[v].median(), 4) for v in V})
    for v in V[1:]:
        x = d[v] - d['hpj_cov01']; o[f'{v}-hpj'] = f'{x.median():+.4f} ({(x > 0).mean():.2f})'
    x = d['j2_hpj_wc_cond'] - d['hybridD_r*']; o['j2-hybridD'] = f'{x.median():+.4f} ({(x > 0).mean():.2f})'
    out.append(o)
T = pd.DataFrame(out); pd.set_option('display.width', 250); print(T.to_string(index=False))
for v in V[1:]:
    g3 = {dr: (d[d.L == '3'][v] - d[d.L == '3']['hpj_cov01']).median() for dr, d in D.groupby('dir')}
    g8 = {dr: (d[d.L == '8'][v] - d[d.L == '8']['hpj_cov01']).median() for dr, d in D.groupby('dir')}
    print(v, 'H2-style (gain3>0 and gain3>gain8 in both dirs):', all(g3[k] > 0 and g3[k] > g8[k] for k in g3), {k: (round(g3[k], 4), round(g8[k], 4)) for k in g3})
print(D.groupby(['dir', 'L']).size().to_dict())
T.to_csv(os.path.join(HERE, 'runs', 'curve_summary.csv'), index=False)
