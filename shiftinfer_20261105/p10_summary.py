"""EXPLORE_P10 summary -> P10_SUMMARY.json: reproducibility (P5, P9, P9 curve), criteria K1-K3, report K4 and curve medians."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); G = ['A37', 'A88', 'B88']
DIRS = {'testB88_priorA': ['B88'], 'testA_priorB': ['A37', 'A88']}
rd = lambda f: pd.concat([pd.read_csv(os.path.join(HERE, f.format(g))) for g in G if os.path.exists(os.path.join(HERE, f.format(g)))], ignore_index=True)
KEY = ['group', 'src', 'tgt']


def pair(df, a, b):
    d = (df[a] - df[b]).dropna()
    return dict(n=int(len(d)), median=round(float(d.median()), 4), mean=round(float(d.mean()), 4), frac_pos=round(float((d > 0).mean()), 3))


def bydir(df, a, b): return {dn: pair(df[df.group.isin(gs)], a, b) for dn, gs in DIRS.items()}


Mn = rd('p10_main_{}.csv'); Cv = rd('p10_curve_{}.csv'); out = {'n_main': Mn.groupby('group').size().to_dict(), 'n_curve': Cv.groupby('group').size().to_dict()}
p5 = rd('p5_cov_{}.csv').merge(Mn, on=KEY); p9 = rd('p9_basis_{}.csv').merge(Mn, on=KEY); p9c = rd('p9_curve_{}.csv').merge(Cv, on=KEY, suffixes=('_p9', ''))
out['repro'] = dict(hpj_vs_P5=float((p5['hpj_cov01'] - p5['k8_all|hpj_cov01']).abs().max()), hybridD_vs_P9=float((p9['hybridD_r*'] - p9['k8_all|hybridD_r*']).abs().max()),
                    curve_vs_P9={c: float((p9c[f'{c}_p9'] - p9c[c]).abs().max()) for nm in ('k3_0', 'k3_1', 'k5_0', 'k5_1', 'k8_all') for v in ('hpj_cov01', 'hybridD_r*') for c in [f'{nm}|{v}']})
for v in ('hpj_cov01', 'hybridD_r*', 'hybridD_lw'):
    for kk in ('k3', 'k5'): Cv[f'{kk}|{v}'] = Cv[[f'{kk}_{i}|{v}' for i in range(4)]].mean(1)
out['median_ba_main'] = {dn: {v: round(float(Mn[Mn.group.isin(gs)][f'k8_all|{v}'].median()), 4) for v in ('hpj_cov01', 'hybridD_r*', 'hybridD_lw')} for dn, gs in DIRS.items()}
out['median_ba_curve'] = {dn: {f'{kk}|{v}': round(float(Cv[Cv.group.isin(gs)][f'{kk}|{v}'].median()), 4) for kk in ('k3', 'k5', 'k8_all') for v in ('hpj_cov01', 'hybridD_r*', 'hybridD_lw')} for dn, gs in DIRS.items()}
k1 = bydir(Mn, 'k8_all|hybridD_lw', 'k8_all|hpj_cov01'); k2 = bydir(Cv, 'k3|hybridD_lw', 'k3|hpj_cov01'); k3 = bydir(Cv, 'k3|hybridD_lw', 'k8_all|hpj_cov01')
out['criteria'] = {
    'K1 long archive hybridD_lw vs hpj (main, all transfers)': dict(res=k1, pass_=all(x['median'] >= -0.003 for x in k1.values())),
    'K2 short archive k3 hybridD_lw vs hpj': dict(res=k2, pass_=all(x['median'] >= 0.003 and x['frac_pos'] >= 0.6 for x in k2.values())),
    'K3 hybridD_lw(k3) vs hpj(k8)': dict(res=k3, pass_=all(x['median'] >= -0.003 for x in k3.values()))}
out['report'] = {'K4 hybridD_lw vs hybridD_r*': {kk: bydir(Cv if kk != 'main' else Mn, f'{kk if kk != "main" else "k8_all"}|hybridD_lw', f'{kk if kk != "main" else "k8_all"}|hybridD_r*') for kk in ('k3', 'k5', 'k8_all', 'main')},
                 'k5 hybridD_lw vs hpj': bydir(Cv, 'k5|hybridD_lw', 'k5|hpj_cov01'),
                 'curve k8 hybridD_lw vs hpj': bydir(Cv, 'k8_all|hybridD_lw', 'k8_all|hpj_cov01'),
                 'per-draw k3 hybridD_lw vs hpj': {i: bydir(Cv, f'k3_{i}|hybridD_lw', f'k3_{i}|hpj_cov01') for i in range(4)}}
json.dump(out, open(os.path.join(HERE, 'P10_SUMMARY.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
