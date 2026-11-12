"""EXPLORE_P9 summary -> P9_SUMMARY.json: reproducibility check vs P5, criteria H1-H5 (+H1D) exactly as written, diagnostics."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); G = ['A37', 'A88', 'B88']
DIRS = {'testB88_priorA': ['B88'], 'testA_priorB': ['A37', 'A88']}
rd = lambda f: pd.concat([pd.read_csv(os.path.join(HERE, f.format(g))) for g in G if os.path.exists(os.path.join(HERE, f.format(g)))], ignore_index=True)


def pair(df, a, b):
    d = (df[a] - df[b]).dropna()
    return dict(n=int(len(d)), median=round(float(d.median()), 4), mean=round(float(d.mean()), 4), frac_pos=round(float((d > 0).mean()), 3))


def bydir(df, a, b): return {dn: pair(df[df.group.isin(gs)], a, b) for dn, gs in DIRS.items()}


def grade(r):
    if all(x['median'] >= 0.01 and x['frac_pos'] >= 2 / 3 for x in r.values()): return 'better'
    if all(x['median'] >= -0.003 for x in r.values()): return 'non-inferior'
    return 'worse'


R = rd('p9_basis_{}.csv'); out = {'n_transfers': R.groupby('group').size().to_dict()}
P5 = rd('p5_cov_{}.csv').merge(R, on=['group', 'src', 'tgt'], suffixes=('_p5', ''))
out['repro_vs_P5'] = {v: float((P5[f'{v}_p5'] - P5[v]).abs().max()) for v in ('em_cov01', 'hpj_cov01')}
V = [c for c in R.columns if c not in ('group', 'src', 'tgt')]
out['median_ba'] = {dn: {v: round(float(R[R.group.isin(gs)][v].median()), 4) for v in V} for dn, gs in DIRS.items()}
out['vs_hpj_cov01'] = {v: bydir(R, v, 'hpj_cov01') for v in V if v != 'hpj_cov01'}
out['vs_em_cov01'] = {v: bydir(R, v, 'em_cov01') for v in V if v != 'em_cov01'}
crit = {}
h1 = bydir(R, 'hybrid_r*', 'hpj_cov01'); crit['H1 hybrid_r* vs hpj_cov01'] = dict(res=h1, grade=grade(h1))
h1d = bydir(R, 'hybridD_r*', 'hpj_cov01'); crit['H1D hybridD_r* vs hpj_cov01'] = dict(res=h1d, grade=grade(h1d))
h3 = bydir(R, 'phys_nom', 'em_cov01'); crit['H3 phys_nom vs em_cov01'] = dict(res=h3, pass_=all(x['median'] >= 0.005 and x['frac_pos'] >= 0.6 for x in h3.values()))
h4 = bydir(R, 'hybrid_r*', 'mllr_cov01'); crit['H4 hybrid_r* vs mllr_cov01'] = dict(res=h4, pass_=all(x['median'] >= 0.01 for x in h4.values()))
h5 = bydir(R, 'hybridD_r*_mw', 'hybridD_r*'); crit['H5 hybridD_mw vs hybridD'] = dict(res=h5, pass_=all(x['median'] >= 0.005 and x['frac_pos'] >= 0.6 for x in h5.values()))
if any(os.path.exists(os.path.join(HERE, f'p9_curve_{g}.csv')) for g in G):
    Cv = rd('p9_curve_{}.csv'); cur = {}
    for v in ('hybrid_r*', 'hybridD_r*', 'phys_arch', 'learn_r*'):
        Cv[f'k3|{v}'] = (Cv[f'k3_0|{v}'] + Cv[f'k3_1|{v}']) / 2; Cv[f'k5|{v}'] = (Cv[f'k5_0|{v}'] + Cv[f'k5_1|{v}']) / 2
    for kk in ('k3', 'k5'): Cv[f'{kk}|hpj_cov01'] = (Cv[f'{kk}_0|hpj_cov01'] + Cv[f'{kk}_1|hpj_cov01']) / 2
    for v in ('hybrid_r*', 'hybridD_r*', 'phys_arch', 'learn_r*'):
        cur[v] = {kk: bydir(Cv, f'{kk}|{v}', f'{kk}|hpj_cov01') for kk in ('k3', 'k5', 'k8_all')}
    out['curve_vs_hpj'] = cur
    out['curve_median_ba'] = {dn: {c: round(float(Cv[Cv.group.isin(gs)][c].median()), 4) for c in Cv.columns if '|' in c} for dn, gs in DIRS.items()}
    k3, k8 = cur['hybrid_r*']['k3'], cur['hybrid_r*']['k8_all']
    crit['H2 short archive hybrid_r* - hpj'] = dict(k3=k3, k8=k8, pass_=all(k3[d]['median'] > 0 and k3[d]['median'] > k8[d]['median'] for d in DIRS))
    k3d, k8d = cur['hybridD_r*']['k3'], cur['hybridD_r*']['k8_all']
    crit['(report) H2 analogue for hybridD'] = dict(k3=k3d, k8=k8d, pass_=all(k3d[d]['median'] > 0 and k3d[d]['median'] > k8d[d]['median'] for d in DIRS))
out['criteria'] = crit
Dg = rd('p9_delta_{}.csv'); Dg['size'] = pd.cut(Dg.psrc, [0, 0.025, 0.15, 1], labels=['small<=2.5%', 'mid', 'large>=15%'])
out['delta_rmse_bs'] = {dn: Dg[Dg.group.isin(gs)].groupby(['size', 'var'], observed=True).rmse_bs.median().unstack().round(3).to_dict() for dn, gs in DIRS.items()}
Cf = rd('p9_coef_{}.csv'); w = Cf[Cf.kind == 'wbs'].copy(); w['dapi'] = w.api_t - w.api_s; w['dlnapi'] = np.log1p(w.api_t / 7.0) - np.log1p(w.api_s / 7.0)
out['coef_weather'] = {dn: dict(spearman_w_dapi=round(float(spearmanr(x.coef, x.dapi)[0]), 3),
                                spearman_w_dlnapi=round(float(spearmanr(x.coef, x.dlnapi)[0]), 3), n=int(len(x)),
                                spearman_transfer_meanw_seasonapi=round(float(spearmanr(*x.groupby(['group', 'src', 'tgt']).agg(m=('coef', 'mean'), s=('season_dapi', 'first')).to_numpy().T)[0]), 3))
                       for dn, gs in DIRS.items() for x in [w[w.group.isin(gs)]]}
t = Cf[Cf.kind == 'tau']; out['tau_days'] = {dn: t[t.group.isin(gs)].groupby('cls').coef.describe()[['50%', 'std']].round(2).to_dict() for dn, gs in DIRS.items()}
json.dump(out, open(os.path.join(HERE, 'P9_SUMMARY.json'), 'w'), indent=1, default=str)
print(json.dumps(dict(n=out['n_transfers'], repro=out['repro_vs_P5'], criteria={k: v.get('grade', v.get('pass_')) for k, v in crit.items()}), indent=1))
print(json.dumps(out['median_ba'], indent=1))
