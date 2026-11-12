"""EXPLORE_P9 §4 archive-length curve: history subsets identical to P8 (rng 20261105; k3_0, k3_1, k5_0, k5_1, k8_all), test transfers
todo[0::2]. Variants hpj_cov01, phys_arch, learn_r*, hybrid_r* (r* re-derived per subset) + hybridD_r* (added, report only).
Usage: python p9_curve.py <test group> -> p9_curve_<grp>.csv"""
import os, sys, itertools, pickle, time
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, DEV
from p9_lib import gen_em2, dim_dates
from p9_basis import build_priors, task_priors, GROUPS
HERE = os.path.dirname(os.path.abspath(__file__))
CV = ['hpj_cov01', 'phys_arch', 'learn_r*', 'hybrid_r*', 'hybridD_r*']

if __name__ == '__main__':
    TG = sys.argv[1]; t0 = time.time(); rng = np.random.default_rng(20261105); trs = 'B' if TG[0] == 'A' else 'A'
    allr = pd.concat([pd.read_parquet(os.path.join(HERE, f'p1_rows_{g}.parquet')) for g in GROUPS], ignore_index=True)
    tr = allr[allr.site != TG[0]]; ty_all = sorted(set(tr.src))
    subsets = {f'k{k}_{i}': sorted(rng.choice(ty_all, k, replace=False).tolist()) for k in (3, 5) for i in (0, 1)}
    subsets['k8_all'] = ty_all; print('subsets', subsets, flush=True)
    arch = sum([pickle.load(open(os.path.join(HERE, f'p9_arch_{g}.pkl'), 'rb')) for g in GROUPS if g[0] == trs], [])
    PS = {nm: build_priors(tr[tr.src.isin(ys) & tr.tgt.isin(ys)], [a for a in arch if a['id'][1] in ys and a['id'][2] in ys], with_fixed=False) for nm, ys in subsets.items()}
    for nm, P in PS.items(): print(nm, 'n_arch', P['n_arch'], 'complete', P['n_complete'], 'r', P['learn']['rstar'][2], P['hand']['rstar'][2], P['disc']['rstar'][2], flush=True)
    site, trk, Z, M = load(TG); out = []
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    for ya, yb in list(itertools.permutations(years, 2))[0::2]:
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]; doy = np.array(dim_dates(M, trk, ya, S, ty, sl, sub).dayofyear)
        pred = {f'{nm}|{v}': np.full(len(Yb), -1) for nm in PS for v in CV}
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            mu, prec, logdet, cls = gauss_fit(Ea[trm], Ya[trm]); Et = Eb[tem]; C = len(cls)
            for nm, P in PS.items():
                PR = task_priors(P, CV, cls, mu, tk, ty, sl, sub, doy)
                for v in CV:
                    m, Psi, Pi = PR[v]
                    pred[f'{nm}|{v}'][tem] = gen_em2(Et, mu, prec, logdet, cls, (m, Psi if Psi is not None else np.zeros((C * D, C * D))), 0.1, Pinv=Pi)[0]
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        out.append(dict(group=TG, src=ya, tgt=yb, **ba)); print(TG, ya, '->', yb, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(out).to_csv(os.path.join(HERE, f'p9_curve_{TG}.csv'), index=False)
    print('DONE', TG)
