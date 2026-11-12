"""Archive-length curve for the weather-conditioned joint prior (user decision 2026-11-05, e:/v.txt): identical history
subsets and test transfers as p9_curve.py (rng 20261105; k3_0, k3_1, k5_0, k5_1, k8_all; todo[0::2]). For every subset the
archive (LW joint prior, weather-equation parameters, conditioned LW) is built from that subset's archive tasks only.
Variants: hpj_cov01 (must equal p9_curve), j2_hpj_wc_cond (main), k1_decay_cond (appendix).
Usage: python curve_b.py <test group> -> runs/curve_<grp>.csv"""
import os, sys, itertools, time
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from evalba import (UP, OUT, GROUPS, archive, fit_joint, joint_prior, make_ctx, load, data, impute, gauss_fit, xydrop, dim_meta,
                    dim_dates, gen_em2, joint_for, DEV)
from cands_r3 import CANDS as R3
CV = [c for c in R3 if c['name'] in ('j2_hpj_wc_cond', 'k1_decay_cond')]

if __name__ == '__main__':
    TG = sys.argv[1]; t0 = time.time(); rng = np.random.default_rng(20261105); trs = 'B' if TG[0] == 'A' else 'A'
    allr = pd.concat([pd.read_parquet(os.path.join(UP, f'p1_rows_{g}.parquet')) for g in GROUPS], ignore_index=True)
    tr = allr[allr.site != TG[0]]; ty_all = sorted(set(tr.src))
    subsets = {f'k{k}_{i}': sorted(rng.choice(ty_all, k, replace=False).tolist()) for k in (3, 5) for i in (0, 1)}
    subsets['k8_all'] = ty_all; print('subsets', subsets, flush=True)
    PS = {}
    for nm, ys in subsets.items():
        P0, A = archive(trs, allr, ys); fit = {}
        for cd in CV:
            gs = cd['fit'](A, P0); fit[cd['name']] = (gs, fit_joint(cd, P0, A, gs))
            print(nm, cd['name'], 'n_arch', len(A), 'theta', np.round(gs['th'], 3).tolist(), 'n_complete_cond', fit[cd['name']][1].get('n_complete'), flush=True)
        PS[nm] = (P0, fit)
    site, trk, Z, M = load(TG); out = []
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    for ya, yb in list(itertools.permutations(years, 2))[0::2]:
        S_ = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S_); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S_)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S_); tk = list(zip(ty, sl, sub)); doy = np.array(dim_dates(M, trk, ya, S_, ty, sl, sub).dayofyear)
        names = ['hpj_cov01'] + [c['name'] for c in CV]
        pred = {f'{nm}|{v}': np.full(len(Yb), -1) for nm in PS for v in names}
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            mu, prec, logdet, cls = gauss_fit(Ea[trm], Ya[trm]); Et = Eb[tem]
            for nm, (P0, fit) in PS.items():
                m, Psi = joint_for(P0['keys'], P0['jm'], P0['jc'], cls, tk)
                pred[f'{nm}|hpj_cov01'][tem] = gen_em2(Et, mu, prec, logdet, cls, (m, Psi), 0.1)[0]
                ctx = make_ctx(TG, M, trk, ya, yb, cls, mu, ty, sl, sub, tk, doy, P0)
                for cd in CV:
                    gs, S = fit[cd['name']]; m, Psi = joint_prior(cd, ctx, gs, S, P0)
                    pred[f"{nm}|{cd['name']}"][tem] = gen_em2(Et, mu, prec, logdet, cls, (m, Psi), 0.1)[0]
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        out.append(dict(group=TG, src=ya, tgt=yb, **ba)); print(TG, ya, '->', yb, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(out).to_csv(os.path.join(OUT, f'curve_{TG}.csv'), index=False)
    print('DONE', TG)
