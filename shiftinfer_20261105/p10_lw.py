"""EXPLORE_P10: variant B = discovered water-cloud basis + full-rank Ledoit-Wolf residual covariance (hybridD_lw).
Usage: python p10_lw.py <test group> main|curve -> p10_main_<grp>.csv | p10_curve_<grp>.csv"""
import os, sys, itertools, pickle, time
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, DEV, K
from p3_joint import joint_for
from p9_lib import gen_em2, dim_dates, disc_B, keyspace, phys_fit
from p9_basis import build_priors, task_priors, GROUPS
HERE = os.path.dirname(os.path.abspath(__file__))
V3 = ['hpj_cov01', 'hybridD_r*', 'hybridD_lw']


def add_lw(P, tr_rows, arch):
    """residual LW covariance of the discovered-basis fit (complete archive tasks), key space."""
    _, V, ids, full = keyspace(tr_rows); idf = [ids[i] for i in np.where(full)[0]]
    _, _, RV = phys_fit(arch, P['keys'], P['jm'], P['jc'], 'disc')
    Rd = np.array([RV[i] for i in idf]); P['res_lw'] = LedoitWolf().fit(Rd).covariance_
    return P


def priors3(P, cls, mu, tk, ty, sl, sub, doy):
    C, D = len(cls), len(tk)
    PR = task_priors(P, ['hpj_cov01', 'hybridD_r*'], cls, mu, tk, ty, sl, sub, doy)
    m_arch = PR['hpj_cov01'][0]
    Bd, kd = disc_B(cls, mu, ty, sl, sub, doy); s = np.array([P['disc']['var'][k] for k in kd])
    _, Pres = joint_for(P['keys'], np.zeros(len(P['jm'])), P['res_lw'], cls, tk)
    PR['hybridD_lw'] = (m_arch, Bd @ np.diag(s) @ Bd.T + Pres, None)
    return PR


def run_task(PR, Et, mu, prec, logdet, cls, D):
    C = len(cls); out = {}
    for v, (m, Psi, Pi) in PR.items():
        out[v] = gen_em2(Et, mu, prec, logdet, cls, (m, Psi if Psi is not None else np.zeros((C * D, C * D))), 0.1, Pinv=Pi)[0]
    return out


if __name__ == '__main__':
    TG, MODE = sys.argv[1], sys.argv[2]; t0 = time.time(); trs = 'B' if TG[0] == 'A' else 'A'
    allr = pd.concat([pd.read_parquet(os.path.join(HERE, f'p1_rows_{g}.parquet')) for g in GROUPS], ignore_index=True)
    tr = allr[allr.site == trs]
    arch = sum([pickle.load(open(os.path.join(HERE, f'p9_arch_{g}.pkl'), 'rb')) for g in GROUPS if g[0] == trs], [])
    if MODE == 'main':
        PS = {'k8_all': add_lw(build_priors(tr, arch, with_fixed=False), tr, arch)}
    else:
        rng = np.random.default_rng(20261105); ty_all = sorted(set(tr.src))
        subsets = {f'k{k}_{i}': sorted(rng.choice(ty_all, k, replace=False).tolist()) for k in (3, 5) for i in (0, 1)}
        subsets.update({f'k{k}_{i}': sorted(rng.choice(ty_all, k, replace=False).tolist()) for k in (3, 5) for i in (2, 3)})
        subsets['k8_all'] = ty_all; print('subsets', subsets, flush=True)
        PS = {}
        for nm, ys in subsets.items():
            trs_ = tr[tr.src.isin(ys) & tr.tgt.isin(ys)]; ar_ = [a for a in arch if a['id'][1] in ys and a['id'][2] in ys]
            PS[nm] = add_lw(build_priors(trs_, ar_, with_fixed=False), trs_, ar_)
    site, trk, Z, M = load(TG); out = []
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    todo = list(itertools.permutations(years, 2)); todo = todo if MODE == 'main' else todo[0::2]
    for ya, yb in todo:
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]; doy = np.array(dim_dates(M, trk, ya, S, ty, sl, sub).dayofyear)
        pred = {f'{nm}|{v}': np.full(len(Yb), -1) for nm in PS for v in V3}
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            mu, prec, logdet, cls = gauss_fit(Ea[trm], Ya[trm]); Et = Eb[tem]
            for nm, P in PS.items():
                for v, lab in run_task(priors3(P, cls, mu, tk, ty, sl, sub, doy), Et, mu, prec, logdet, cls, D).items(): pred[f'{nm}|{v}'][tem] = lab
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        out.append(dict(group=TG, src=ya, tgt=yb, **ba)); print(TG, ya, '->', yb, {v: round(x, 3) for v, x in ba.items() if v.startswith('k8')}, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(out).to_csv(os.path.join(HERE, f'p10_{MODE}_{TG}.csv'), index=False)
    print('DONE', TG)
