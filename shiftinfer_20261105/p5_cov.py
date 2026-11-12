"""EXPLORE_P5: (i) stronger Gaussian baseline from the dev freeze (cov_shrink_0.1: mean EM + covariance (1-lam)S0 + lam*S_EM, lam 0.1);
(ii) joint-prior MAP-EM (P3) combined with the same covariance update; (iii) prior-strength sensitivity tau in {0.5, 2} (Psi scaled by tau).
Usage: python p5_cov.py <test group> -> p5_cov_<grp>.csv"""
import os, sys, itertools, time
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, NIT, DEV
from p3_joint import build_joint, joint_for
HERE = os.path.dirname(os.path.abspath(__file__))


def gen_em(X, mu, prec, logdet, cls, prior=None, lam=0.0, reg=1e-3):
    """Means: plain EM (prior None) or joint-prior MAP; covariance: fixed (lam 0) or partial update; priors pi always updated."""
    D = X.shape[1]; C = len(cls); S0 = {c: np.linalg.inv(prec[c]) for c in cls}; P = dict(prec); LD = dict(logdet)
    if prior is not None:
        m, Psi = prior; Pinv = np.linalg.inv(Psi + 1e-6 * np.eye(C * D)); rhs0 = Pinv @ m
        delta = {c: m[a * D:(a + 1) * D].copy() for a, c in enumerate(cls)}
    else:
        delta = {c: np.zeros(D) for c in cls}
    pi = {c: 1.0 / C for c in cls}
    for _ in range(NIT):
        L = np.stack([np.log(pi[c]) - 0.5 * LD[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c] - delta[c], P[c], X - mu[c] - delta[c]) for c in cls], 1)
        R = np.exp(L - logsumexp(L, 1, keepdims=True))
        if prior is not None:
            A = Pinv.copy(); b = rhs0.copy()
        for a, c in enumerate(cls):
            w = R[:, a]; pi[c] = max(w.mean(), 1e-6); sw = max(w.sum(), 1e-9); sx = (w[:, None] * (X - mu[c])).sum(0)
            if prior is None: delta[c] = sx / sw
            else:
                sl = slice(a * D, (a + 1) * D); A[sl, sl] += sw * P[c]; b[sl] += P[c] @ sx
        if prior is not None:
            sol = np.linalg.solve(A, b); delta = {c: sol[a * D:(a + 1) * D] for a, c in enumerate(cls)}
        if lam > 0:
            for a, c in enumerate(cls):
                w = R[:, a]; sw = max(w.sum(), 1e-9); Xc = X - mu[c] - delta[c]
                Se = (w[:, None] * Xc).T @ Xc / sw + reg * np.eye(D); Sc = (1 - lam) * S0[c] + lam * Se
                P[c] = np.linalg.inv(Sc); LD[c] = np.linalg.slogdet(Sc)[1]
    L = np.stack([np.log(pi[c]) - 0.5 * LD[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c] - delta[c], P[c], X - mu[c] - delta[c]) for c in cls], 1)
    return np.array(cls)[L.argmax(1)]


if __name__ == '__main__':
    TG = sys.argv[1]; t0 = time.time()
    allr = pd.concat([pd.read_parquet(os.path.join(HERE, f'p1_rows_{g}.parquet')) for g in ('A37', 'A88', 'B88')], ignore_index=True)
    keys, jm, jc = build_joint(allr[allr.site != TG[0]])
    site, trk, Z, M = load(TG); out = []
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    for ya, yb in itertools.permutations(years, 2):
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub))
        V = ('em_cov01', 'hpj_cov01', 'hpj_tau05', 'hpj_tau2'); pred = {v: np.full(len(Yb), -1) for v in V}
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            mu, prec, logdet, cls = gauss_fit(Ea[trm], Ya[trm]); Et = Eb[tem]
            m, Psi = joint_for(keys, jm, jc, cls, tk)
            pred['em_cov01'][tem] = gen_em(Et, mu, prec, logdet, cls, None, 0.1)
            pred['hpj_cov01'][tem] = gen_em(Et, mu, prec, logdet, cls, (m, Psi), 0.1)
            pred['hpj_tau05'][tem] = gen_em(Et, mu, prec, logdet, cls, (m, 0.5 * Psi), 0.0)
            pred['hpj_tau2'][tem] = gen_em(Et, mu, prec, logdet, cls, (m, 2.0 * Psi), 0.0)
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        out.append(dict(group=TG, src=ya, tgt=yb, **ba)); print(TG, ya, '->', yb, {v: round(x, 3) for v, x in ba.items()}, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(out).to_csv(os.path.join(HERE, f'p5_cov_{TG}.csv'), index=False)
    print('DONE', TG)
