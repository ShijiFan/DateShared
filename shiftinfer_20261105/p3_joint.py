"""EXPLORE_P3: MAP-EM with a JOINT (cross-crop) historical Gaussian prior on all class shifts, learned on the other site.
Delta = [delta_c] over classes x canonical dims; prior N(m, Psi) with Ledoit-Wolf Psi on complete training task vectors.
M-step: (blockdiag(n_c P_c) + Psi^-1) Delta = [P_c s_c] + Psi^-1 m.  Variants: hpj (prior mean m), hpj_zero (mean 0).
Usage: python p3_joint.py <test group> -> p3_hpj_<grp>.csv"""
import os, sys, itertools, time
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, NIT, DEV, K
HERE = os.path.dirname(os.path.abspath(__file__))


def build_joint(tr):
    tr = tr[tr.target.notna()].copy(); tr['key'] = list(zip(tr.dtype, tr.slot, tr['sub']))
    keys = sorted(set(tr.key)); kix = {k: i for i, k in enumerate(keys)}; nk = len(keys)
    V = []
    for _, d in tr.groupby(['group', 'src', 'tgt', 'fold']):
        v = np.full(K * nk, np.nan)
        v[d.cls.to_numpy() * nk + np.array([kix[k] for k in d.key])] = d.target.to_numpy()
        V.append(v)
    V = np.array(V); full = ~np.isnan(V).any(1)
    print('joint prior: tasks', len(V), 'complete', int(full.sum()), 'dim', V.shape[1], flush=True)
    return keys, V[full].mean(0), LedoitWolf().fit(V[full]).covariance_


def joint_for(keys, mean, cov, cls, task_keys):
    kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); D = len(task_keys); n = len(cls) * D
    m = np.zeros(n); P = np.zeros((n, n)); src, dst = [], []
    for a, c in enumerate(cls):
        for i, k in enumerate(task_keys):
            if k in kix: src.append(c * nk + kix[k]); dst.append(a * D + i)
    m[dst] = mean[src]; P[np.ix_(dst, dst)] = cov[np.ix_(src, src)]
    dvar = np.diag(cov)
    for a, c in enumerate(cls):
        for i, k in enumerate(task_keys):
            if k not in kix:
                same = [c * nk + kix[kk] for kk in keys if kk[0] == k[0]]; P[a * D + i, a * D + i] = float(dvar[same].mean())
    return m, P


def hpj_em(X, mu, prec, logdet, cls, m, Psi):
    D = X.shape[1]; C = len(cls); Pinv = np.linalg.inv(Psi + 1e-6 * np.eye(C * D)); rhs0 = Pinv @ m
    delta = {c: m[a * D:(a + 1) * D].copy() for a, c in enumerate(cls)}; pi = {c: 1.0 / C for c in cls}
    for _ in range(NIT):
        L = np.stack([np.log(pi[c]) - 0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c] - delta[c], prec[c], X - mu[c] - delta[c]) for c in cls], 1)
        R = np.exp(L - logsumexp(L, 1, keepdims=True))
        A = Pinv.copy(); b = rhs0.copy()
        for a, c in enumerate(cls):
            w = R[:, a]; pi[c] = max(w.mean(), 1e-6); sl = slice(a * D, (a + 1) * D)
            A[sl, sl] += w.sum() * prec[c]; b[sl] += prec[c] @ (w[:, None] * (X - mu[c])).sum(0)
        sol = np.linalg.solve(A, b); delta = {c: sol[a * D:(a + 1) * D] for a, c in enumerate(cls)}
    L = np.stack([np.log(pi[c]) - 0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c] - delta[c], prec[c], X - mu[c] - delta[c]) for c in cls], 1)
    return np.array(cls)[L.argmax(1)], delta


if __name__ == '__main__':
    TG = sys.argv[1]; t0 = time.time()
    allr = pd.concat([pd.read_parquet(os.path.join(HERE, f'p1_rows_{g}.parquet')) for g in ('A37', 'A88', 'B88')], ignore_index=True)
    keys, jm, jc = build_joint(allr[allr.site != TG[0]])
    te_rows = allr[allr.group == TG]; tidx = {key: g for key, g in te_rows.groupby(['src', 'tgt', 'fold'])}
    site, trk, Z, M = load(TG); out, derr = [], []
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    for ya, yb in itertools.permutations(years, 2):
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = len(tk)
        pred = {v: np.full(len(Yb), -1) for v in ('hpj', 'hpj_zero')}
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            mu, prec, logdet, cls = gauss_fit(Ea[trm], Ya[trm]); Et = Eb[tem]
            m, P = joint_for(keys, jm, jc, cls, tk)
            pred['hpj'][tem], dl = hpj_em(Et, mu, prec, logdet, cls, m, P)
            pred['hpj_zero'][tem], _ = hpj_em(Et, mu, prec, logdet, cls, np.zeros_like(m), P)
            g = tidx[(ya, yb, k)]                                    # shift-error diagnostic vs oracle (backscatter dims)
            for c in cls:
                gc = g[(g.cls == c)].sort_values('dim')
                if gc.target.notna().all():
                    bs = gc.dtype.to_numpy() < 2
                    derr.append(dict(src=ya, tgt=yb, fold=k, cls=c, se_hpj=float(np.mean((dl[c][bs] - gc.target.to_numpy()[bs]) ** 2)),
                                     se_em=float(np.mean((gc.dEM.to_numpy()[bs] - gc.target.to_numpy()[bs]) ** 2))))
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        out.append(dict(group=TG, src=ya, tgt=yb, **ba)); print(TG, ya, '->', yb, {v: round(x, 3) for v, x in ba.items()}, round(time.time() - t0), 's', flush=True)
    pd.DataFrame(out).to_csv(os.path.join(HERE, f'p3_hpj_{TG}.csv'), index=False)
    pd.DataFrame(derr).to_csv(os.path.join(HERE, f'p3_derr_{TG}.csv'), index=False); print('DONE', TG)
