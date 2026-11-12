"""Toy B round 3 (reflection on r2): per-date shared moisture offsets with vegetation attenuation (water-cloud form) explain
0.67/0.70 of held-out shift energy with K=27 and no learned modes (beats PCA27 when testing A). Now:
 (a) + per-class phenology timing (K=34); (b) hybrid = date_lvl_wc + r residual PCA modes learned on the training site,
 compared with PCA at the same K; (c) vegetation proxy alternatives for the attenuation."""
import sys, numpy as np, pandas as pd
from evalb import load, run_m2, m2_score, pca_design, weights, log, KEY
from r2 import date_cols, bs, att

T = load(); C = sorted(T.cls.unique())
def dlwc(d, b): w = np.where(bs(d), att(d, b), 1.0); return date_cols(d) * w[:, None]
def dlwc_time(d, th): return np.hstack([dlwc(d, th[0]), np.stack([d.dmu_s.to_numpy() * (d.cls.to_numpy() == c) for c in C], 1)])
def dl_vv(d, th):   # attenuation proxy = source VV level instead of VH
    w = np.where(bs(d), np.exp(-abs(th[0]) * (d.vv_s.to_numpy() + 20) / 10), 1.0); return date_cols(d) * w[:, None]


def hybrid(r, b=1.0):
    res = {}
    for trs, tes in (('A', 'B'), ('B', 'A')):
        tr, te = T[T.site == trs], T[T.site == tes]; sd = weights(tr)
        # residual of training tasks after per-task LS on the physics basis, in key space (weighted units)
        R = tr.copy(); R['delta'] = 0.0
        for tid, d in tr.groupby('tid'):
            wt = 1 / d.dtype.map(sd).to_numpy(); B = dlwc(d, b) * wt[:, None]; y = d.delta.to_numpy() * wt
            z, *_ = np.linalg.lstsq(B, y, rcond=None); R.loc[d.index, 'delta'] = (y - B @ z) / wt
        pr = pca_design(R, r, sd)
        des = lambda d, th: np.hstack([dlwc(d, b), pr(d, None)])
        K = 27 + r; pk = pca_design(tr, K, sd)
        res[f'test{tes}'] = dict(K=K, test=m2_score(te, des, None, sd), pcaK_test=m2_score(te, pk, None, sd))
    return res


if __name__ == '__main__':
    for n, (g, th, K) in {'dlwc_time': (dlwc_time, [1.0], 34), 'dl_vvproxy': (dl_vv, [0.5], 27)}.items():
        r = run_m2(g, np.array(th, float), K, T); log(n, 'M2', r, 'r3')
        print('M2', n, 'K', K, {k: (round(v['test']['all'], 3), 'pcaK', round(v['pcaK_test']['all'], 3), v['theta']) for k, v in r.items()}, flush=True)
    for rr in (3, 7, 13):
        r = hybrid(rr); log(f'hybrid_dlwc_pca{rr}', 'M2', r, 'r3 b fixed 1.0 (median of r2 fits)')
        print('M2 hybrid r', rr, {k: (v['K'], round(v['test']['all'], 3), 'pcaK', round(v['pcaK_test']['all'], 3)) for k, v in r.items()}, flush=True)
