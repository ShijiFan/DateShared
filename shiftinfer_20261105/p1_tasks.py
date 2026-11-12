"""EXPLORE_P1 pass 1: per (group, src, tgt, fold) task statistics, EM shift estimate, oracle shift, and fixed-shift evaluations
that need no regressor (g_raw, mean_em, oracle_delta, within_gauss). Usage: python p1_tasks.py <A37|A88|B88>
-> p1_rows_<grp>.parquet (regression rows), p1_base_<grp>.csv (per-transfer BA of baseline methods)."""
import os, sys, json, itertools, time
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); MY = os.path.join(HERE, '..', 'multiyear_20261017'); SHIFT = os.path.join(HERE, '..', 'shift_20261017')
ROOT = r'D:\sar_data\multiyear'
H = {'__file__': os.path.join(SHIFT, 'p3n_baselines.py'), '__name__': 'frozen'}
exec(open(os.path.join(SHIFT, 'p3n_baselines.py'), encoding='utf-8').read().split('\nrows = []')[0], H)
impute, gauss_fit, em_full, K, NIT = (H[k] for k in ('impute', 'gauss_fit', 'em_full', 'K', 'NIT'))
sys.path.insert(0, MY); from xyexcl import drop as xydrop
DEV = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]           # 2023 not read (EXPLORE_P1)
QS = [10, 25, 50, 75, 90]


def load(grp):
    site, trk = grp[0], int(grp[1:])
    Z = np.load(os.path.join(ROOT, f'features_{site}.npz'), allow_pickle=True); M = json.load(open(os.path.join(ROOT, f'features_{site}_meta.json')))
    return site, trk, Z, M


def data(Z, M, trk, yr, S):
    m = M[f'{yr}_T{trk}']; pos = [m['slots'].index(s) for s in S]; dc = [c for p in pos for c in (2 * p, 2 * p + 1)]
    g = lambda l: Z[f'{yr}_T{trk}_{l}'].astype(float)
    return g('VV')[:, dc], g('VH')[:, dc], g('COH')[:, pos], Z[f'{yr}_T{trk}_y'], Z[f'{yr}_T{trk}_folds']


def dim_meta(S):
    """type (0 VV, 1 VH, 2 COH), slot id, sub-index within slot, for E = [vv(2/slot), vh(2/slot), coh(1/slot)]."""
    t, s, u = [], [], []
    for ty in (0, 1):
        for sl in S:
            for sub in (0, 1): t.append(ty); s.append(sl); u.append(sub)
    for sl in S: t.append(2); s.append(sl); u.append(0)
    return np.array(t), np.array(s), np.array(u)


def loglik(X, mu, prec, logdet, cls, delta):
    return np.stack([-0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c] - delta[c], prec[c], X - mu[c] - delta[c]) for c in cls], 1)


def classify_fixed(X, mu, prec, logdet, cls, delta, n_it=NIT):
    """Means fixed at mu+delta; EM over class priors only; returns predicted labels."""
    L0 = loglik(X, mu, prec, logdet, cls, delta); lp = np.log(np.full(len(cls), 1.0 / len(cls)))
    for _ in range(n_it):
        L = L0 + lp; R = np.exp(L - logsumexp(L, 1, keepdims=True)); lp = np.log(np.maximum(R.mean(0), 1e-6))
    return np.array(cls)[(L0 + lp).argmax(1)]


def task_stats(Es, Et, ys_tr, mu, cls):
    """Per-dimension marginal statistics (target TEST vs source TRAIN) and per-class source statistics."""
    qs, qt = np.percentile(Es, QS, 0), np.percentile(Et, QS, 0)
    dim = dict(dmean=Et.mean(0) - Es.mean(0), dstd=Et.std(0) - Es.std(0), **{f'dq{q}': qt[i] - qs[i] for i, q in enumerate(QS)})
    gm = Es.mean(0)
    cl = {c: dict(rel=mu[c] - gm, sd=Es[ys_tr == c].std(0), psrc=float((ys_tr == c).mean())) for c in cls}
    return dim, cl


if __name__ == '__main__':
    GRP = sys.argv[1]; site, trk, Z, M = load(GRP); t0 = time.time()
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    rows, base = [], []
    wcache = {}
    for ya, yb in itertools.permutations(years, 2):
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); D = Ea.shape[1]
        pred = {v: np.full(len(Yb), -1) for v in ('g_raw', 'mean_em', 'oracle_delta', 'within_gauss')}
        for k in range(6):
            tr = (Fa[:, k] == 'TRAIN') & ~xydrop(GRP, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
            if not te.sum(): continue
            ys_, yt = Ya[tr], Yb[te]; Es, Et = Ea[tr], Eb[te]
            mu, prec, logdet, cls = gauss_fit(Es, ys_)
            dEM, R = em_full(Et, mu, prec, logdet, cls, np.ones(D, bool)); piEM = R.mean(0)
            pred['mean_em'][te] = np.array(cls)[R.argmax(1)]
            zero = {c: np.zeros(D) for c in cls}
            pred['g_raw'][te] = classify_fixed(Et, mu, prec, logdet, cls, zero)
            dOR = {}
            for c in cls:
                mt = yt == c
                dOR[c] = Et[mt].mean(0) - mu[c] if (mt.sum() >= 20 and (ys_ == c).sum() >= 20) else np.full(D, np.nan)
            pred['oracle_delta'][te] = classify_fixed(Et, mu, prec, logdet, cls, {c: np.nan_to_num(dOR[c]) for c in cls})
            # within-target Gaussian (target fold-k TRAIN -> TEST)
            trb = Fb[:, k] == 'TRAIN'
            mu2, prec2, ld2, cls2 = gauss_fit(Eb[trb], Yb[trb])
            pred['within_gauss'][te] = classify_fixed(Et, mu2, prec2, ld2, cls2, {c: np.zeros(D) for c in cls2})
            dim, cl = task_stats(Es, Et, ys_, mu, cls)
            for j, c in enumerate(cls):
                for d in range(D):
                    rows.append(dict(group=GRP, site=site, src=ya, tgt=yb, fold=k, cls=c, dim=d, dtype=ty[d], slot=sl[d], sub=sub[d],
                                     dEM=dEM[c][d], piEM=piEM[j], psrc=cl[c]['psrc'], rel=cl[c]['rel'][d], csd=cl[c]['sd'][d],
                                     **{kk: v[d] for kk, v in dim.items()}, target=dOR[c][d]))
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        base.append(dict(group=GRP, src=ya, tgt=yb, n_slots=len(S), **ba))
        print(GRP, ya, '->', yb, {v: round(x, 3) for v, x in ba.items()}, round(time.time() - t0), 's', flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(HERE, f'p1_rows_{GRP}.parquet'))
    pd.DataFrame(base).to_csv(os.path.join(HERE, f'p1_base_{GRP}.csv'), index=False)
    print('DONE', GRP, len(rows), 'rows', round(time.time() - t0), 's')
