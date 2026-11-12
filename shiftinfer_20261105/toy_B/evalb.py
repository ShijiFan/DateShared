"""Toy B evaluator (CHARTER_LINES_ZH.md §2.4). Leave-site-out: fit global parameters on one site's archive, score on the other.
M1 (weather-only): prediction = f(features; theta); score R^2 against zero shift, per dtype (VV, VH, COH) and backscatter mean.
M2 (structure): per-task design B(features; theta) (rows x K), z fitted per task by weighted LS (oracle fit); score = explained
fraction of weighted shift energy on the held-out site; always reported next to PCA with the same K (learned on the training site).
Rows weighted by 1/std(delta | dtype) from the training site so dB and coherence are comparable."""
import os, json, time
import numpy as np, pandas as pd
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
KEY = ['cls', 'dtype', 'slot', 'sub']; TASK = ['group', 'src', 'tgt']


def load():
    T = pd.read_csv(os.path.join(HERE, 'shift_table.csv'))
    # cross-dtype lookups for the source year: VH level of the same class/slot/sub (vegetation proxy), VV level
    for nm, ty in (('vh_s', 1), ('vv_s', 0)):
        L = T[T.dtype == ty][['group', 'src', 'cls', 'slot', 'sub', 'mu_s']].drop_duplicates(['group', 'src', 'cls', 'slot', 'sub']).rename(columns={'mu_s': nm})
        T = T.merge(L, on=['group', 'src', 'cls', 'slot', 'sub'], how='left')
    T[['vh_s', 'vv_s']] = T.groupby(['group', 'src', 'cls', 'slot'])[['vh_s', 'vv_s']].transform(lambda s: s.fillna(s.mean()))
    T['tid'] = T.groupby(TASK).ngroup()
    return T


def weights(tr):
    sd = tr.groupby('dtype').delta.std(); return sd


def r2_by_type(d, pred):
    out = {}
    for ty, nm in ((0, 'VV'), (1, 'VH'), (2, 'COH')):
        m = d.dtype.to_numpy() == ty; y = d.delta.to_numpy()[m]; p = pred[m]
        out[nm] = 1 - ((y - p) ** 2).sum() / (y ** 2).sum()
    out['BS'] = (out['VV'] + out['VH']) / 2
    return out


def run_m1(f, theta0, T, bounds=None):
    """f(df, theta) -> prediction array. theta fitted by least squares (weighted) on the training site."""
    res = {}
    for trs, tes in (('A', 'B'), ('B', 'A')):
        tr, te = T[T.site == trs], T[T.site == tes]; sd = weights(tr); wt = 1 / tr.dtype.map(sd).to_numpy()
        if len(theta0):
            fit = least_squares(lambda th: (f(tr, th) - tr.delta.to_numpy()) * wt, theta0, bounds=bounds if bounds else (-np.inf, np.inf))
            th = fit.x
        else: th = np.array([])
        res[f'test{tes}'] = dict(train=r2_by_type(tr, f(tr, th)), test=r2_by_type(te, f(te, th)), theta=np.round(th, 4).tolist())
    return res


def task_explained(d, B, wt):
    """weighted LS per task; returns residual energy and total energy (weighted)."""
    y = d.delta.to_numpy() * wt; Bw = B * wt[:, None]
    z, *_ = np.linalg.lstsq(Bw, y, rcond=None); r = y - Bw @ z
    return (r ** 2).sum(), (y ** 2).sum(), z


def m2_score(df, design, th, sd):
    rs, tot = 0.0, 0.0; per_type = np.zeros((3, 2))
    for _, d in df.groupby('tid'):
        wt = 1 / d.dtype.map(sd).to_numpy(); B = design(d, th)
        y = d.delta.to_numpy() * wt; Bw = B * wt[:, None]; z, *_ = np.linalg.lstsq(Bw, y, rcond=None); r = y - Bw @ z
        rs += (r ** 2).sum(); tot += (y ** 2).sum()
        for ty in range(3):
            m = d.dtype.to_numpy() == ty; per_type[ty] += [(r[m] ** 2).sum(), (y[m] ** 2).sum()]
    return dict(all=1 - rs / tot, VV=1 - per_type[0, 0] / per_type[0, 1], VH=1 - per_type[1, 0] / per_type[1, 1], COH=1 - per_type[2, 0] / per_type[2, 1])


def pca_design(tr, K, sd):
    """PCA basis in key space from complete training tasks (weighted units); returns design(d, th)."""
    keys = tr[KEY].drop_duplicates().sort_values(KEY).apply(tuple, axis=1).tolist(); kix = {k: i for i, k in enumerate(keys)}
    V = []
    for _, d in tr.groupby('tid'):
        v = np.full(len(keys), np.nan); wt = 1 / d.dtype.map(sd).to_numpy()
        v[[kix[k] for k in d[KEY].apply(tuple, axis=1)]] = d.delta.to_numpy() * wt; V.append(v)
    V = np.array(V); V = V[~np.isnan(V).any(1)]
    U = np.linalg.svd(V, full_matrices=False)[2][:K].T                       # keys x K (weighted units)
    def design(d, th):
        ix = [kix.get(k, -1) for k in d[KEY].apply(tuple, axis=1)]; wt = 1 / d.dtype.map(sd).to_numpy()
        B = np.zeros((len(d), K)); ok = np.array(ix) >= 0; B[ok] = U[np.array(ix)[ok]]
        return B / wt[:, None]                                                   # back to raw units (weighted again in scoring)
    return design


def run_m2(design, theta0, K, T, bounds=None):
    res = {}
    for trs, tes in (('A', 'B'), ('B', 'A')):
        tr, te = T[T.site == trs], T[T.site == tes]; sd = weights(tr)
        if len(theta0):
            def obj(th):
                out = []
                for _, d in tr.groupby('tid'):
                    wt = 1 / d.dtype.map(sd).to_numpy(); B = design(d, th); y = d.delta.to_numpy() * wt; Bw = B * wt[:, None]
                    z, *_ = np.linalg.lstsq(Bw, y, rcond=None); out.append(y - Bw @ z)
                return np.concatenate(out)
            th = least_squares(obj, theta0, bounds=bounds if bounds else (-np.inf, np.inf), max_nfev=200).x
        else: th = np.array([])
        pc = pca_design(tr, K, sd)
        res[f'test{tes}'] = dict(K=K, test=m2_score(te, design, th, sd), pcaK_test=m2_score(te, pc, None, sd), train=m2_score(tr, design, th, sd), theta=np.round(th, 4).tolist())
    return res


def rnd(x):
    if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, float): return round(x, 4)
    return x


def log(name, kind, res, note=''):
    with open(os.path.join(HERE, 'toy_results.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(name=name, kind=kind, note=note, res=rnd(res), t=time.strftime('%H:%M:%S')), ensure_ascii=False) + '\n')
