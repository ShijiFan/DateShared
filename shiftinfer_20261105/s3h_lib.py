"""S3h (EXPLORE_S3D_Q1_ZH.md): shared data access, transfers, archive variants and pooled-archive data for s3h_cpu.py / s3h_deep.py.
Frozen method (frozen/hybridD_lw.py) and the archive task caches p9_arch_<g>.pkl are read only."""
import os, sys, json, itertools, pickle
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, 'frozen'), os.path.join(HERE, '..', 'multiyear_20261017')): sys.path.insert(0, p)
import hybridD_lw as F
from p9_lib import gen_em2, dim_dates
from p1_tasks import dim_meta, H
from xyexcl import drop as nl_drop
hgb, assemble = H['hgb'], H['assemble']
DEVY = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]
ROOT = {'A': r'D:\sar_data\multiyear', 'B': r'D:\sar_data\multiyear', 'FR': r'D:\sar_data\fr_a'}
GROUPS = {'B88': ('B', 88), 'A37': ('A', 37), 'A88': ('A', 88), 'FR': ('FR', 59)}
POOL = {'B88': ['A37', 'A88'], 'A37': ['B88'], 'A88': ['B88'], 'FR': ['A37', 'A88', 'B88']}
ARCH = {'B88': 'archive_NL_A.npz', 'A37': 'archive_NL_B.npz', 'A88': 'archive_NL_B.npz', 'FR': 'archive_NL_all.npz'}
_Z, _M = {}, {}


def site(s):
    if s not in _Z:
        f = 'features_FR' if s == 'FR' else f'features_{s}'
        _Z[s] = np.load(os.path.join(ROOT[s], f + '.npz'), allow_pickle=True); _M[s] = json.load(open(os.path.join(ROOT[s], f + '_meta.json')))
    return _Z[s], _M[s]


def years(g):
    s, t = GROUPS[g]; _, M = site(s)
    ys = sorted(int(k.split('_')[0]) for k in M if k.endswith(f'_T{t}'))
    return ys if s == 'FR' else [y for y in ys if y in DEVY]


def transfers(g):
    p = list(itertools.permutations(years(g), 2))
    return p if g == 'FR' else p[0::2]


def slots(g, y):
    s, t = GROUPS[g]; return site(s)[1][f'{y}_T{t}']['slots']


def common(g, ya, yb): return tuple(sorted(set(slots(g, ya)) & set(slots(g, yb))))


def extract(g, y, S):
    """E = [VV(2/slot), VH(2/slot), COH(1/slot)] for slots S; slots this season lacks are NaN. Returns E, y, folds, complete."""
    s, t = GROUPS[g]; Z, M = site(s); tag = f'{y}_T{t}'; m = M[tag]; n = len(Z[tag + '_y'])
    vv = np.full((n, 2 * len(S)), np.nan); vh = vv.copy(); co = np.full((n, len(S)), np.nan)
    for j, sl in enumerate(S):
        if sl not in m['slots']: continue
        p = m['slots'].index(sl)
        vv[:, 2 * j:2 * j + 2] = Z[tag + '_VV'][:, 2 * p:2 * p + 2]; vh[:, 2 * j:2 * j + 2] = Z[tag + '_VH'][:, 2 * p:2 * p + 2]; co[:, j] = Z[tag + '_COH'][:, p]
    return np.hstack([vv, vh, co]).astype(float), Z[tag + '_y'], Z[tag + '_folds'], all(sl in m['slots'] for sl in S)


def layout(g, y, S):
    s, t = GROUPS[g]; ty, sl, sub = dim_meta(S)
    return list(zip(ty, sl, sub)), np.array(dim_dates(site(s)[1], t, y, S, ty, sl, sub).dayofyear)


_FRX = None


def xydrop(g, ya, yb, k, n):
    global _FRX
    if g != 'FR': return nl_drop(g, ya, yb, k, n)
    if _FRX is None: _FRX = np.load(os.path.join(ROOT['FR'], 'xyexcl_FR.npz'))
    m = np.zeros(n, bool)
    if int(ya) != int(yb): m[_FRX[f'{int(ya)}_{int(yb)}_{int(k)}']] = True
    return m


# ---------------- archives ----------------
def tasks_for(g):
    return sum((pickle.load(open(os.path.join(HERE, f'p9_arch_{p}.pkl'), 'rb')) for p in POOL[g]), [])


def frozen_archive(g): return F.Archive.load(os.path.join(HERE, 'frozen', ARCH[g]))


def basis_for(mode, cls, mu, ty, sl, sub, doy):
    B, kinds = F.basis(cls, mu, ty, sl, sub, doy)
    keep = {'full': lambda k: True, 'no_tau': lambda k: k != 'tau', 'no_date': lambda k: k == 'tau'}[mode]
    ix = np.array([keep(k) for k in kinds]); return B[:, ix], [k for k in kinds if keep(k)]


def _vectors(tasks):        # as Codex 06 run_cpu.vectors
    keys = sorted({tuple(int(x) for x in k) for t in tasks for c in t['cls'] if np.isfinite(t['orc'][c]).all() for k in t['tk']})
    ki = {k: i for i, k in enumerate(keys)}; nk = len(keys); V = []
    for t in tasks:
        v = np.full(F.K * nk, np.nan)
        for c in t['cls']:
            for i, k in enumerate(t['tk']):
                if np.isfinite(t['orc'][c][i]): v[c * nk + ki[tuple(int(x) for x in k)]] = t['orc'][c][i]
        V.append(v)
    V = np.array(V); return keys, V, np.isfinite(V).all(1)


def build_control(tasks, mode):
    """Archive built exactly as F.Archive.build but with basis columns restricted by mode (Codex 06 build_control, + no_date)."""
    tasks = sorted(tasks, key=lambda t: t['id']); keys, V, full = _vectors(tasks)
    ki = {k: i for i, k in enumerate(keys)}; nk = len(keys); mean = V[full].mean(0); coef = {}; res = []
    for t, isfull in zip(tasks, full):
        cls = t['cls']; tk = [tuple(int(x) for x in k) for k in t['tk']]
        B, kinds = basis_for(mode, cls, t['mu'], t['ty'], t['sl'], t['sub'], t['doy'])
        m, _ = F._map(keys, mean, np.eye(len(mean)), cls, tk)
        y = np.concatenate([t['orc'][c] for c in cls]) - m; ok = np.isfinite(y)
        th, *_ = np.linalg.lstsq(B[ok], y[ok], rcond=None)
        r = np.full(len(y), np.nan); r[ok] = y[ok] - B[ok] @ th
        for k, x in zip(kinds, th): coef.setdefault(k, []).append(x)
        if isfull:
            v = np.full(F.K * nk, np.nan); D = len(tk)
            for a, c in enumerate(cls):
                for i, k in enumerate(tk): v[c * nk + ki[k]] = r[a * D + i]
            res.append(v)
    return F.Archive(keys, mean, {k: float(np.mean(np.square(v))) for k, v in coef.items()},
                     LedoitWolf().fit(np.array(res)).covariance_, {'mode': mode, 'n_tasks': len(tasks)})


def control_prior(arch, mode, cls, mu, tk, doy):
    m, _ = F._map(arch.keys, arch.mean, np.eye(len(arch.mean)), cls, tk)
    _, R = F._map(arch.keys, np.zeros(len(arch.mean)), arch.res_cov, cls, tk)
    ty, sl, sub = (np.array([k[i] for k in tk]) for i in range(3))
    B, kinds = basis_for(mode, cls, mu, ty, sl, sub, doy)
    return m, B @ np.diag([arch.s_a[k] for k in kinds]) @ B.T + R


def shuffle_prior(arch, cls, mu, tk, doy, seed):
    """Codex 06: permute the physical covariance over dates within each class and observation type; residual and mean kept."""
    m, P = arch.prior(cls, mu, tk, doy); _, R = F._map(arch.keys, np.zeros(len(arch.mean)), arch.res_cov, cls, tk)
    C = P - R; perm = np.arange(len(m)); rng = np.random.default_rng(seed)
    for a, c in enumerate(cls):
        for ty in (0, 1, 2):
            ix = np.array([a * len(tk) + i for i, k in enumerate(tk) if k[0] == ty]); perm[ix] = rng.permutation(ix)
    return m, C[np.ix_(perm, perm)] + R


def check_full(g, tasks):
    a, b = build_control(tasks, 'full'), frozen_archive(g)
    d = max(np.max(np.abs(a.mean - b.mean)), np.max(np.abs(a.res_cov - b.res_cov)), max(abs(a.s_a[k] - b.s_a[k]) for k in b.s_a))
    assert d < 1e-10, d; return d


# ---------------- pooled archive seasons ----------------
def pool_seasons(g, S):
    """All labelled seasons of the archive groups, on slots S. Each: dict(g, y, E (NaN where slot missing), Y, complete, tk, doy)."""
    out = []
    for p in POOL[g]:
        for y in years(p):
            E, Y, _, comp = extract(p, y, S)
            d = dict(g=p, y=y, E=E, Y=Y, complete=comp)
            if comp: d['tk'], d['doy'] = layout(p, y, S)
            out.append(d)
    return out


def shift_season(ps, Et, arch):
    """Move one archive season to the target by the frozen MAP-EM estimate (unlabelled target fold). Incomplete seasons unchanged."""
    if not ps['complete']: return ps['E'], False
    Ep = F.impute(ps['E'], ps['E']); mu, prec, ld, cls = F.gauss_fit(Ep, ps['Y']); m, Psi = arch.prior(cls, mu, ps['tk'], ps['doy'])
    _, d = F.map_em(Et, mu, prec, ld, cls, m, Psi)
    return F.resynthesize(ps['E'], ps['Y'], d), True


def to_seq(E):
    """E -> db_run layout (n, 4, 2S) = VV, VH, VH-VV, COH repeated twice."""
    n2 = (E.shape[1] // 5) * 2; vv, vh, co = E[:, :n2], E[:, n2:2 * n2], E[:, 2 * n2:]
    return np.stack([vv, vh, vh - vv, np.repeat(co, 2, axis=1)], axis=1).astype(np.float32)


def split(E):
    n2 = (E.shape[1] // 5) * 2; return E[:, :n2], E[:, n2:2 * n2], E[:, 2 * n2:]
