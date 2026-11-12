"""EXPLORE_P9 helpers: shift bases (hand physics, discovered water-cloud date basis, MLLR), archive-estimated coefficient
scales, PPCA learned modes, Woodbury prior precision, and a MAP-EM that returns the shift estimate.
gen_em2 is a line-for-line copy of p5_cov.gen_em (same updates) that also returns delta and accepts a precomputed Pinv."""
import os, numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.optimize import least_squares
from p1_tasks import NIT, K
from p3_joint import joint_for
HERE = os.path.dirname(os.path.abspath(__file__))
KNMI = os.path.join(HERE, '..', 'multiyear_20261017', 'knmi_daily_my.csv'); STN = {'A': 283, 'B': 280}
B_ATT = 1.0                                   # vegetation attenuation fixed (EXPLORE_P9 §7b)
NOM = dict(L=(1.0, 1.0, 0.05), tau=10.0, e=(0.5, 0.5, 0.05))   # phys_nom nominal scales (EXPLORE_P9 §2)


# ---------------- MAP-EM ----------------
def gen_em2(X, mu, prec, logdet, cls, prior=None, lam=0.0, reg=1e-3, Pinv=None):
    D = X.shape[1]; C = len(cls); S0 = {c: np.linalg.inv(prec[c]) for c in cls}; P = dict(prec); LD = dict(logdet)
    if prior is not None:
        m, Psi = prior
        if Pinv is None: Pinv = np.linalg.inv(Psi + 1e-6 * np.eye(C * D))
        rhs0 = Pinv @ m
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
    return np.array(cls)[L.argmax(1)], delta


def woodbury(W, s, e):
    """inverse of W diag(s) W^T + diag(e)."""
    Ei = 1.0 / e; WE = W * Ei[:, None]
    M = np.diag(1.0 / s) + W.T @ WE
    return np.diag(Ei) - WE @ np.linalg.solve(M, WE.T)


# ---------------- task geometry ----------------
def dim_dates(M, trk, yr, S, ty, sl, sub):
    m = M[f'{yr}_T{trk}']; dt = pd.to_datetime(m['dates']); out = []
    for t, s, u in zip(ty, sl, sub):
        p = m['slots'].index(s)
        out.append(dt[2 * p + u] if t < 2 else dt[2 * p] + (dt[2 * p + 1] - dt[2 * p]) / 2)
    return pd.DatetimeIndex(out)


def vh_level(muc, ty, sl, sub):
    """source VH level at the same (slot, sub) for backscatter dims; slot mean of VH for coherence dims."""
    out = np.zeros(len(ty))
    for d in range(len(ty)):
        if ty[d] < 2: out[d] = muc[np.where((ty == 1) & (sl == sl[d]) & (sub == sub[d]))[0][0]]
        else: out[d] = muc[np.where((ty == 1) & (sl == sl[d]))[0]].mean()
    return out


def dmu_dt(muc, ty, doy):
    g = np.zeros(len(ty))
    for t in range(3):
        ix = np.where(ty == t)[0]; o = ix[np.argsort(doy[ix], kind='stable')]
        if len(o) > 1:
            x = doy[o].astype(float); x = x + np.arange(len(x)) * 1e-3            # guard duplicate doy
            g[o] = np.gradient(muc[o], x)
    return g


def hand_B(cls, mu, ty, doy):
    D = len(ty); C = len(cls); cols, kinds = [], []
    for t in range(3):
        cols.append(np.tile((ty == t).astype(float), C)); kinds.append(('L', t))
    for a, c in enumerate(cls):
        v = np.zeros(C * D); v[a * D:(a + 1) * D] = dmu_dt(mu[c], ty, doy); cols.append(v); kinds.append(('tau', 0))
    return np.stack(cols, 1), kinds


def disc_B(cls, mu, ty, sl, sub, doy, b=B_ATT):
    D = len(ty); C = len(cls); cols, kinds = [], []
    att = {c: np.exp(-b * (vh_level(mu[c], ty, sl, sub) + 25) / 10) for c in cls}
    for s, u in sorted(set((int(sl[d]), int(sub[d])) for d in range(D) if ty[d] < 2)):
        msk = (ty < 2) & (sl == s) & (sub == u)
        cols.append(np.concatenate([att[c] * msk for c in cls])); kinds.append(('wbs', 0))
    for s in sorted(set(int(sl[d]) for d in range(D) if ty[d] == 2)):
        cols.append(np.tile(((ty == 2) & (sl == s)).astype(float), C)); kinds.append(('wcoh', 0))
    for a, c in enumerate(cls):
        v = np.zeros(C * D); v[a * D:(a + 1) * D] = dmu_dt(mu[c], ty, doy); cols.append(v); kinds.append(('tau', 0))
    return np.stack(cols, 1), kinds


def mllr_B(cls, mu, D):
    C = len(cls); cols = []
    for d in range(D):
        sc = np.zeros(C * D); bi = np.zeros(C * D)
        for a, c in enumerate(cls): sc[a * D + d] = mu[c][d]; bi[a * D + d] = 1.0
        cols += [sc, bi]
    return np.stack(cols, 1)


# ---------------- weather (KNMI API, 60-day, decay 0.9; same as toy B) ----------------
_W = {}
def api_at(site, dates):
    if site not in _W:
        w = pd.read_csv(KNMI, parse_dates=['date']); w = w[w.station_code == STN[site]].set_index('date').sort_index()
        _W[site] = (w.RH.clip(lower=0) / 10.0)
    P = _W[site]; out = []
    for t in dates:
        p = P.loc[t - pd.Timedelta(days=59):t].to_numpy()[::-1]; out.append(float((p * 0.9 ** np.arange(len(p))).sum()))
    return np.array(out)


def season_api(site, yr):
    """H3 definition (PREREG_MULTIYEAR): mean daily API over 1 Mar - 30 Jun."""
    d = pd.date_range(f'{yr}-03-01', f'{yr}-06-30'); return float(api_at(site, d).mean())


def fit_mw(shift_table, train_site):
    """water-cloud weather equation (toy B wc_api) fitted on the archive site only."""
    T = shift_table[shift_table.site == train_site]; sd = T.groupby('dtype').delta.std(); wt = 1 / T.dtype.map(sd).to_numpy()
    def f(d, th):
        a = np.choose(d.dtype.to_numpy(), th[:3]); att = np.exp(-abs(th[3]) * (d.vh_s.to_numpy() + 25) / 10)
        return a * att * (np.log1p(d.API_t.to_numpy() / abs(th[4])) - np.log1p(d.API_s.to_numpy() / abs(th[4])))
    return least_squares(lambda th: (f(T, th) - T.delta.to_numpy()) * wt, np.array([1.5, 1.5, 0.0, 0.5, 3.0])).x


def mw_mean(th, cls, mu, ty, sl, sub, api_s, api_t):
    D = len(ty); out = []
    g = np.log1p(api_t / abs(th[4])) - np.log1p(api_s / abs(th[4]))
    for c in cls:
        att = np.exp(-abs(th[3]) * (vh_level(mu[c], ty, sl, sub) + 25) / 10)
        out.append(np.choose(ty, th[:3]) * att * g)
    return np.concatenate(out)


# ---------------- archive priors ----------------
def keyspace(rows):
    """V (complete tasks x K*nk) exactly as p3_joint.build_joint, plus task ids."""
    tr = rows[rows.target.notna()].copy(); tr['key'] = list(zip(tr.dtype, tr.slot, tr['sub']))
    keys = sorted(set(tr.key)); kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); V, ids = [], []
    for tid, d in tr.groupby(['group', 'src', 'tgt', 'fold']):
        v = np.full(K * nk, np.nan); v[d.cls.to_numpy() * nk + np.array([kix[k] for k in d.key])] = d.target.to_numpy(); V.append(v); ids.append(tid)
    V = np.array(V); full = ~np.isnan(V).any(1)
    return keys, V, ids, full


def key_type(keys):
    return np.tile(np.array([k[0] for k in keys]), K)


def ppca(V, ktype, r=None, thr=0.8):
    """PPCA on complete rows; r None -> smallest r with >= thr explained variance. Returns U (dim x r), e (dim,), r, evr."""
    n = len(V); Vc = V - V.mean(0); U_, s, Vt = np.linalg.svd(Vc, full_matrices=False); lam = s ** 2 / (n - 1)
    evr = np.cumsum(lam) / lam.sum()
    if r is None: r = int(np.searchsorted(evr, thr) + 1)
    rec = (U_[:, :r] * s[:r]) @ Vt[:r]; res2 = ((Vc - rec) ** 2).sum(0) / (n - 1)
    e = np.zeros(V.shape[1])
    for t in range(3): e[ktype == t] = res2[ktype == t].mean()
    sbar = e.mean(); U = Vt[:r].T * np.sqrt(np.maximum(lam[:r] - sbar, 1e-12))
    return U, e, r, evr


def map_lowrank(keys, U, e, tot, cls, tk):
    """map key-space low-rank factor and diagonal to task dims; missing keys -> zero factor rows, diag = type-mean total var."""
    kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); D = len(tk); Ut = np.zeros((len(cls) * D, U.shape[1])); et = np.zeros(len(cls) * D)
    ty = np.array([k[0] for k in tk])
    for a, c in enumerate(cls):
        for i, k in enumerate(tk):
            if k in kix: Ut[a * D + i] = U[c * nk + kix[k]]; et[a * D + i] = e[c * nk + kix[k]]
            else: et[a * D + i] = tot[ty[i]]
    return Ut, et


def phys_fit(arch, keys, jm, jc, which, kfull_ids=None):
    """Per archive task LS of (oracle - m) on its own basis; returns coefficient variances by kind, residual var by dtype,
    and key-space residual vectors for complete tasks (dict id -> vector)."""
    kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); coef = {}; res_by_t = {0: [], 1: [], 2: []}; RV = {}
    for t in arch:
        cls, tk, mu, ty, sl, sub, doy = t['cls'], t['tk'], t['mu'], t['ty'], t['sl'], t['sub'], t['doy']
        B, kinds = hand_B(cls, mu, ty, doy) if which == 'hand' else disc_B(cls, mu, ty, sl, sub, doy)
        m, _ = joint_for(keys, jm, jc, cls, tk)
        y = np.concatenate([t['orc'][c] for c in cls]) - m; ok = ~np.isnan(y)
        th, *_ = np.linalg.lstsq(B[ok], y[ok], rcond=None); r = np.full(len(y), np.nan); r[ok] = y[ok] - B[ok] @ th
        for kd, v in zip(kinds, th): coef.setdefault(kd, []).append(v)
        D = len(tk)
        for a, c in enumerate(cls):
            for i in range(D):
                if ok[a * D + i]: res_by_t[ty[i]].append(r[a * D + i] ** 2)
        v = np.full(K * nk, np.nan)
        for a, c in enumerate(cls):
            for i, k in enumerate(tk):
                if k in kix: v[c * nk + kix[k]] = r[a * D + i]
        RV[t['id']] = v
    var = {kd: float(np.mean(np.square(v))) for kd, v in coef.items()}
    e0 = np.array([np.mean(res_by_t[t]) for t in range(3)])
    return var, e0, RV
