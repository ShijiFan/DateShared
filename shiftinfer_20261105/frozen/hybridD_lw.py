"""FROZEN METHOD hybridD_lw (user sign-off 2026-11-06; EXPLORE_P10_ZH.md). Self-contained: numpy, scipy, scikit-learn.

Model: per-class Ledoit-Wolf Gaussians fitted on ONE labelled source year; target year unlabelled.
Target class means mu_c + delta_c, prior delta ~ N(m, Psi), Psi = B diag(s_a) B^T + Sigma_res (archive-learned);
B = discovered water-cloud basis: per acquisition date one column shared by all crops on VV and VH, scaled by
exp(-(VH_src_c + 25)/10) (b = 1.0); per slot one coherence column; per class one phenology-timing column d mu_c / dt.
MAP-EM (NIT = 30) with partial covariance update lam = 0.1. No weather data, no target labels.

Feature layout (must match the archive): E = [VV(slot,sub) for slot in S for sub in (0,1)] + [VH(...)] + [COH(slot) for slot in S];
task keys tk[d] = (dtype, slot, sub) with dtype 0 VV, 1 VH, 2 COH; doy[d] = source-year day of year of that acquisition
(COH: pair midpoint). Classes 0..6 = PermGrass, TempGrass, Maize, Potato, Beet, WinterCereal, SpringCereal.

API:
  arch = Archive.build(tasks)            # tasks: list of dicts from the labelled archive (see build docstring)
  arch.save(path) / Archive.load(path)   # frozen "weights"
  labels, delta = adapt(Es, ys, Et, tk, doy, arch)
  Es_shifted = resynthesize(Es, ys, delta)   # C2: shift source samples, then retrain any classifier"""
import json
import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

K = 7; NIT = 30; LAM = 0.1; B_ATT = 1.0; MIN_N = 30
CLASSES = ["PermGrass", "TempGrass", "Maize", "Potato", "Beet", "WinterCereal", "SpringCereal"]


# ---------------- source model ----------------
def impute(X, ref):
    X = X.copy(); m = np.isnan(X); X[m] = np.take(np.nanmedian(ref, 0), np.nonzero(m)[1]); return X


def gauss_fit(Xs, ys):
    mu, prec, logdet, cls = {}, {}, {}, []
    for c in range(K):
        Xc = Xs[ys == c]
        if len(Xc) < MIN_N: continue
        S = LedoitWolf().fit(Xc).covariance_
        mu[c] = Xc.mean(0); prec[c] = np.linalg.inv(S); logdet[c] = np.linalg.slogdet(S)[1]; cls.append(c)
    return mu, prec, logdet, cls


# ---------------- discovered basis ----------------
def _vh_level(muc, ty, sl, sub):
    out = np.zeros(len(ty))
    for d in range(len(ty)):
        if ty[d] < 2: out[d] = muc[np.where((ty == 1) & (sl == sl[d]) & (sub == sub[d]))[0][0]]
        else: out[d] = muc[np.where((ty == 1) & (sl == sl[d]))[0]].mean()
    return out


def _dmu_dt(muc, ty, doy):
    g = np.zeros(len(ty))
    for t in range(3):
        ix = np.where(ty == t)[0]; o = ix[np.argsort(doy[ix], kind='stable')]
        if len(o) > 1:
            x = doy[o].astype(float) + np.arange(len(o)) * 1e-3
            g[o] = np.gradient(muc[o], x)
    return g


def basis(cls, mu, ty, sl, sub, doy, b=B_ATT):
    D = len(ty); C = len(cls); cols, kinds = [], []
    att = {c: np.exp(-b * (_vh_level(mu[c], ty, sl, sub) + 25) / 10) for c in cls}
    for s, u in sorted(set((int(sl[d]), int(sub[d])) for d in range(D) if ty[d] < 2)):
        msk = (ty < 2) & (sl == s) & (sub == u)
        cols.append(np.concatenate([att[c] * msk for c in cls])); kinds.append('wbs')
    for s in sorted(set(int(sl[d]) for d in range(D) if ty[d] == 2)):
        cols.append(np.tile(((ty == 2) & (sl == s)).astype(float), C)); kinds.append('wcoh')
    for a, c in enumerate(cls):
        v = np.zeros(C * D); v[a * D:(a + 1) * D] = _dmu_dt(mu[c], ty, doy); cols.append(v); kinds.append('tau')
    return np.stack(cols, 1), kinds


def _map(keys, mean, cov, cls, tk):
    """key-space mean/cov -> task dims (missing keys: zero mean, type-mean variance of the same class)."""
    kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); D = len(tk); n = len(cls) * D
    m = np.zeros(n); P = np.zeros((n, n)); src, dst = [], []
    for a, c in enumerate(cls):
        for i, k in enumerate(tk):
            if k in kix: src.append(c * nk + kix[k]); dst.append(a * D + i)
    m[dst] = mean[src]; P[np.ix_(dst, dst)] = cov[np.ix_(src, src)]
    dvar = np.diag(cov)
    for a, c in enumerate(cls):
        for i, k in enumerate(tk):
            if k not in kix:
                same = [c * nk + kix[kk] for kk in keys if kk[0] == k[0]]; P[a * D + i, a * D + i] = float(dvar[same].mean())
    return m, P


# ---------------- archive ("weights") ----------------
class Archive:
    def __init__(self, keys, mean, s_a, res_cov, meta=None):
        self.keys = [tuple(int(x) for x in k) for k in keys]; self.mean = np.asarray(mean); self.s_a = dict(s_a)
        self.res_cov = np.asarray(res_cov); self.meta = meta or {}

    @staticmethod
    def build(tasks, meta=None):
        """tasks: list of dicts with cls (list), tk (list of (dtype,slot,sub)), mu {c: vec}, ty, sl, sub, doy (arrays),
        orc {c: oracle shift vec or NaN} (oracle = target-year class mean - source class mean, >= 20 parcels each), id (sortable)."""
        tasks = sorted(tasks, key=lambda t: t['id'])
        keys = sorted({tuple(int(x) for x in k) for t in tasks for c in t['cls'] if np.isfinite(t['orc'][c]).all() for k in t['tk']})
        kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); V = []
        for t in tasks:
            v = np.full(K * nk, np.nan)
            for c in t['cls']:
                o = t['orc'][c]
                for i, k in enumerate(t['tk']):
                    if np.isfinite(o[i]): v[c * nk + kix[tuple(int(x) for x in k)]] = o[i]
            V.append(v)
        V = np.array(V); full = ~np.isnan(V).any(1); mean = V[full].mean(0)
        coef, R = {}, []
        for t, f in zip(tasks, full):
            cls, tk = t['cls'], [tuple(int(x) for x in k) for k in t['tk']]
            B, kinds = basis(cls, t['mu'], t['ty'], t['sl'], t['sub'], t['doy'])
            m, _ = _map(keys, mean, np.eye(len(mean)), cls, tk)
            y = np.concatenate([t['orc'][c] for c in cls]) - m; ok = ~np.isnan(y)
            th, *_ = np.linalg.lstsq(B[ok], y[ok], rcond=None); r = np.full(len(y), np.nan); r[ok] = y[ok] - B[ok] @ th
            for kd, v in zip(kinds, th): coef.setdefault(kd, []).append(v)
            if f:
                D = len(tk); rv = np.full(K * nk, np.nan)
                for a, c in enumerate(cls):
                    for i, k in enumerate(tk): rv[c * nk + kix[k]] = r[a * D + i]
                R.append(rv)
        s_a = {kd: float(np.mean(np.square(v))) for kd, v in coef.items()}
        res_cov = LedoitWolf().fit(np.array(R)).covariance_
        return Archive(keys, mean, s_a, res_cov, dict(meta or {}, n_tasks=len(tasks), n_complete=int(full.sum())))

    def save(self, path):
        np.savez_compressed(path, keys=np.array(self.keys), mean=self.mean, res_cov=self.res_cov,
                            s_a=json.dumps(self.s_a), meta=json.dumps(self.meta))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return Archive(z['keys'].tolist(), z['mean'], json.loads(str(z['s_a'])), z['res_cov'], json.loads(str(z['meta'])))

    def prior(self, cls, mu, tk, doy):
        tk = [tuple(int(x) for x in k) for k in tk]; ty = np.array([k[0] for k in tk]); sl = np.array([k[1] for k in tk]); sub = np.array([k[2] for k in tk])
        m, _ = _map(self.keys, self.mean, np.eye(len(self.mean)), cls, tk)
        _, Pres = _map(self.keys, np.zeros(len(self.mean)), self.res_cov, cls, tk)
        B, kinds = basis(cls, mu, ty, sl, sub, np.asarray(doy))
        return m, B @ np.diag([self.s_a[k] for k in kinds]) @ B.T + Pres


# ---------------- MAP-EM ----------------
def map_em(X, mu, prec, logdet, cls, m, Psi, lam=LAM, reg=1e-3):
    D = X.shape[1]; C = len(cls); S0 = {c: np.linalg.inv(prec[c]) for c in cls}; P = dict(prec); LD = dict(logdet)
    Pinv = np.linalg.inv(Psi + 1e-6 * np.eye(C * D)); rhs0 = Pinv @ m
    delta = {c: m[a * D:(a + 1) * D].copy() for a, c in enumerate(cls)}; pi = {c: 1.0 / C for c in cls}
    for _ in range(NIT):
        L = np.stack([np.log(pi[c]) - 0.5 * LD[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c] - delta[c], P[c], X - mu[c] - delta[c]) for c in cls], 1)
        R = np.exp(L - logsumexp(L, 1, keepdims=True)); A = Pinv.copy(); b = rhs0.copy()
        for a, c in enumerate(cls):
            w = R[:, a]; pi[c] = max(w.mean(), 1e-6); sw = max(w.sum(), 1e-9); sx = (w[:, None] * (X - mu[c])).sum(0)
            sl = slice(a * D, (a + 1) * D); A[sl, sl] += sw * P[c]; b[sl] += P[c] @ sx
        sol = np.linalg.solve(A, b); delta = {c: sol[a * D:(a + 1) * D] for a, c in enumerate(cls)}
        if lam > 0:
            for a, c in enumerate(cls):
                w = R[:, a]; sw = max(w.sum(), 1e-9); Xc = X - mu[c] - delta[c]
                Se = (w[:, None] * Xc).T @ Xc / sw + reg * np.eye(D); Sc = (1 - lam) * S0[c] + lam * Se
                P[c] = np.linalg.inv(Sc); LD[c] = np.linalg.slogdet(Sc)[1]
    L = np.stack([np.log(pi[c]) - 0.5 * LD[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c] - delta[c], P[c], X - mu[c] - delta[c]) for c in cls], 1)
    return np.array(cls)[L.argmax(1)], delta


def adapt(Es, ys, Et, tk, doy, arch):
    """Es (n_s x D) labelled source year (imputed), ys labels, Et (n_t x D) unlabelled target year (imputed with source medians)."""
    mu, prec, logdet, cls = gauss_fit(Es, ys)
    m, Psi = arch.prior(cls, mu, tk, doy)
    return map_em(Et, mu, prec, logdet, cls, m, Psi)


def resynthesize(Es, ys, delta):
    """C2 (mean shift as in P4): x_i <- x_i + delta_{y_i}; classes without delta unchanged."""
    E2 = Es.copy()
    for c, d in delta.items(): E2[ys == c] += d
    return E2
