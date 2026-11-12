"""S3i shared: archive build with the diagonal fallback (verbatim copy of build() in s3i_curve.py) and the K-year archive sets."""
import zlib
import numpy as np
import s3h_lib as L
F = L.F


def build(ex):
    """S3i amendment (2026-11-12 02:45, EXPLORE_S3D_Q1_ZH.md): F.Archive.build unchanged whenever it works. With K = 2 archive years it can
    happen that no task covers every archive key, so LedoitWolf gets no rows and fails. Then the archive is built the same way except that
    the residual covariance is diagonal: per-key mean squared residual over all tasks where it is observed (keys never observed: mean)."""
    try: return F.Archive.build(ex), False
    except ValueError: pass
    tasks = sorted(ex, key=lambda t: t['id'])
    keys = sorted({tuple(int(x) for x in k) for t in tasks for c in t['cls'] if np.isfinite(t['orc'][c]).all() for k in t['tk']})
    kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); V = []
    for t in tasks:
        v = np.full(F.K * nk, np.nan)
        for c in t['cls']:
            o = t['orc'][c]
            for i, k in enumerate(t['tk']):
                if np.isfinite(o[i]) and tuple(int(x) for x in k) in kix: v[c * nk + kix[tuple(int(x) for x in k)]] = o[i]
        V.append(v)
    V = np.array(V); mean = np.nanmean(V, 0); mean = np.where(np.isfinite(mean), mean, 0.0)
    coef, R = {}, []
    for t in tasks:
        cls, tk = t['cls'], [tuple(int(x) for x in k) for k in t['tk']]
        B, kinds = F.basis(cls, t['mu'], t['ty'], t['sl'], t['sub'], t['doy'])
        m, _ = F._map(keys, mean, np.eye(len(mean)), cls, tk)
        y = np.concatenate([t['orc'][c] for c in cls]) - m; ok = ~np.isnan(y)
        th, *_ = np.linalg.lstsq(B[ok], y[ok], rcond=None); r = np.full(len(y), np.nan); r[ok] = y[ok] - B[ok] @ th
        for kd, v in zip(kinds, th): coef.setdefault(kd, []).append(v)
        D = len(tk); rv = np.full(F.K * nk, np.nan)
        for a, c in enumerate(cls):
            for i, k in enumerate(tk):
                if k in kix: rv[c * nk + kix[k]] = r[a * D + i]
        R.append(rv)
    var = np.nanmean(np.square(np.array(R)), 0); var = np.where(np.isfinite(var), var, np.nanmean(var))
    return F.Archive(keys, mean, {kd: float(np.mean(np.square(v))) for kd, v in coef.items()}, np.diag(var), {'diag_fallback': True}), True


def sets(G, ya, yb, S, tasks, KS):
    """Same draw as s3i_curve.py: archive years minus source/target, permuted with crc32('S3i-G-ya-yb'), first K years (nested)."""
    PSall = [p for p in L.pool_seasons(G, S) if p['y'] not in (ya, yb)]
    AY = sorted({int(p['y']) for p in PSall})
    order = [int(y) for y in np.random.default_rng(zlib.crc32(f'S3i-{G}-{ya}-{yb}'.encode())).permutation(AY)]
    out = {}
    for K in KS:
        yrs = set(order[:K]); PS = [p for p in PSall if int(p['y']) in yrs]
        ex = [t for t in tasks if {int(t['id'][1]), int(t['id'][2])} <= yrs]
        AX, fb = build(ex) if K >= 2 else (None, False)
        if fb: print('diag_fallback', G, ya, yb, 'K', K, 'years', sorted(yrs), 'tasks', len(ex), flush=True)
        out[K] = dict(yrs=sorted(yrs), PS=PS, YP=np.concatenate([p['Y'] for p in PS]), AX=AX, n_tasks=len(ex))
    return out
