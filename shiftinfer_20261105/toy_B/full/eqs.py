"""Building blocks for B-full candidate equations (only allowed inputs: source class means / derivatives, doy, polarisation,
class, KNMI RH/EV24 derivatives). All global parameters are fitted on archive tasks of the training site only."""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from p9_lib_snap import disc_B, vh_level, dmu_dt, fit_mw, mw_mean, api_at
HERE = os.path.dirname(os.path.abspath(__file__))


def vh_blocks(ctx):
    if '_vh' not in ctx: ctx['_vh'] = np.concatenate([vh_level(ctx['mu'][c], ctx['ty'], ctx['sl'], ctx['sub']) for c in ctx['cls']])
    return ctx['_vh']


def tile(ctx, v):
    v = np.asarray(v); return np.tile(v, len(ctx['cls'])) if v.ndim == 1 else np.tile(v, (len(ctx['cls']), 1))


def tyb(ctx):
    return tile(ctx, ctx['ty'])


# ---------- flattened archive (for global parameter fits) ----------
def flat(A, feats):
    """concatenate over archive tasks: ty, vh, y_raw (oracle), y_arch (oracle - m_arch), and requested weather features."""
    out = {k: [] for k in ['ty', 'vh', 'y_raw', 'y_arch', 'tid', 'cls'] + list(feats)}
    for i, t in enumerate(A):
        y = np.concatenate([t['orc'][c] for c in t['cls']]); ok = ~np.isnan(y)
        out['ty'].append(tyb(t)[ok]); out['vh'].append(vh_blocks(t)[ok]); out['y_raw'].append(y[ok]); out['y_arch'].append((y - t['m_arch'])[ok])
        out['cls'].append(np.repeat(t['cls'], len(t['ty']))[ok])
        out['tid'].append(np.full(ok.sum(), i))
        for f in feats: out[f].append(tile(t, f(t) if callable(f) else t.w(f))[ok])
    return {k: np.concatenate(v) for k, v in out.items()}


def wls(fun, th0, F, ykey, bounds=None):
    sd = pd.Series(F[ykey]).groupby(F['ty']).std().to_numpy(); wt = 1 / sd[F['ty']]
    r = least_squares(lambda th: (fun(F, th) - F[ykey]) * wt, np.asarray(th0, float), bounds=bounds if bounds else (-np.inf, np.inf))
    return r.x


# ---------- bases ----------
def B_disc(ctx, gs):
    return disc_B(ctx['cls'], ctx['mu'], ctx['ty'], ctx['sl'], ctx['sub'], ctx['doy'])


def B_disc_sep(ctx, gs):
    """date columns separate for VV and VH (each with VH attenuation), coherence per slot, per-class timing."""
    cls, mu, ty, sl, sub, doy = ctx['cls'], ctx['mu'], ctx['ty'], ctx['sl'], ctx['sub'], ctx['doy']; D = len(ty); cols, kinds = [], []
    att = {c: np.exp(-1.0 * (vh_level(mu[c], ty, sl, sub) + 25) / 10) for c in cls}
    for p in (0, 1):
        for s, u in sorted(set((int(sl[d]), int(sub[d])) for d in range(D) if ty[d] == p)):
            msk = (ty == p) & (sl == s) & (sub == u); cols.append(np.concatenate([att[c] * msk for c in cls])); kinds.append(('wbs', p))
    for s in sorted(set(int(sl[d]) for d in range(D) if ty[d] == 2)):
        cols.append(np.tile(((ty == 2) & (sl == s)).astype(float), len(cls))); kinds.append(('wcoh', 0))
    for a, c in enumerate(cls):
        v = np.zeros(len(cls) * D); v[a * D:(a + 1) * D] = dmu_dt(mu[c], ty, doy); cols.append(v); kinds.append(('tau', 0))
    return np.stack(cols, 1), kinds


# ---------- means ----------
def m_arch(ctx, gs):
    return ctx['m_arch']


def fit_mw_p9(A, P0):
    """replica of P9 hybridD_r*_mw: fit_mw on toy-B shift_table of the training site."""
    st = pd.read_csv(os.path.join(os.path.dirname(HERE), 'shift_table.csv'))
    L = st[st.dtype == 1][['group', 'src', 'cls', 'slot', 'sub', 'mu_s']].drop_duplicates(['group', 'src', 'cls', 'slot', 'sub']).rename(columns={'mu_s': 'vh_s'})
    st = st.merge(L, on=['group', 'src', 'cls', 'slot', 'sub'], how='left')
    st['vh_s'] = st.groupby(['group', 'src', 'cls', 'slot'])['vh_s'].transform(lambda s: s.fillna(s.mean()))
    return dict(thw=fit_mw(st, A[0]['site']))


def m_mw_p9(ctx, gs):
    if '_mwp9' not in ctx:
        ctx['_mwp9'] = mw_mean(gs['thw'], ctx['cls'], ctx['mu'], ctx['ty'], ctx['sl'], ctx['sub'], api_at(ctx['site'], ctx['dd_s']), api_at(ctx['site'], ctx['dd_t']))
    return ctx['_mwp9']


def g_api(ctx, k):
    return np.log1p(ctx.w('api_t') / k) - np.log1p(ctx.w('api_s') / k)


def wc_term(F_or_ctx, th, is_ctx):
    """a_p * exp(-b (VH+25)/10) * [ln(1+API_t/k) - ln(1+API_s/k)], a per dtype (VV, VH, COH)."""
    a, b, k = th[:3], abs(th[3]), abs(th[4]) + 1e-3
    if is_ctx:
        c = F_or_ctx; ty = tyb(c); vh = vh_blocks(c); g = tile(c, g_api(c, k))
    else:
        F = F_or_ctx; ty = F['ty']; vh = F['vh']; g = np.log1p(F['api_t'] / k) - np.log1p(F['api_s'] / k)
    return np.choose(ty, a) * np.exp(-b * (vh + 25) / 10) * g


def coh_rain_term(F_or_ctx, th, is_ctx):
    """coherence decorrelation by rain between the two acquisitions: c1 * [ln(1+R_t) - ln(1+R_s)] on COH dims."""
    if is_ctx:
        c = F_or_ctx; ty = tyb(c); g = tile(c, np.log1p(c.w('rainpair_t')) - np.log1p(c.w('rainpair_s')))
    else:
        F = F_or_ctx; ty = F['ty']; g = np.log1p(F['rainpair_t']) - np.log1p(F['rainpair_s'])
    return th[0] * (ty == 2) * g
