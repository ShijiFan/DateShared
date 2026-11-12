"""B-full round 3. Reflection on r2: attaching the weather equation to the full-rank joint prior helps only when the LW
covariance is re-estimated on weather residuals (j2 +0.004/+0.001 median vs hpj, j1 without conditioning is negative); the
gain is small because the equation explains ~1/3 of the shift. So enrich the weather equation itself (all parameters fitted
on the archive site only) and keep the j2 attachment: (k1) API decay as a fitted parameter, (k2) class-specific amplitude,
(k3) an additional short-term surface-wetting term (3-day rain), (k4) API of net water input P - alpha*E, (k5) k1 + the
discovered date/timing directions as extra prior variance (j5 analogue)."""
import numpy as np
from eqs import B_disc, flat, wls, tile, tyb, vh_blocks, wc_term

FEATS = ['api_s', 'api_t', 'r60_s', 'r60_t', 'e60_s', 'e60_t', 'P3_s', 'P3_t']
_F = {}
def F_of(A):
    k = tuple(t['id'] for t in A)          # was (site, len(A)): equivalent for one archive per site, unsafe for subsets
    if k not in _F: _F[k] = flat(A, FEATS)
    return _F[k]


def get(o, is_ctx):
    if not is_ctx: return o.__getitem__
    def g(n):
        if n == 'ty': return tyb(o)
        if n == 'vh': return vh_blocks(o)
        if n == 'cls': return np.repeat(o['cls'], len(o['ty']))
        return tile(o, o.w(n))
    return g


def sig(q): return 1 / (1 + np.exp(-q))
def lg(x, k): return np.log1p(np.maximum(x, 0) / k)


def eq_decay(o, th, c=False):           # th: aVV aVH aCOH b k q(decay)
    g = get(o, c); d = sig(th[5]); w = d ** np.arange(60); k = abs(th[4]) + 1e-3
    G = lg(g('r60_t') @ w, k) - lg(g('r60_s') @ w, k)
    return np.choose(g('ty'), th[:3]) * np.exp(-abs(th[3]) * (g('vh') + 25) / 10) * G


def eq_cls(o, th, c=False):             # th: a_bs[cls 0..6] (7), aCOH, b, k ; same amplitude for VV and VH, VH carries attenuation
    g = get(o, c); ty = g('ty'); k = abs(th[9]) + 1e-3
    G = lg(g('api_t'), k) - lg(g('api_s'), k); a = np.where(ty == 2, th[7], th[:7][g('cls')])
    return a * np.exp(-abs(th[8]) * (g('vh') + 25) / 10) * G


def eq_short(o, th, c=False):           # th: aVV aVH aCOH b k  sVV sVH k2
    g = get(o, c); ty = g('ty'); att = np.exp(-abs(th[3]) * (g('vh') + 25) / 10); k, k2 = abs(th[4]) + 1e-3, abs(th[7]) + 1e-3
    long_ = np.choose(ty, th[:3]) * (lg(g('api_t'), k) - lg(g('api_s'), k))
    short = np.choose(ty, [th[5], th[6], 0.0]) * (lg(g('P3_t'), k2) - lg(g('P3_s'), k2))
    return att * (long_ + short)


def eq_net(o, th, c=False):             # th: aVV aVH aCOH b k alpha ; API of P - alpha*E (decay 0.9)
    g = get(o, c); w = 0.9 ** np.arange(60); k = abs(th[4]) + 1e-3; al = abs(th[5])
    G = lg((g('r60_t') - al * g('e60_t')) @ w, k) - lg((g('r60_s') - al * g('e60_s')) @ w, k)
    return np.choose(g('ty'), th[:3]) * np.exp(-abs(th[3]) * (g('vh') + 25) / 10) * G


def mk(eq, th0):
    fit = lambda A, P0: dict(th=wls(lambda F, th: eq(F, th, False), th0, F_of(A), 'y_arch'))
    wfun = lambda ctx, gs: eq(ctx, gs['th'], True)
    return fit, wfun


f_wc = lambda A, P0: dict(th=wls(lambda F, th: wc_term(F, th, False), [1.5, 1.5, 0.0, 1.0, 8.0], F_of(A), 'y_arch'))
w_wc = lambda ctx, gs: wc_term(ctx, gs['th'], True)
f1, w1 = mk(eq_decay, [1.5, 1.5, 0.0, 1.0, 8.0, 2.197])
f2, w2 = mk(eq_cls, [1.5] * 7 + [0.0, 1.0, 8.0])
f3, w3 = mk(eq_short, [1.5, 1.5, 0.0, 1.0, 8.0, 0.5, 0.5, 5.0])
f4, w4 = mk(eq_net, [1.5, 1.5, 0.0, 1.0, 8.0, 0.5])

CANDS = [
    dict(name='em_cov01'), dict(name='hpj_cov01'),
    dict(name='j2_hpj_wc_cond', joint=True, fit=f_wc, wfun=w_wc, cond=True),
    dict(name='j5_hpj_wc_cond_plusD', joint=True, fit=f_wc, wfun=w_wc, cond=True, basis=B_disc),
    dict(name='k1_decay_cond', joint=True, fit=f1, wfun=w1, cond=True),
    dict(name='k2_clsamp_cond', joint=True, fit=f2, wfun=w2, cond=True),
    dict(name='k3_short_cond', joint=True, fit=f3, wfun=w3, cond=True),
    dict(name='k4_net_cond', joint=True, fit=f4, wfun=w4, cond=True),
    dict(name='k5_decay_cond_plusD', joint=True, fit=f1, wfun=w1, cond=True, basis=B_disc),
]
