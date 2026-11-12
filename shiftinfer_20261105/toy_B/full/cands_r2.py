"""B-full round 2. Reflection after the P9 interim summary (testA, archive B): every low-rank basis prior (physD, hybridD,
learn_r*) is below hpj's full-rank LW prior (0.56-0.60 vs 0.63), while learn_r8 reaches 0.67 -> the 80%-rank structured
prior is over-confident, not the equations per se. So attach the discovered equations to the full-rank joint prior instead
of replacing it: (i) weather mean on top of the archive mean, (ii) LW prior re-estimated on weather residuals
('weather-conditioned joint prior'), (iii) extra prior variance along the discovered date/timing directions."""
import numpy as np
from eqs import B_disc, flat, wls, wc_term, coh_rain_term

_F = {}
def F_of(A):
    k = (A[0]['site'], len(A))
    if k not in _F: _F[k] = flat(A, ['api_s', 'api_t', 'rainpair_s', 'rainpair_t'])
    return _F[k]


def fit_wc(A, P0):
    return dict(th=wls(lambda F, th: wc_term(F, th, False), [1.5, 1.5, 0.0, 1.0, 8.0], F_of(A), 'y_arch'))


def fit_wc_coh(A, P0):
    f = lambda F, th: wc_term(F, th[:5], False) + coh_rain_term(F, th[5:], False)
    return dict(th=wls(f, [1.5, 1.5, 0.0, 1.0, 8.0, -0.02], F_of(A), 'y_arch'))


w0 = lambda ctx, gs: 0.0
w_wc = lambda ctx, gs: wc_term(ctx, gs['th'], True)
w_wc_coh = lambda ctx, gs: wc_term(ctx, gs['th'][:5], True) + coh_rain_term(ctx, gs['th'][5:], True)

CANDS = [
    dict(name='em_cov01'), dict(name='hpj_cov01'),
    dict(name='j0_hpj_rep', joint=True, wfun=w0),                                     # must equal hpj_cov01
    dict(name='j1_hpj_wc', joint=True, fit=fit_wc, wfun=w_wc),
    dict(name='j2_hpj_wc_cond', joint=True, fit=fit_wc, wfun=w_wc, cond=True),
    dict(name='j3_hpj_wccoh_cond', joint=True, fit=fit_wc_coh, wfun=w_wc_coh, cond=True),
    dict(name='j4_hpj_plusD', joint=True, wfun=w0, basis=B_disc),
    dict(name='j5_hpj_wc_cond_plusD', joint=True, fit=fit_wc, wfun=w_wc, cond=True, basis=B_disc),
]
