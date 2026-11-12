"""B-full round 1: references, P9 replicas (reproduction check), and the first candidate equations.
Motivation (toy B -> BA): P9's hybridD_r*_mw puts the weather mean on top of scales estimated around the archive mean;
if the weather term explains part of the shift, the remaining coefficient variance should shrink (c1, c2). Coherence:
rain between the two acquisitions of a pair decorrelates (c3). VV and VH may not share one per-date offset (c4)."""
import numpy as np
from eqs import B_disc, B_disc_sep, m_arch, fit_mw_p9, m_mw_p9, flat, wls, wc_term, coh_rain_term

_F = {}
def F_of(A):
    k = (A[0]['site'], len(A))
    if k not in _F: _F[k] = flat(A, ['api_s', 'api_t', 'rainpair_s', 'rainpair_t'])
    return _F[k]


def fit_wcA(A, P0):
    th = wls(lambda F, th: wc_term(F, th, False), [1.5, 1.5, 0.0, 1.0, 8.0], F_of(A), 'y_arch'); return dict(th=th)


def m_wcA(ctx, gs):
    return ctx['m_arch'] + wc_term(ctx, gs['th'], True)


def fit_wcA_coh(A, P0):
    f = lambda F, th: wc_term(F, th[:5], False) + coh_rain_term(F, th[5:], False)
    th = wls(f, [1.5, 1.5, 0.0, 1.0, 8.0, -0.02], F_of(A), 'y_arch'); return dict(th=th)


def m_wcA_coh(ctx, gs):
    return ctx['m_arch'] + wc_term(ctx, gs['th'][:5], True) + coh_rain_term(ctx, gs['th'][5:], True)


CANDS = [
    dict(name='em_cov01'), dict(name='hpj_cov01'),
    dict(name='rep_hybridD', mean=m_arch, basis=B_disc, ppca=True),
    dict(name='rep_hybridD_mw', fit=fit_mw_p9, mean=m_mw_p9, scale_mean=m_arch, basis=B_disc, ppca=True),
    dict(name='c1_wcA', fit=fit_wcA, mean=m_wcA, basis=B_disc, ppca=True),
    dict(name='c2_mwp9_refit', fit=fit_mw_p9, mean=m_mw_p9, basis=B_disc, ppca=True),
    dict(name='c3_wcA_cohrain', fit=fit_wcA_coh, mean=m_wcA_coh, basis=B_disc, ppca=True),
    dict(name='c4_sepVVVH', mean=m_arch, basis=B_disc_sep, ppca=True),
]
