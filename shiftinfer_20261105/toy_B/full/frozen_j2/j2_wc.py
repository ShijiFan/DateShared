"""FROZEN SECONDARY ARM j2 (weather-conditioned joint prior; B-full CONFIRM_ZH.md, user decision 2026-11-06: chapter
"meteorological prior enhancement", pre-registered secondary arm N6 of the new-region confirmation). Inference only;
weights (keys, conditioned LW mean/cov, weather parameters th) are produced by build_and_verify_j2.py from the NL archive
with the exact B-full evaluator code, then frozen.

Prior: delta ~ N(m_cond + w(ctx), Sigma_cond), w = a_p * exp(-b (VH_src_c + 25)/10) * [ln(1+API_t/k) - ln(1+API_s/k)],
a_p per dtype (VV, VH, COH), API = sum_{j=0..59} 0.9^j P(t-j) at the acquisition (pair end date for COH), daily precipitation
in mm. Sigma_cond / m_cond = Ledoit-Wolf over complete archive tasks of (oracle shift - w). MAP-EM and source model are those
of the frozen main method (frozen/hybridD_lw.py, imported unchanged).

API:
  arch = J2Archive.load(path)
  labels, delta = adapt(Es, ys, Et, tk, api_s, api_t, arch)    # api_s/api_t: per task dim (len D), source / target year"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'frozen'))
import hybridD_lw as H

DECAY = 0.9; WIN = 60


def api(P, dates, decay=DECAY, win=WIN):
    """P: daily precipitation (mm, pd.Series indexed by date, negatives already clipped); same formula as evalba.api."""
    out = []
    for t in pd.DatetimeIndex(dates):
        p = P.loc[t - pd.Timedelta(days=win - 1):t].to_numpy()[::-1]; out.append(float((p * decay ** np.arange(len(p))).sum()))
    return np.array(out)


def api_for_task(P, m, tk):
    """API per task dim from a features meta entry m (dict with 'slots', 'dates' as in features_*_meta.json; dates[2p+u] is
    acquisition u of slot index p): backscatter dims use their own acquisition, COH dims the pair end date (= evalba.pair_dates)."""
    dt = pd.to_datetime(m['dates']); d1 = []
    for t, q, u in [tuple(int(x) for x in k) for k in tk]:
        p = m['slots'].index(q); d1.append(dt[2 * p + u] if t < 2 else dt[2 * p + 1])
    return api(P, d1)


def wc(th, cls, mu, tk, api_s, api_t):
    tk = [tuple(int(x) for x in k) for k in tk]; ty = np.array([k[0] for k in tk]); sl = np.array([k[1] for k in tk]); sub = np.array([k[2] for k in tk])
    a, b, k = np.asarray(th[:3]), abs(th[3]), abs(th[4]) + 1e-3
    vh = np.concatenate([H._vh_level(mu[c], ty, sl, sub) for c in cls]); C = len(cls)
    g = np.tile(np.log1p(np.asarray(api_t) / k) - np.log1p(np.asarray(api_s) / k), C)
    return np.choose(np.tile(ty, C), a) * np.exp(-b * (vh + 25) / 10) * g


class J2Archive:
    def __init__(self, keys, mean, cov, th, meta=None):
        self.keys = [tuple(int(x) for x in k) for k in keys]; self.mean = np.asarray(mean); self.cov = np.asarray(cov)
        self.th = np.asarray(th, float); self.meta = meta or {}

    def save(self, path):
        np.savez_compressed(path, keys=np.array(self.keys), mean=self.mean, cov=self.cov, th=self.th, meta=json.dumps(self.meta))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return J2Archive(z['keys'].tolist(), z['mean'], z['cov'], z['th'], json.loads(str(z['meta'])))

    def prior(self, cls, mu, tk, api_s, api_t):
        m, P = H._map(self.keys, self.mean, self.cov, cls, [tuple(int(x) for x in k) for k in tk])
        return m + wc(self.th, cls, mu, tk, api_s, api_t), P


def adapt(Es, ys, Et, tk, api_s, api_t, arch):
    mu, prec, logdet, cls = H.gauss_fit(Es, ys)
    m, Psi = arch.prior(cls, mu, tk, api_s, api_t)
    return H.map_em(Et, mu, prec, logdet, cls, m, Psi)
