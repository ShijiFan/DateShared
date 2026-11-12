"""Toy B round 1: starting references and first skeletons (proposer = Claude). Results appended to toy_results.jsonl."""
import sys, numpy as np
from evalb import load, run_m1, run_m2, log

T = load(); C = sorted(T.cls.unique())
ty = lambda d, k: (d.dtype.to_numpy() == k).astype(float)
dA = lambda d: (d.API_t - d.API_s).to_numpy()
tsel = lambda d, th3: np.choose(d.dtype.to_numpy(), th3)           # per-dtype coefficient


# ---------- M1: weather-only prediction ----------
M1 = {
    'zero': (lambda d, th: np.zeros(len(d)), []),
    'api_lin': (lambda d, th: tsel(d, th[:3]) * dA(d), [0.01, 0.01, 0.0]),
    'api_sat': (lambda d, th: tsel(d, th[:3]) * (np.log1p(d.API_t.to_numpy() / abs(th[3])) - np.log1p(d.API_s.to_numpy() / abs(th[3]))), [1.0, 1.0, 0.0, 10.0]),
    # water-cloud intuition: soil-moisture sensitivity attenuated by vegetation (proxy: source VH level of the same class/date)
    'api_veg': (lambda d, th: tsel(d, th[:3]) * dA(d) * np.exp(-abs(th[3]) * (d.vh_s.to_numpy() + 25) / 10), [0.02, 0.02, 0.0, 0.5]),
    'p_short_long': (lambda d, th: tsel(d, th[:3]) * (d.P3_t - d.P3_s).to_numpy() + tsel(d, th[3:6]) * (d.P30_t - d.P30_s).to_numpy(), [0.01] * 6),
    # phenology timing driven by water balance difference: delta = tau * dmu, tau = c1 * dWB30
    'timing_wb': (lambda d, th: th[0] * ((d.P30_t - d.E30_t) - (d.P30_s - d.E30_s)).to_numpy() * d.dmu_s.to_numpy(), [0.1]),
}
# ---------- M2: structure with K per-task coefficients (oracle z) ----------
def hand(d, th):
    B = [ty(d, 0), ty(d, 1), ty(d, 2)] + [d.dmu_s.to_numpy() * (d.cls.to_numpy() == c) for c in C]
    return np.stack(B, 1)
def level_cls(d, th):
    return np.stack([ty(d, k) * (d.cls.to_numpy() == c) for k in range(3) for c in C], 1)
def hand_stretch(d, th):
    s = (d.doy_s.to_numpy() - 175) / 60.0
    return np.hstack([hand(d, th), np.stack([d.dmu_s.to_numpy() * s * (d.cls.to_numpy() == c) for c in C], 1)])
def level3(d, th): return np.stack([ty(d, 0), ty(d, 1), ty(d, 2)], 1)
def time7(d, th): return np.stack([d.dmu_s.to_numpy() * (d.cls.to_numpy() == c) for c in C], 1)
M2 = {'level3': (level3, 3), 'time7': (time7, 7), 'hand': (hand, 10), 'hand_stretch': (hand_stretch, 17), 'level_cls': (level_cls, 21)}

if __name__ == '__main__':
    which = sys.argv[1:] or list(M1) + list(M2)
    for n in which:
        if n in M1:
            f, th = M1[n]; r = run_m1(f, np.array(th, float), T); log(n, 'M1', r, 'r1')
            print('M1', n, {k: (round(v['test']['BS'], 3), round(v['test']['COH'], 3), v['theta']) for k, v in r.items()}, flush=True)
        else:
            g, K = M2[n]; r = run_m2(g, np.array([]), K, T); log(n, 'M2', r, 'r1')
            print('M2', n, 'K', K, {k: (round(v['test']['all'], 3), 'pcaK', round(v['pcaK_test']['all'], 3), 'VV', round(v['test']['VV'], 3), 'VH', round(v['test']['VH'], 3), 'COH', round(v['test']['COH'], 3)) for k, v in r.items()}, flush=True)
