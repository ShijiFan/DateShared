"""Toy B round 2 (reflection on r1): PCA >> hand basis at equal K, and per-date API explains ~1/3 of backscatter shift.
Hypothesis: dominant mode = per-acquisition-date soil-moisture offset shared by all crops, attenuated by vegetation
(water-cloud: d sigma0_c(t) ~ exp(-b V_c(t)) * d sigma0_soil(t)), V proxy = source VH level of the class at that date."""
import sys, numpy as np
from evalb import load, run_m1, run_m2, log

T = load(); C = sorted(T.cls.unique())
ty = lambda d, k: (d.dtype.to_numpy() == k).astype(float)
tsel = lambda d, th3: np.choose(d.dtype.to_numpy(), th3)
att = lambda d, b: np.exp(-abs(b) * (d.vh_s.to_numpy() + 25) / 10)
gA = lambda d, k: np.log1p(d.API_t.to_numpy() / abs(k)) - np.log1p(d.API_s.to_numpy() / abs(k))

M1 = {
    'wc_api': (lambda d, th: tsel(d, th[:3]) * att(d, th[3]) * gA(d, th[4]), [1.5, 1.5, 0.0, 0.5, 3.0]),
    'wc_api_pol': (lambda d, th: np.where(d.dtype == 0, th[0] * att(d, th[3]), np.where(d.dtype == 1, th[1] * att(d, th[4]), th[2])) * gA(d, th[5]), [1.5, 1.5, 0.0, 0.5, 0.5, 3.0]),
}
bs = lambda d: (d.dtype.to_numpy() < 2)
def date_cols(d):
    """one column per (slot, sub) date shared by VV and VH (backscatter), plus one per slot for COH."""
    dk = [(s, u) for s in range(9) for u in (0, 1)]; cols = []
    for s, u in dk: cols.append(bs(d) * (d.slot.to_numpy() == s) * (d['sub'].to_numpy() == u))
    for s in range(9): cols.append(ty(d, 2) * (d.slot.to_numpy() == s))
    return np.stack(cols, 1).astype(float)
def date_lvl(d, th): return date_cols(d)
def date_lvl_wc(d, th):
    w = np.where(bs(d), att(d, th[0]), 1.0); return date_cols(d) * w[:, None]
def wx_hand(d, th):
    wx = np.where(bs(d), att(d, th[0]), 0.0) * gA(d, th[1])
    return np.stack([wx, ty(d, 0), ty(d, 1), ty(d, 2)] + [d.dmu_s.to_numpy() * (d.cls.to_numpy() == c) for c in C], 1)
M2 = {'date_lvl': (date_lvl, [], 27), 'date_lvl_wc': (date_lvl_wc, [0.5], 27), 'wx_hand': (wx_hand, [0.5, 3.0], 11)}

if __name__ == '__main__':
    which = sys.argv[1:] or list(M1) + list(M2)
    for n in which:
        if n in M1:
            f, th = M1[n]; r = run_m1(f, np.array(th, float), T); log(n, 'M1', r, 'r2')
            print('M1', n, {k: (round(v['test']['VV'], 3), round(v['test']['VH'], 3), round(v['test']['BS'], 3), v['theta']) for k, v in r.items()}, flush=True)
        else:
            g, th, K = M2[n]; r = run_m2(g, np.array(th, float), K, T); log(n, 'M2', r, 'r2')
            print('M2', n, 'K', K, {k: (round(v['test']['all'], 3), 'pcaK', round(v['pcaK_test']['all'], 3), 'VV', round(v['test']['VV'], 3), 'VH', round(v['test']['VH'], 3), 'COH', round(v['test']['COH'], 3), v['theta']) for k, v in r.items()}, flush=True)
