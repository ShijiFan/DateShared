"""Fig. (sens) - the frozen method is insensitive to its three settings. One-at-a-time sensitivity (EXPLORE_S3D_Q1_ZH.md, S3f): canopy
attenuation b (archive rebuilt for each b), covariance update lambda and the number of EM iterations; all other settings frozen
(b = 1, lambda = 0.1, 30 iterations). Median BA of the Gaussian classifier over the transfers todo[0::2] at site B (B88, 28) and site A
(A37 + A88, 56); dashed: prior-free mean-shift EM at the same transfers. Data: ../../s3f_sens_{B88,A37,A88}.csv (s3f_core.py)."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, W2, INK, INK2, GRID, COL
setup()
S = pd.concat([pd.read_csv(os.path.join(UP, f's3f_sens_{g}.csv')) for g in ('B88', 'A37', 'A88')]).drop_duplicates(['group', 'src', 'tgt', 'config'])
S['site'] = np.where(S.group == 'B88', 'B', 'A')
# s3f_core.py appends sens rows (5 columns) under the 7-column header of the main run, so their proposed BA sits in column ba_raw
S.loc[S.config != 'frozen', 'ba_lw'] = S.loc[S.config != 'frozen', 'ba_raw']
FROZ = {'b': 1.0, 'lam': 0.1, 'nit': 30.0}
GR = {'b': [0, 0.5, 1, 1.5, 2], 'lam': [0, 0.05, 0.1, 0.2, 0.3], 'nit': [10, 20, 30, 50]}
XL = {'b': 'Canopy attenuation $b$', 'lam': r'Covariance update $\lambda$', 'nit': 'EM iterations'}
def cfg(p, v): return 'frozen' if v == FROZ[p] else f'{p}={v:g}'
fig, axs = plt.subplots(1, 3, figsize=(W2, 1.85), gridspec_kw=dict(wspace=0.3, left=0.075, right=0.99, top=0.9, bottom=0.24)); rows = []
F0 = S[S.config == 'frozen'].set_index(['group', 'src', 'tgt'])
for ax, p in zip(axs, GR):
    for site, mk, fill in (('B', 'o', True), ('A', 's', False)):
        ys = []
        for v in GR[p]:
            d = S[(S.config == cfg(p, v)) & (S.site == site)].set_index(['group', 'src', 'tgt'])
            dd = d.ba_lw - F0.loc[d.index, 'ba_lw']; ys.append(d.ba_lw.median())
            rows.append(dict(param=p, value=v, site=site, n=len(d), median_ba=float(d.ba_lw.median()), d_med_vs_frozen=float(dd.median()), share_better=float((dd > 0).mean())))
        ax.plot(GR[p], ys, mk + '-', color=COL['lw'], mfc=COL['lw'] if fill else 'white', ms=3.6, lw=1.0, label=f'Site {site}')
        ax.axhline(F0[F0.site == site].ba_em.median(), color=INK2, lw=0.7, ls=(0, (3, 2)), label='Prior-free EM' if site == 'A' else None)
    ax.axvline(FROZ[p], color=GRID, lw=3, zorder=0); ax.set_xlabel(XL[p]); ax.grid(axis='y', color=GRID, lw=0.4); ax.set_xticks(GR[p])
    if p == 'b': ax.set_ylabel('Median BA'); ax.legend(frameon=False, fontsize=6, loc='lower right')
[tag(a, s) for a, s in zip(axs, 'abc')]
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'figN_sens.{e}'), dpi=600 if e == 'png' else None)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'figN_sens_data.csv'), index=False); print(R.round(4).to_string(index=False))
