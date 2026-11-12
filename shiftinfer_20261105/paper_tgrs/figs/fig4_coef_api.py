"""Fig. 4 - label-free shared date offsets of the frozen method vs the antecedent-precipitation difference at the acquisition date.
Sources: diag_frozen_coef_<grp>.csv (NL, KNMI 283/280) and newregion/CONFIRM_FR_coef.csv (France, Meteo-France 76679001). One point per
(transfer, fold, acquisition date). Panels: (a) NL site B, (b) NL site A, (c) France FR-A. Spearman rho in each panel."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, COL, W2, INK2, GRID
setup()
c = pd.concat([pd.read_csv(os.path.join(UP, f'diag_frozen_coef_{g}.csv')) for g in ('A37', 'A88', 'B88')]); c = c[c.kind == 'wbs']
f = pd.read_csv(os.path.join(UP, 'newregion', 'CONFIRM_FR_coef.csv'))
P = [(c[c.group == 'B88'], 'Netherlands, site B'), (c[c.group.isin(['A37', 'A88'])], 'Netherlands, site A'), (f, 'France, Pays de Caux')]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
for ax, (d, title), t in zip(axs, P, 'abc'):
    x = d.api_t - d.api_s; y = d.coef; r = spearmanr(x, y)[0]
    ax.hexbin(x, y, gridsize=40, cmap='Blues', mincnt=1, linewidths=0, bins='log', rasterized=True)   # vector PDF hexbin offsets break in viewers
    ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
    ax.set_title(title, loc='left', color=INK2); ax.text(0.03, 0.95, rf'$\rho$ = {r:.2f}, $n$ = {len(d):,}', transform=ax.transAxes, va='top', fontsize=7)
    ax.set_xlabel(r'$\Delta$API at acquisition (mm)'); tag(ax, t)
axs[0].set_ylabel('Shared date offset $w_t$ (dB)')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig4_coef_api.pdf')); fig.savefig(os.path.join(HERE, 'fig4_coef_api.png'), dpi=300)
print({t: round(spearmanr(d.api_t - d.api_s, d.coef)[0], 3) for d, t in P})
