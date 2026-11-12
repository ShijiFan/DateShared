"""Fig. 3 - archive length: median BA vs number of archive years (3/5/8) for the empirical full-rank prior (hpj), the date-shared prior with
low-rank residual (P9 hybridD_r*) and the frozen date-shared prior (hybridD_lw). Source: p10_curve_<grp>.csv (todo[0::2]; k3/k5 = mean of
4 year draws per transfer). Panels: test site B (archive A), test site A (archive B). Error bars: interquartile range over transfers."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, COL, MK, W2, GRID, INK2
setup()
C = pd.concat([pd.read_csv(os.path.join(UP, f'p10_curve_{g}.csv')) for g in ('A37', 'A88', 'B88')], ignore_index=True)
V = {'hpj_cov01': ('hpj', 'Empirical full-rank prior'), 'hybridD_r*': ('lowrank', 'Date-shared basis + low-rank residual'), 'hybridD_lw': ('lw', 'Date-shared prior (proposed)')}
def lvl(d, L, v): return d[[f'k{L}_{i}|{v}' for i in range(4)]].mean(axis=1) if L != 8 else d[f'k8_all|{v}']
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3))
for ax, (gs, title), t in zip(axs, ((['B88'], 'Test site B, archive from site A'), (['A37', 'A88'], 'Test site A, archive from site B')), 'ab'):
    d = C[C.group.isin(gs)]; ax.grid(axis='y', color=GRID, lw=0.5)
    for i, (v, (k, lab)) in enumerate(V.items()):
        x = np.array([3, 5, 8]) + (i - 1) * 0.12; med = [lvl(d, L, v).median() for L in (3, 5, 8)]
        q1 = [lvl(d, L, v).quantile(0.25) for L in (3, 5, 8)]; q3 = [lvl(d, L, v).quantile(0.75) for L in (3, 5, 8)]
        ax.errorbar(x, med, yerr=[np.subtract(med, q1), np.subtract(q3, med)], color=COL[k], marker=MK[k], ms=4, capsize=0, elinewidth=0.6, label=lab,
                    mec='white', mew=0.6)
    ax.set_xticks([3, 5, 8]); ax.set_xlabel('Archive length (years)'); ax.set_title(title, loc='left', color=INK2); tag(ax, t)
axs[0].set_ylabel('Balanced accuracy (median)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(HERE, 'fig3_archive.pdf')); fig.savefig(os.path.join(HERE, 'fig3_archive.png'), dpi=300)
