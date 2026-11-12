"""Fig. 5 - gain over DANN by shift severity (terciles of the TempCNN no-adaptation BA; low BA = severe shift), for the proposed prior used
directly (Gaussian) and as resynthesis before DANN. Sources: results/BASELINES.json (NL gauss_lw - dl_dann), P11D_SUMMARY.json (NL dann_lw -
dl_dann), newregion/FR_DEEP_SUMMARY.json (France). Bars: paired median difference; labels: share of transfers positive."""
import os, sys, json
import numpy as np, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, COL, W2, INK2, GRID
setup()
EX = json.load(open(os.path.join(UP, 'results', 'EXCL.json')))   # S3j 2026-11-12: NL archive without the transfer's seasons
F = json.load(open(os.path.join(UP, 'newregion', 'FR_DEEP_SUMMARY.json'))); LV = ('severe', 'mid', 'mild')
panels = [('Netherlands, site B', [EX['testB88_priorA']['by_severity'][l]['lw_x - dl_dann'] for l in LV], [EX['testB88_priorA']['by_severity'][l]['dann_lw_x - dl_dann'] for l in LV]),
          ('Netherlands, site A', [EX['testA_priorB']['by_severity'][l]['lw_x - dl_dann'] for l in LV], [EX['testA_priorB']['by_severity'][l]['dann_lw_x - dl_dann'] for l in LV]),
          ('France, Pays de Caux', [F['by_severity'][l]['lw_all - dl_dann'] for l in LV], [F['by_severity'][l]['dann_lw - dl_dann'] for l in LV])]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), sharey=True); x = np.arange(3); w = 0.36
for ax, (title, a, b), t in zip(axs, panels, 'abc'):
    for off, d, k, lab in ((-w / 2, a, 'lw', 'Proposed prior vs. DANN'), (w / 2, b, 'dann', 'DANN + resynthesis vs. DANN')):
        v = [r['median'] for r in d]; bars = ax.bar(x + off, v, width=w - 0.04, color=COL[k], label=lab)
        for xi, r in zip(x + off, d): ax.text(xi, r['median'] + (0.002 if r['median'] >= 0 else -0.002), f"{100 * r['frac_pos']:.0f}%", ha='center',
                                               va='bottom' if r['median'] >= 0 else 'top', fontsize=6, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(x); ax.set_xticklabels(['Largest\nshift', 'Middle', 'Smallest\nshift'])
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_title(title, loc='left', color=INK2); ax.set_ylim(-0.045, 0.045); tag(ax, t)
axs[0].set_ylabel(r'$\Delta$BA vs. DANN (median)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(HERE, 'fig5_severity.pdf')); fig.savefig(os.path.join(HERE, 'fig5_severity.png'), dpi=300)
