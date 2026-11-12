"""Fig. (archive size) - S3i: median BA against the number of labelled archive years K (0, 1, 2, 4, all = 6; source and target years always
excluded, same K years for every method of a transfer) and the paired gain of shifting the pooled seasons with the prior.
Sources: s3i_medians.csv, s3i_summary.csv (s3i_summary.py; K = 0 and K = all end points from S3h, archive without the transfer's seasons).
Row 1: median BA - pooled HGB training (K = 0: source season only), pooled training with every season shifted to the target by the prior,
single-source Gaussian classifier with the prior (K = 0: prior-free EM). Row 2: paired median difference (prior-shifted pooling - pooling)
with 90% year-block bootstrap interval, HGB and TempCNN."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, COL, MK, W2, GRID, INK2, INK
setup()
M = pd.read_csv(os.path.join(UP, 's3i_medians.csv')); S = pd.read_csv(os.path.join(UP, 's3i_summary.csv'))
XK = {'k0': 0, 'k1': 1, 'k2': 2, 'k4': 4, 'kall': 6}
REG = [('NL site B', 'Site B (archive: site A)'), ('NL site A', 'Site A (archive: site B)'), ('France', 'Pays de Caux (archive: the Netherlands)')]
LINES = [('hgb_pool', 'Pooled training (HGB)', COL['em'], MK['em']), ('hgb_poolsh', 'Pooled, seasons shifted by the prior (HGB)', COL['lw'], MK['lw']),
         ('g_lw', 'Source season + prior (Gaussian)', COL['hpj'], MK['hpj'])]
fig, axs = plt.subplots(2, 3, figsize=(W2, 3.9), gridspec_kw={'height_ratios': [1.25, 1]}, sharex=True)
for j, (reg, title) in enumerate(REG):
    ax = axs[0, j]; m = M[M.region == reg].set_index('K'); ax.grid(axis='y', color=GRID, lw=0.5)
    for col, lab, c, mk in LINES:
        ks = [k for k in XK if pd.notna(m.loc[k, col])]
        ax.plot([XK[k] for k in ks], [m.loc[k, col] for k in ks], color=c, marker=mk, ms=4, mec='white', mew=0.6, label=lab)
    ax.set_title(title, loc='left', color=INK2); tag(ax, 'abc'[j]); ax.set_ylim(0.5, 0.83)
    ax = axs[1, j]; ax.grid(axis='y', color=GRID, lw=0.5); ax.axhline(0, color=INK2, lw=0.6)
    for clf, lab, c, mk, off, face in (('hgb', 'HGB', COL['lw'], MK['lw'], -0.12, None), ('tcn', 'Temporal CNN', COL['dann'], MK['dann'], 0.12, 'white')):
        r = S[(S.region == reg) & S.a.str.startswith(f'{clf}_poolsh_')]
        x = np.array([XK[a.split('_')[-1]] for a in r.a]) + off
        ax.errorbar(x, r['median'], yerr=[r['median'] - r.lo90, r.hi90 - r['median']], fmt=mk, color=c, ms=4, capsize=0, elinewidth=0.8,
                    mfc=face or c, mec=c, label=lab)
    ax.set_ylim(-0.035, 0.075); tag(ax, 'def'[j]); ax.set_xticks([0, 1, 2, 4, 6]); ax.set_xticklabels(['0', '1', '2', '4', '6 (all)'])
    ax.set_xlabel('Labelled archive years')
axs[0, 0].set_ylabel('Balanced accuracy (median)'); axs[1, 0].set_ylabel('Gain of shifting\nthe pooled seasons')
h, l = axs[0, 0].get_legend_handles_labels(); h2, l2 = axs[1, 0].get_legend_handles_labels()
fig.legend(h + h2, l + [f'Gain, {x}' for x in l2], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(HERE, 'fig_archsize.pdf')); fig.savefig(os.path.join(HERE, 'fig_archsize.png'), dpi=300)
print('written')
