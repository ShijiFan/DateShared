"""Fig. (paired) - the advantage holds transfer by transfer, not only in the median. Paired per-transfer differences in balanced accuracy,
proposed (TempCNN on the season resynthesised with the frozen-method shifts) minus each baseline, in the three regions.
Same merge and transfers as s3d_table.py / Table tab:sota: NL site B = B88 (28), NL site A = A37 + A88 (56), France (56).
Sources: ../../s3d_{B88,A37,A88,FR}.csv (RAINCOAT-SRS rule A, SFUDA FR), ../../s3c_official_*.csv (SFUDA NL), ../../s3a_deep_*.csv
(AdaBN, DANN NL), ../../s3j_deep_*.csv (proposed NL, archive without the transfer's years; S3j), ../../newregion/FR_DEEP_rows.csv (FR deep rows)."""
import os, sys, importlib.util
import numpy as np, pandas as pd, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, W2, INK, INK2, GRID
setup()
spec = importlib.util.spec_from_file_location('s3dt', os.path.join(UP, 's3d_table.py')); src = open(os.path.join(UP, 's3d_table.py'), encoding='utf-8').read()
G = {'__file__': os.path.join(UP, 's3d_table.py')}; exec(src.split('T = {')[0], G)             # nl(), fr() merges only
T = {'Site B': G['nl'](['B88']), 'Site A': G['nl'](['A37', 'A88']), 'Pays de Caux': G['fr']()}
BL = [('srs_A', 'RAINCOAT-SRS', '#eb6834'), ('sfuda', 'SFUDA', '#eda100'), ('dl_adabn', 'AdaBN', '#1baf7a'), ('dl_dann', 'DANN', '#6250d6')]
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.0), sharex=True, gridspec_kw=dict(wspace=0.08, left=0.125, right=0.99, top=0.88, bottom=0.2))
rows = []; rng = np.random.default_rng(0)
for ax, (reg, d) in zip(axs, T.items()):
    for i, (k, lab, col) in enumerate(BL):
        dd = (d['tcn_lw'] - d[k]).to_numpy(); y = len(BL) - 1 - i
        ax.scatter(dd, y + rng.uniform(-0.18, 0.18, len(dd)), s=5, color=col, alpha=0.75, lw=0)
        q = np.percentile(dd, [25, 50, 75]); ax.plot([q[0], q[2]], [y, y], color=INK, lw=1.6, solid_capstyle='butt'); ax.plot(q[1], y, '|', color=INK, ms=7, mew=1.6)
        share = (dd > 0).mean(); ax.text(0.425, y, f'{100 * share:.0f}%', fontsize=6.5, va='center', ha='right', color=INK2)
        rows.append(dict(region=reg, baseline=lab, n=len(dd), median=float(q[1]), q25=float(q[0]), q75=float(q[2]), share_pos=float(share)))
    ax.axvline(0, color=INK2, lw=0.6); ax.grid(axis='x', color=GRID, lw=0.4); ax.set_title(f'{reg} ({len(d)} transfers)', fontsize=7.5, pad=3)
    ax.set_yticks(range(len(BL))); ax.set_yticklabels([b[1] for b in BL][::-1] if ax is axs[0] else []); ax.set_xlim(-0.13, 0.43); ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3])
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
axs[1].set_xlabel(r'$\Delta$BA, proposed $-$ baseline'); [tag(a, s) for a, s in zip(axs, 'abc')]
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'figN_paired.{e}'), dpi=600 if e == 'png' else None)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'figN_paired_data.csv'), index=False); print(R.round(3).to_string(index=False))
