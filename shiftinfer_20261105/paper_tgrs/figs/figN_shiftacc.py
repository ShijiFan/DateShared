"""Fig. (shiftacc) - the date-shared prior estimates the crop shifts themselves more accurately, most for crops with few parcels.
Data: ../../s3f_shift_{B88,A37,A88}.csv.gz (s3f_core.py main; EXPLORE_S3D_Q1_ZH.md S3f): for every transfer (todo[0::2]), fold and crop with
>= 20 source-train and >= 20 target-test parcels, the true shift (target-test class mean - source-train class mean, per feature) and the
estimates of prior-free mean-shift EM and of the proposed MAP-EM. Backscatter features only (VV, VH; coherence excluded).
(a)-(b) estimated against true shift, all three Dutch groups pooled (hexbin, log counts) with RMSE; (c)-(d) RMSE per crop at site B and
site A (A37 + A88), crops ordered by median number of target-test parcels per fold (labels)."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, W2, INK, INK2, GRID, COL
setup()
CL = ['Perm. gr.', 'Temp. gr.', 'Maize', 'Potato', 'Beet', 'Wint. cer.', 'Spr. cer.']
S = pd.concat([pd.read_csv(os.path.join(UP, f's3f_shift_{g}.csv.gz')) for g in ('B88', 'A37', 'A88')]); S = S[S.ty < 2]
S['site'] = np.where(S.group == 'B88', 'B', 'A')
rmse = lambda e, t: float(np.sqrt(np.mean((e - t) ** 2)))
fig = plt.figure(figsize=(W2, 2.15)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.25, 1.25], wspace=0.42, left=0.07, right=0.99, top=0.9, bottom=0.27)
lim = np.percentile(np.abs(S.true), 99.5); rows = []
for j, (k, lab) in enumerate((('em', 'Prior-free EM'), ('lw', 'Proposed'))):
    ax = fig.add_subplot(gs[0, j]); hb = ax.hexbin(S.true, S[k], gridsize=45, extent=(-lim, lim, -lim, lim), bins='log', cmap='Greys', mincnt=1, lw=0)
    ax.plot([-lim, lim], [-lim, lim], color=COL['lw'] if k == 'lw' else INK2, lw=0.8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    r = rmse(S[k], S.true); ax.text(0.04, 0.96, f'{lab}\nRMSE {r:.2f} dB', transform=ax.transAxes, va='top', fontsize=6.5)
    ax.set_xlabel('True shift (dB)'); ax.set_ylabel('Estimated shift (dB)' if j == 0 else ''); tag(ax, 'ab'[j]); rows.append(dict(site='all', cls='all', method=k, rmse=r, n=len(S)))
for j, site in enumerate(('B', 'A')):
    ax = fig.add_subplot(gs[0, 2 + j]); d = S[S.site == site]
    nt = d.groupby(['group', 'src', 'tgt', 'fold', 'cls']).n_tgt.first().groupby('cls').median().sort_values()
    x = np.arange(len(nt))
    for o, (k, lab, c) in enumerate((('em', 'Prior-free EM', INK2), ('lw', 'Proposed', COL['lw']))):
        v = [rmse(d[d.cls == c_][k], d[d.cls == c_].true) for c_ in nt.index]
        ax.bar(x + (o - 0.5) * 0.38, v, 0.36, color=c, label=lab)
        rows += [dict(site=site, cls=int(c_), method=k, rmse=vv, n=int((d.cls == c_).sum()), median_n_tgt=float(nt[c_])) for c_, vv in zip(nt.index, v)]
    ax.set_xticks(x); ax.set_xticklabels([f'{CL[c_]} ({nt[c_]:.0f})' for c_ in nt.index], rotation=45, ha='right', fontsize=6, rotation_mode='anchor')
    ax.set_ylabel('RMSE (dB)' if j == 0 else ''); ax.grid(axis='y', color=GRID, lw=0.4); ax.set_title(f'Site {site}', fontsize=7.5, pad=2); tag(ax, 'cd'[j])
    if j == 1: ax.legend(frameon=False, fontsize=6, loc='upper right')
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'figN_shiftacc.{e}'), dpi=600 if e == 'png' else None)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, 'figN_shiftacc_data.csv'), index=False); print(R.round(3).to_string(index=False))
