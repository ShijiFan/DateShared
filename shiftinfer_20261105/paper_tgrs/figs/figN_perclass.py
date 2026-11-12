"""Fig. (perclass) - where the accuracy gain comes from: per-crop recall of the Gaussian classifier without adaptation, with prior-free
mean-shift EM and with the proposed MAP-EM. Data: ../../s3f_pred_{B88,A37,A88}.csv.gz (s3f_core.py main; target-test predictions of every
transfer todo[0::2] and fold). Statistic: per-transfer recall of each crop, median over transfers; crops ordered by their share of
parcels at the site (median share in brackets). (a) site B, (b) site A (A37 + A88). Numbers above: share of transfers in which the proposed
method's recall exceeds EM's."""
import os, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, W2, INK, INK2, GRID, COL, MK
setup()
CL = ['Perm. gr.', 'Temp. gr.', 'Maize', 'Potato', 'Beet', 'Wint. cer.', 'Spr. cer.']
P = pd.concat([pd.read_csv(os.path.join(UP, f's3f_pred_{g}.csv.gz')) for g in ('B88', 'A37', 'A88')]); P['site'] = np.where(P.group == 'B88', 'B', 'A')
rec = []
for (site, g, s, t), d in P.groupby(['site', 'group', 'src', 'tgt']):
    for c in range(7):
        m = d.y == c
        if m.sum() < 20: continue
        rec.append(dict(site=site, group=g, src=s, tgt=t, cls=c, share=m.mean(), n=int(m.sum()), **{k: float((d.loc[m, f'p_{k}'] == c).mean()) for k in ('raw', 'em', 'lw')}))
R = pd.DataFrame(rec); R.to_csv(os.path.join(HERE, 'figN_perclass_data.csv'), index=False)
fig = plt.figure(figsize=(W2, 2.3)); gs = fig.add_gridspec(1, 2, wspace=0.12, left=0.07, right=0.99, top=0.91, bottom=0.25)
M = [('raw', 'No adaptation', '#b4b2aa', 'o'), ('em', 'Prior-free EM', INK2, MK['em']), ('lw', 'Proposed', COL['lw'], MK['lw'])]
summ = []
for j, site in enumerate(('B', 'A')):
    ax = fig.add_subplot(gs[0, j]); d = R[R.site == site]; order = d.groupby('cls').share.median().sort_values(ascending=False).index; x = np.arange(len(order))
    for o, (k, lab, c, mk) in enumerate(M):
        v = d.groupby('cls')[k].median().reindex(order); ax.plot(x + (o - 1) * 0.18, v, mk, color=c, ms=3.6, label=lab, ls='none')
    for i, c_ in enumerate(order):
        dd = d[d.cls == c_]; ax.text(i, 1.035, f'{100 * (dd.lw > dd.em).mean():.0f}%', ha='center', va='center', fontsize=5.5, color=COL['lw'])
        summ.append(dict(site=site, cls=CL[c_], share=float(dd.share.median()), **{k: float(dd[k].median()) for k, *_ in M},
                         gain_lw_em=float((dd.lw - dd.em).median()), win=float((dd.lw > dd.em).mean()), n_transfers=len(dd)))
    ax.set_xticks(x); ax.set_xticklabels([f'{CL[c_]} ({100 * d[d.cls == c_].share.median():.1f}%)' for c_ in order], rotation=40, ha='right', rotation_mode='anchor', fontsize=6)
    ax.set_ylim(0, 1.07); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel('Recall (median)' if j == 0 else ''); ax.tick_params(labelleft=(j == 0)); ax.grid(axis='y', color=GRID, lw=0.4); ax.set_title(f'Site {site}', fontsize=7.5, pad=2); tag(ax, 'ab'[j])
    if j == 0: ax.legend(frameon=False, fontsize=6, loc='lower left', handletextpad=0.2)
Su = pd.DataFrame(summ); rho = spearmanr(Su.share, Su.gain_lw_em)
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'figN_perclass.{e}'), dpi=600 if e == 'png' else None)
Su.to_csv(os.path.join(HERE, 'figN_perclass_summary.csv'), index=False); print(Su.round(3).to_string(index=False)); print('rho', rho)
