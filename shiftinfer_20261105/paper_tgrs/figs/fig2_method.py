"""Fig. 2 - method structure as a thin-line engineering diagram with real-data insets (site B, source season 2025, orbit 88):
(i) source class-mean VV trajectories mu_c(t); (ii) water-cloud date-column weights a_c(t) = exp(-(VH_c(t)+25)/10) (crops x dates);
(iii) timing column d mu_c / dt for maize; boxes: archive -> (sigma_a^2, Sigma_res); prior; MAP-EM on the unlabelled target season;
outputs: Gaussian labels and resynthesised source season for any classifier."""
import os, sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from style import setup, W2, INK, INK2, COL
setup()
Z = np.load(r'D:\sar_data\multiyear\features_B.npz', allow_pickle=True); M = json.load(open(r'D:\sar_data\multiyear\features_B_meta.json'))
t = '2025_T88'; m = M[t]; doy = np.array([pd.Timestamp(d).dayofyear for d in m['dates']]); y = Z[f'{t}_y']
VV, VH = Z[f'{t}_VV'].astype(float), Z[f'{t}_VH'].astype(float)
CL = ['Perm. grass', 'Temp. grass', 'Maize', 'Potato', 'Beet', 'W. cereal', 'S. cereal']; CC = ['#1baf7a', '#7fd3b4', '#eda100', '#eb6834', '#e87ba4', '#2a78d6', '#6250d6']
mu = np.array([VV[y == c].mean(0) for c in range(7)]); vh = np.array([VH[y == c].mean(0) for c in range(7)]); att = np.exp(-(vh + 25) / 10)
fig = plt.figure(figsize=(W2, 2.55)); ax0 = fig.add_axes([0, 0, 1, 1]); ax0.set_xlim(0, 100); ax0.set_ylim(0, 36); ax0.axis('off')
def box(x, y_, w, h, title, lw=0.7, ec=INK2):
    ax0.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle='round,pad=0.2,rounding_size=0.8', fc='white', ec=ec, lw=lw))
    ax0.text(x + 0.8, y_ + h - 1.0, title, fontsize=7, va='top', color=INK)
def arrow(x0, y0, x1, y1): ax0.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.7, color=INK2, shrinkA=0, shrinkB=0))
# (1) source season
box(0.8, 19, 21, 16, 'Labelled source season'); a = fig.add_axes([0.035, 0.57, 0.18, 0.28])
for c in (0, 2, 5): a.plot(doy, mu[c], color=CC[c], lw=0.9)
a.set_xticks([]); a.set_yticks([]); a.set_ylabel(r'$\mu_c(t)$', fontsize=6); a.set_xlabel('day of year', fontsize=6)
# (2) archive
box(0.8, 1.5, 21, 15, 'Archive of past season pairs'); ax0.text(1.8, 9.0, 'labelled pairs $(y_a,y_b)$\n$\\rightarrow$ observed shifts $\\delta^{(k)}$\n$\\rightarrow$ $\\sigma_a^2$, $\\Sigma_{\\mathrm{res}}$ (Ledoit-Wolf)', fontsize=6.5, va='center', color=INK)
# (3) date-shared basis
box(25.5, 1.5, 34, 33.5, r'Date-shared shift basis $\mathbf{B}$', lw=1.0, ec=COL['lw'])
a = fig.add_axes([0.335, 0.53, 0.135, 0.32]); im = a.imshow(att, aspect='auto', cmap='Oranges', vmin=0, vmax=att.max())
a.set_yticks(range(7)); a.set_yticklabels(CL, fontsize=5); a.set_xticks([]); a.set_xlabel('acquisition date $t$', fontsize=6)
a.set_title(r'date columns $a_c(t)\,w_t$', fontsize=6.5, pad=2)
a = fig.add_axes([0.495, 0.53, 0.085, 0.32]); a.plot(doy, np.gradient(mu[2], doy), color=CC[2], lw=0.9); a.axhline(0, color='#cccccc', lw=0.5)
a.set_xticks([]); a.set_yticks([]); a.set_title(r'timing $\tau_c\,\partial\mu_c/\partial t$', fontsize=6.5, pad=2); a.set_xlabel('maize', fontsize=6)
ax0.text(27, 7.5, r'$\boldsymbol{\delta}\sim\mathcal{N}(\mathbf{m},\ \mathbf{B}\,\mathrm{diag}(\boldsymbol{\sigma}_a^2)\mathbf{B}^\top+\boldsymbol{\Sigma}_{\mathrm{res}})$', fontsize=7.5, color=INK)
ax0.text(27, 3.6, 'shared moisture offset per date, attenuated by the canopy;\nphenological delay per crop; full residual', fontsize=6, color=INK2)
# (4) MAP-EM
box(63.5, 13, 14.5, 12, 'MAP-EM on the\nunlabelled target'); ax0.text(64.5, 15.5, 'responsibilities $r_{ic}$\nlinear M-step (3)\n30 iterations', fontsize=6, color=INK2)
# (5) outputs
box(82, 23, 17, 9, 'Gaussian classifier'); ax0.text(83, 25.2, r'labels from $\mu_c+\hat{\boldsymbol{\delta}}_c$', fontsize=6.3, color=INK2)
box(82, 4, 17, 12, 'Resynthesis'); ax0.text(83, 6.0, r'$\mathbf{x}\leftarrow\mathbf{x}+\hat{\boldsymbol{\delta}}_{y}$' '\n' 'then HGB, RF,\nTempCNN or DANN', fontsize=6.3, color=INK2)
arrow(22.3, 27, 25.2, 27); arrow(22.3, 9, 25.2, 9); arrow(60.2, 19, 63.2, 19); arrow(78.3, 21, 81.7, 27.5); arrow(78.3, 17, 81.7, 10)
fig.savefig(os.path.join(HERE, 'fig2_method.pdf')); fig.savefig(os.path.join(HERE, 'fig2_method.png'), dpi=300)
