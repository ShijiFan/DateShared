"""Fig. 1 - the phenomenon, with real data (site B, orbit 88). Season pair chosen by a fixed rule independent of any result: among the
development seasons (2023 excluded), the ordered pair with the largest difference in mean March-June antecedent precipitation index
(KNMI 280). (a) Class-mean VV trajectories in both seasons; (b) per-date VV shift of every crop (target - source) and, below, the dAPI
at each acquisition (no dual axis); thick line = date-wise mean over crops (the shared part), with its share of the shift energy
1 - sum (s - mean_d)^2 / sum s^2 (2026-11-12, reviewer-read: make the shared part visible); (c) |VV shift| against the source-season VH
level of the crop at that date (canopy attenuation), with Spearman rho."""
import os, sys, json, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE); sys.path.insert(0, UP)
from style import setup, tag, W2, INK2, GRID
from p9_lib import api_at, season_api
setup()
Z = np.load(r'D:\sar_data\multiyear\features_B.npz', allow_pickle=True); M = json.load(open(r'D:\sar_data\multiyear\features_B_meta.json'))
DEV = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]; SA = {y: season_api('B', y) for y in DEV}
ya, yb = max(itertools.permutations(DEV, 2), key=lambda p: SA[p[1]] - SA[p[0]])
CL = ['Perm. grass', 'Temp. grass', 'Maize', 'Potato', 'Beet', 'Winter cereal', 'Spring cereal']
CC = ['#1baf7a', '#7fd3b4', '#eda100', '#eb6834', '#e87ba4', '#2a78d6', '#6250d6']
S = sorted(set(M[f'{ya}_T88']['slots']) & set(M[f'{yb}_T88']['slots']))
def vv(y):
    m = M[f'{y}_T88']; pos = [m['slots'].index(s) for s in S]; dc = [c for p in pos for c in (2 * p, 2 * p + 1)]
    dt = pd.to_datetime(m['dates']); return Z[f'{y}_T88_VV'][:, dc].astype(float), Z[f'{y}_T88_VH'][:, dc].astype(float), Z[f'{y}_T88_y'], [dt[c] for c in dc]
VA, HA, YA, DA = vv(ya); VB, HB, YB, DB = vv(yb)
doy = np.array([d.dayofyear for d in DA]); dapi = api_at('B', DB) - api_at('B', DA)
sh = {c: VB[YB == c].mean(0) - VA[YA == c].mean(0) for c in range(7) if (YA == c).sum() >= 20 and (YB == c).sum() >= 20}
fig = plt.figure(figsize=(W2, 2.7)); gs = GridSpec(4, 3, figure=fig, width_ratios=[1.1, 1.1, 0.9], height_ratios=[1, 1, 1, 0.7], hspace=0.15, wspace=0.42)
ax = fig.add_subplot(gs[:3, 0])
for c, dy in ((0, -0.3), (2, 0.5), (5, -0.5)):
    ax.plot(doy, VA[YA == c].mean(0), color=CC[c], lw=1.2); ax.plot(doy, VB[YB == c].mean(0), color=CC[c], lw=1.2, ls='--')
    ax.text(doy[-1] + 4, VB[YB == c].mean(0)[-1] + dy, CL[c], color=CC[c], fontsize=6.5, va='center')
ax.plot([], [], color=INK2, lw=1, label=f'{ya} (source)'); ax.plot([], [], color=INK2, lw=1, ls='--', label=f'{yb} (target)')
ax.legend(frameon=False, loc='lower left'); ax.set_xlabel('Day of year'); ax.set_ylabel(r'VV $\sigma^0$ (dB)'); ax.set_xlim(doy[0] - 5, doy[-1] + 50); tag(ax, 'a')
ax = fig.add_subplot(gs[:3, 1]); axb = fig.add_subplot(gs[3, 1], sharex=ax)
for c, s in sh.items(): ax.plot(doy, s, color=CC[c], lw=0.9, label=CL[c])
SH = np.stack(list(sh.values())); md = SH.mean(0); share = 1 - ((SH - md) ** 2).sum() / (SH ** 2).sum()
ax.plot(doy, md, color='#222222', lw=2.0, zorder=5); ax.text(0.03, 0.04, f'black: mean over crops\n({100 * share:.0f}% of shift energy)', transform=ax.transAxes, fontsize=6.3, color='#222222', va='bottom')
ax.axhline(0, color=GRID, lw=0.6); ax.set_ylabel(r'VV shift (dB)'); plt.setp(ax.get_xticklabels(), visible=False); tag(ax, 'b')
axb.bar(doy, dapi, width=6, color='#86b6ef'); axb.axhline(0, color=GRID, lw=0.6); axb.set_ylabel(r'$\Delta$API' '\n(mm)'); axb.set_xlabel('Day of year')
ax = fig.add_subplot(gs[:, 2])
for c, s in sh.items(): ax.scatter(HA[YA == c].mean(0), np.abs(s), s=6, color=CC[c], lw=0)
from scipy.stats import spearmanr
rho = spearmanr(np.concatenate([HA[YA == c].mean(0) for c in sh]), np.concatenate([np.abs(s) for s in sh.values()]))[0]
ax.text(0.97, 0.96, rf'Spearman $\rho={rho:.2f}$', transform=ax.transAxes, ha='right', va='top', fontsize=6.5)
ax.set_xlabel(r'Source VH $\sigma^0$ (dB)'); ax.set_ylabel(r'$|$VV shift$|$ (dB)'); tag(ax, 'c')
h = [plt.Line2D([], [], color=CC[c], lw=1.2) for c in sh]; fig.legend(h, [CL[c] for c in sh], loc='lower center', ncol=len(sh), frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.07, right=0.99, top=0.95, bottom=0.25)
fig.savefig(os.path.join(HERE, 'fig1_shift.pdf')); fig.savefig(os.path.join(HERE, 'fig1_shift.png'), dpi=300)
print('pair', ya, yb, 'shared share', round(float(share), 3), 'spearman', round(float(rho), 3), 'r(|shift|,VH)', round(float(np.corrcoef(np.concatenate([HA[YA == c].mean(0) for c in sh]), np.concatenate([np.abs(s) for s in sh.values()]))[0, 1]), 3))
