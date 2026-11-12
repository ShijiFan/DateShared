"""Fig. (traj) - what the inter-annual shift looks like in real data. Site B, relative orbit 88, all eight development seasons
(2017-2022, 2024, 2025; 2023 not used). (a) Class-mean VV and (b) VH trajectories of the seven crops over day of year, one thin line per
season, coloured by the season's mean March-June antecedent precipitation index (API, KNMI station 280; p9_lib.season_api);
(c) daily API of every season from 1 March to 31 October with the Sentinel-1 acquisition dates as ticks.
Sources: D:/sar_data/multiyear/features_B.npz + features_B_meta.json (parcel means, dB), ../multiyear_20261017/knmi_daily_my.csv."""
import os, sys, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.gridspec import GridSpec
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE); sys.path.insert(0, UP)
from style import setup, tag, W2, INK2, GRID
from p9_lib import api_at, season_api
setup()
Z = np.load(r'D:\sar_data\multiyear\features_B.npz', allow_pickle=True); M = json.load(open(r'D:\sar_data\multiyear\features_B_meta.json'))
DEV = [2017, 2018, 2019, 2020, 2021, 2022, 2024, 2025]; SA = {y: season_api('B', y) for y in DEV}
CL = ['Perm. grass', 'Temp. grass', 'Maize', 'Potato', 'Beet', 'Winter cereal', 'Spring cereal']
norm = mpl.colors.Normalize(min(SA.values()), max(SA.values())); cmap = mpl.colors.LinearSegmentedColormap.from_list('vt', mpl.colormaps['viridis_r'](np.linspace(0.18, 1.0, 256)))
fig = plt.figure(figsize=(W2, 4.3)); gs = GridSpec(3, 7, figure=fig, height_ratios=[1, 1, 0.75], hspace=0.42, wspace=0.12,
                                                  left=0.065, right=0.9, top=0.95, bottom=0.09)
axv = [fig.add_subplot(gs[0, j]) for j in range(7)]; axh = [fig.add_subplot(gs[1, j], sharex=axv[j]) for j in range(7)]
out = []
for y in DEV:
    m = M[f'{y}_T88']; doy = pd.to_datetime(m['dates']).dayofyear.to_numpy(); yy = Z[f'{y}_T88_y']; col = cmap(norm(SA[y]))
    for pol, axs in (('VV', axv), ('VH', axh)):
        X = Z[f'{y}_T88_{pol}'].astype(float)
        for c in range(7):
            mc = np.nanmean(X[yy == c], 0); axs[c].plot(doy, mc, color=col, lw=0.8, alpha=0.95)
            out += [dict(year=y, pol=pol, cls=c, doy=int(d), mean_db=float(v)) for d, v in zip(doy, mc)]
for j in range(7):
    axv[j].set_title(CL[j], fontsize=7, pad=2)
    for a in (axv[j], axh[j]):
        a.grid(color=GRID, lw=0.4); a.set_xlim(60, 300); a.set_xticks([90, 180, 270])
        if j: a.tick_params(labelleft=False)
    axv[j].tick_params(labelbottom=False); axh[j].set_xticklabels(['Apr', 'Jul', 'Oct'])
for row in (axv, axh):
    lo = min(a.get_ylim()[0] for a in row); hi = max(a.get_ylim()[1] for a in row)
    for a in row: a.set_ylim(lo, hi)
axv[0].set_ylabel(r'VV $\sigma^0$ (dB)'); axh[0].set_ylabel(r'VH $\sigma^0$ (dB)'); tag(axv[0], 'a'); tag(axh[0], 'b')
ax = fig.add_subplot(gs[2, :]); ax.grid(color=GRID, lw=0.4)
for i, y in enumerate(DEV):
    d = pd.date_range(f'{y}-03-01', f'{y}-10-31'); a = api_at('B', d); col = cmap(norm(SA[y]))
    ax.plot(d.dayofyear, a, color=col, lw=0.8)
    dd = pd.to_datetime(M[f'{y}_T88']['dates']).dayofyear; ax.plot(dd, np.full(len(dd), -4 - 3.2 * i), '|', color=col, ms=3.5, mew=0.8)
ax.set_xlim(60, 300); ax.set_xticks([60, 91, 121, 152, 182, 213, 244, 274]); ax.set_xticklabels(['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'])
ax.set_ylabel('API (mm)'); ax.set_ylim(-4 - 3.2 * len(DEV), None); tag(ax, 'c')
ax.text(301, -4 - 1.6 * len(DEV), 'S1 dates', fontsize=6, color=INK2, va='center', ha='left')
cax = fig.add_axes([0.915, 0.36, 0.012, 0.56]); cb = fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), cax=cax)
cb.set_label('Mar--Jun mean API (mm)', fontsize=7); cb.ax.tick_params(labelsize=6)
for y in DEV: cb.ax.plot([0, 1], [SA[y]] * 2, color='w', lw=0.5)
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'figN_traj.{e}'), dpi=600 if e == 'png' else None)
pd.DataFrame(out).to_csv(os.path.join(HERE, 'figN_traj_data.csv'), index=False)
print({y: round(v, 1) for y, v in SA.items()})
