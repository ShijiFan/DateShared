"""Reference crop parcels (map + composition; no model output). (a)-(c) 2019 parcel declarations (BRP / RPG) mapped to the seven
classes on the 40-m analysis grid; other land uses and parcels with fewer than two pure pixels in light grey. (d)-(f) Number of eligible
parcels per class in every season used (feature files = exactly the parcels entering the experiments), log scale.
Run with D:\\anaconda\\envs\\sar."""
import os, sys, json
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from style import setup, tag, W2, INK, INK2, GRID
setup()
CLS = ['PermGrass', 'TempGrass', 'Maize', 'Potato', 'Beet', 'WinterCereal', 'SpringCereal']
CL = ['Perm. grass', 'Temp. grass', 'Maize', 'Potato', 'Beet', 'Winter cereal', 'Spring cereal']
CC = ['#1baf7a', '#7fd3b4', '#eda100', '#eb6834', '#e87ba4', '#2a78d6', '#6250d6']
REG = {'A': ('Site A', r'D:\sar_data\multiyear\A', r'D:\sar_data\multiyear\features_A.npz', 'T88'),
       'B': ('Site B', r'D:\sar_data\multiyear\B', r'D:\sar_data\multiyear\features_B.npz', 'T88'),
       'FR': ('Pays de Caux', r'D:\sar_data\fr_a\FR', r'D:\sar_data\fr_a\features_FR.npz', 'T59')}


def classmap(root, yr):
    with rasterio.open(os.path.join(root, 'labels', f'parcel_raster_{yr}.tif')) as s: a = s.read(1); T = s.transform
    t = pd.read_csv(os.path.join(root, 'labels', f'parcel_table_{yr}.csv'))
    lut = np.full(int(a.max()) + 1, 7, np.int16); lut[0] = 8                       # 7 other/ineligible parcel, 8 no parcel
    e = t[(t.eligible_ge2 == 1) & t['class'].isin(CLS)]; lut[e.source_row.to_numpy()] = e['class'].map(CLS.index).to_numpy()
    return lut[a], T


fig = plt.figure(figsize=(W2, 4.3)); gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], hspace=0.28, wspace=0.12, left=0.06, right=0.995, top=0.95, bottom=0.12)
cmap = ListedColormap(CC + ['#d9d7cf', '#ffffff'])
for j, (k, (name, root, fz, trk)) in enumerate(REG.items()):
    ax = fig.add_subplot(gs[0, j]); m, T = classmap(root, 2019); H, W = m.shape
    ext = (T.c / 1e3, (T.c + W * T.a) / 1e3, (T.f + H * T.e) / 1e3, T.f / 1e3)
    ax.imshow(m, cmap=cmap, vmin=-0.5, vmax=8.5, extent=ext, interpolation='nearest')
    xs = ext[0] + 0.06 * (ext[1] - ext[0]); ys = ext[2] + 0.05 * (ext[3] - ext[2])
    ax.add_patch(Rectangle((xs - 0.8, ys - 1.0), 11.6, 4.4, fc='white', ec='none', alpha=0.85))
    ax.plot([xs, xs + 10], [ys, ys], color=INK, lw=1.6, solid_capstyle='butt'); ax.text(xs + 5, ys + 0.7, '10 km', color=INK, ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(True) for s in ('top', 'right')]
    ax.set_title(f'{name}, 2019 declarations', fontsize=7, pad=2); tag(ax, 'abc'[j])
    Z = np.load(fz, allow_pickle=True); yrs = sorted(int(x.split('_')[0]) for x in Z.files if x.endswith(f'_{trk}_y'))
    yrs = [y for y in yrs if y != 2023 or k == 'FR']
    N = np.array([np.bincount(Z[f'{y}_{trk}_y'], minlength=7) for y in yrs])
    bx = fig.add_subplot(gs[1, j]); x = np.arange(len(yrs)); w = 0.11
    for c in range(7): bx.bar(x + (c - 3) * w, N[:, c], w, color=CC[c], lw=0)
    bx.set_yscale('log'); bx.set_ylim(20, 2e4); bx.set_xticks(x); bx.set_xticklabels([str(y)[2:] for y in yrs]); bx.grid(axis='y', color=GRID, lw=0.5)
    bx.set_xlabel('Season (20xx)'); bx.set_ylabel('Eligible parcels' if j == 0 else ''); tag(bx, 'def'[j])
h = [Patch(color=c, label=l) for c, l in zip(CC, CL)] + [Patch(color='#d9d7cf', label='Other / ineligible')]
fig.legend(handles=h, loc='lower center', ncol=8, frameon=False, fontsize=7, bbox_to_anchor=(0.53, -0.005), handlelength=1.0, columnspacing=0.9)
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'fig_parcels.{e}'))
print('ok')
