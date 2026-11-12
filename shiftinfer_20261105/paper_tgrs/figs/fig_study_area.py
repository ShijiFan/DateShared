"""Study areas (map figure; no model output). (a) Location of the two Dutch sites and the French confirmation region (Natural Earth 1:10m
countries). (b)-(d) Sentinel-1 false-colour composites of one mid-season acquisition (R: VV, G: VH, B: VV-VH ratio, dB; RTC gamma0,
40-m analysis grid) with the 5-km blocks coloured by spatial fold. Date rule (fixed, no result used): the 2019 acquisition of the
mapping orbit closest to 1 July. Block = floor(UTM x / 5 km), floor(UTM y / 5 km); fold = assigned_test_fold of the 2019 eligible
parcels (NL) or FR_BLOCK_FOLDS.json (France). Run with D:\\anaconda\\envs\\sar (geopandas, rasterio)."""
import os, sys, glob, json, zipfile
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, rasterio, geopandas as gpd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from style import setup, tag, W2, INK, INK2
setup()
NE = r'E:\research\SAR\merged_tgrs_20260930\figures_src\ne\ne_10m_admin_0_countries.zip'
REG = {'A': dict(name='Site A (NL)', root=r'D:\sar_data\multiyear\A', trk='T88', aoi=(6.00, 51.98, 6.54, 52.38)),
       'B': dict(name='Site B (NL)', root=r'D:\sar_data\multiyear\B', trk='T88', aoi=(6.55, 52.75, 7.05, 53.10)),
       'FR': dict(name='Pays de Caux (FR)', root=r'D:\sar_data\fr_a\FR', trk='T59', aoi=(0.55, 49.55, 1.10, 49.90))}
FOLDC = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']


def pick(root, trk, pol):
    fs = sorted(glob.glob(os.path.join(root, 'aligned', '2019', trk, f'{pol}_2019-*.tif')))
    d = [pd.Timestamp(os.path.basename(f)[len(pol) + 1:-4]) for f in fs]; i = int(np.argmin([abs((x - pd.Timestamp('2019-07-01')).days) for x in d]))
    return fs[i], d[i]


def stretch(a, lo, hi): return np.clip((a - lo) / (hi - lo), 0, 1)


def todb(a):
    a = a.astype(float); a[a <= 0] = np.nan
    return 10 * np.log10(a) if np.nanmedian(a) < 1.5 else a      # linear power -> dB (no-op if already dB)


def blocks(key, root):
    if key == 'FR': return {tuple(int(v) for v in k.split('_')): f for k, f in json.load(open(os.path.join(root, 'FR_BLOCK_FOLDS.json'))).items()}
    t = pd.read_csv(os.path.join(root, 'labels', 'spatial_splits_2019.csv')); t = t[t.eligible_ge2 == 1]
    return {tuple(int(v) for v in b.split('_')): int(g.assigned_test_fold.mode()[0]) for b, g in t.groupby('block_5km')}


fig = plt.figure(figsize=(W2, 2.45)); gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1], wspace=0.10, left=0.055, right=0.995, top=0.91, bottom=0.12)
# (a) locator
ax = fig.add_subplot(gs[0]); ne = gpd.read_file(f'zip://{NE}')
ne = ne.cx[-3:10, 47:55]; ne.plot(ax=ax, color='#f2f1ec', edgecolor='#9a9893', linewidth=0.4)
ne[ne.ADMIN.isin(['Netherlands', 'France'])].plot(ax=ax, color='#e3e1d8', edgecolor='#77756f', linewidth=0.5)
for k, r in REG.items():
    x0, y0, x1, y1 = r['aoi']; ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='#c0392b', lw=0.9))
    ax.text(x1 + 0.18, (y0 + y1) / 2, {'A': 'A', 'B': 'B', 'FR': 'Pays de Caux'}[k], fontsize=7, va='center', color=INK)
ax.set_xlim(-2.5, 9.0); ax.set_ylim(47.6, 54.3); ax.set_aspect(1 / np.cos(np.radians(51)))
ax.set_xticks([0, 4, 8]); ax.set_xticklabels(['0°', '4°E', '8°E']); ax.set_yticks([48, 50, 52, 54]); ax.set_yticklabels(['48°N', '50°N', '52°N', '54°N'])
for s in ('top', 'right'): ax.spines[s].set_visible(True)
ax.text(5.1, 53.7, 'NL', fontsize=7, color=INK2); ax.text(2.0, 48.3, 'FR', fontsize=7, color=INK2); tag(ax, 'a')
meta = {}
for j, (k, r) in enumerate(REG.items()):
    ax = fig.add_subplot(gs[j + 1]); fvv, d = pick(r['root'], r['trk'], 'VV'); fvh, _ = pick(r['root'], r['trk'], 'VH')
    with rasterio.open(fvv) as s: vv = todb(s.read(1)); T = s.transform; H, Wd = s.shape; crs = str(s.crs)
    with rasterio.open(fvh) as s: vh = todb(s.read(1))
    land = (vh > -24) & (vv > -20); q = lambda a: np.nanpercentile(a[land], [2, 98])
    rgb = np.dstack([stretch(vv, *q(vv)), stretch(vh, *q(vh)), stretch(vv - vh, *q(vv - vh))])
    rgb[~land | np.isnan(vv)] = 0.93                                   # open water / no data: light grey
    ext = (T.c / 1e3, (T.c + Wd * T.a) / 1e3, (T.f + H * T.e) / 1e3, T.f / 1e3)
    ax.imshow(rgb, extent=ext, interpolation='nearest')
    for (bx, by), f in blocks(k, r['root']).items():
        ax.add_patch(Rectangle((bx * 5, by * 5), 5, 5, fill=False, ec=FOLDC[f], lw=0.7))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    xs = ext[0] + 0.06 * (ext[1] - ext[0]); ys = ext[2] + 0.05 * (ext[3] - ext[2])
    ax.add_patch(Rectangle((xs - 0.8, ys - 1.0), 11.6, 4.4, fc='white', ec='none', alpha=0.85))
    ax.plot([xs, xs + 10], [ys, ys], color=INK, lw=1.6, solid_capstyle='butt'); ax.text(xs + 5, ys + 0.7, '10 km', color=INK, ha='center', va='bottom', fontsize=6.5)
    ax.annotate('', xy=(ext[1] - 2.5, ext[3] - 2), xytext=(ext[1] - 2.5, ext[3] - 6.5), arrowprops=dict(arrowstyle='-|>', color='white', lw=0.9))
    ax.text(ext[1] - 2.5, ext[3] - 8.6, 'N', color='white', ha='center', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(True) for s in ('top', 'right')]
    ax.set_title(f"{r['name']}, orbit {r['trk'][1:]}, {d.day} {d:%b %Y}", fontsize=7, pad=2); tag(ax, 'bcd'[j])
    meta[k] = dict(vv=fvv, vh=fvh, crs=crs, extent_km=[round(e, 2) for e in ext])
h = [Line2D([], [], color=c, lw=1.2, label=f'Fold {i + 1}') for i, c in enumerate(FOLDC)]
fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.62, -0.01), handlelength=1.4, columnspacing=1.0)
for ext_ in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'fig_study_area.{ext_}'))
json.dump(meta, open(os.path.join(HERE, 'fig_study_area_manifest.json'), 'w'), indent=1); print(json.dumps(meta, indent=1))
