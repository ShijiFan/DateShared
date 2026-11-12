"""Classification maps of one transfer per region (map figure). Usage: python fig_classmap.py main|supp.
main: representative transfer = gain of the proposed TempCNN over no adaptation closest to the regional median (S3g amendment);
supp: the originally registered rule, the season pair with the largest increase in mean March-June antecedent precipitation.
Window rule (no result used): the 5 x 5 km window, on a 0.5-km step, containing the most evaluated target parcels.
Columns: (1) reference classes; (2)-(4) error maps of TempCNN without adaptation, DANN and the proposed TempCNN (correct parcels grey,
misclassified parcels red), with the share of correctly classified parcels in the window and the balanced accuracy over the whole
transfer; (5) per-class recall over all evaluated target parcels of the transfer. Run with D:\\anaconda\\envs\\sar after s3g_mappred.py."""
import os, sys, json
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, HERE)
from style import setup, tag, W2, INK, INK2, GRID, COL
setup()
CC = ['#1baf7a', '#7fd3b4', '#eda100', '#eb6834', '#e87ba4', '#2a78d6', '#6250d6']
CL = ['Perm. grass', 'Temp. grass', 'Maize', 'Potato', 'Beet', 'Winter cereal', 'Spring cereal']
AB = ['PG', 'TG', 'MZ', 'PO', 'BE', 'WC', 'SC']
ROOT = {'B88': r'D:\sar_data\multiyear\B', 'A88': r'D:\sar_data\multiyear\A', 'FR': r'D:\sar_data\fr_a\FR'}
NAME = {'B88': 'Site B', 'A88': 'Site A', 'FR': 'Pays de Caux'}
MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'; PICK = {}
for l in open(os.path.join(UP, 's3g_pick.json')):
    r = json.loads(l)
    if r.get('rule', 'api') == ('median_gain' if MODE == 'main' else 'api'): PICK[r['group']] = r
METH = [('none', 'No adaptation', '#9a9893'), ('dann', 'DANN', COL['dann']), ('prop_tcn', 'Proposed', COL['lw'])]
cls_cmap = ListedColormap(CC + ['#e4e2da', '#ffffff'])                       # 7 parcel not evaluated, 8 no parcel
err_cmap = ListedColormap(['#c9c6bc', '#d1352b', '#eeede8', '#ffffff'])         # 0 correct, 1 wrong, 2 not evaluated, 3 no parcel
WIN, STEP = 5000, 500


def window(lab_ok, T):
    H, W = lab_ok.shape; px = int(WIN / T.a); st = int(STEP / T.a); best = (-1, 0, 0)
    for r0 in range(0, H - px + 1, st):
        for c0 in range(0, W - px + 1, st):
            n = len(np.unique(lab_ok[r0:r0 + px, c0:c0 + px])) - 1
            if n > best[0]: best = (n, r0, c0)
    return best


def recall(y, p): return np.array([np.mean(p[y == c] == c) if (y == c).any() else np.nan for c in range(7)])


groups = [g for g in ('B88', 'A88', 'FR') if g in PICK]
fig = plt.figure(figsize=(W2, 1.62 * len(groups) + 0.42))
gs = fig.add_gridspec(len(groups), 5, width_ratios=[1, 1, 1, 1, 1.25], wspace=0.06, hspace=0.36, left=0.035, right=0.945, top=0.94, bottom=0.115)
meta = {}
for i, g in enumerate(groups):
    pk = PICK[g]; yb = pk['tgt']; sfx = f"_{pk['src']}_{yb}" if MODE == 'main' else ''
    P = pd.read_csv(os.path.join(UP, f's3g_pred_{g}{sfx}.csv')); P = P[P.fold >= 0]
    with rasterio.open(os.path.join(ROOT[g], 'labels', f'parcel_raster_{yb}.tif')) as s: a = s.read(1); T = s.transform
    ok = np.zeros(int(a.max()) + 1, bool); ok[P.row.to_numpy()] = True; lab_ok = np.where(ok[a], a, 0)
    n, r0, c0 = window(lab_ok, T); px = int(WIN / T.a); sub = a[r0:r0 + px, c0:c0 + px]
    inwin = np.isin(P.row.to_numpy(), np.unique(sub)); meta[g] = dict(src=pk['src'], tgt=yb, window_rc=[int(r0), int(c0)], n_parcels=int(n), ba=pk['ba'])
    ax = fig.add_subplot(gs[i, 0]); lut = np.full(len(ok), 7, np.int16); lut[0] = 8; lut[P.row.to_numpy()] = P.y.to_numpy()
    ax.imshow(lut[sub], cmap=cls_cmap, vmin=-0.5, vmax=8.5, interpolation='nearest')
    ax.set_ylabel(f"{NAME[g]}\n{pk['src']}$\\rightarrow${yb}", fontsize=7); ax.set_title('Reference' if i == 0 else '', fontsize=7, pad=2)
    ax.add_patch(Rectangle((px * 0.05, px * 0.90), px * 0.40, px * 0.035, fc=INK, ec='none'))
    ax.text(px * 0.25, px * 0.87, '2 km', ha='center', va='bottom', fontsize=6, color=INK, bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(True) for s in ('top', 'right')]; tag(ax, 'abcdefghijklmno'[i * 5])
    y = P.y.to_numpy(); meta[g]['window_acc'] = {}
    for j, (k, title, _) in enumerate(METH):
        ax = fig.add_subplot(gs[i, j + 1]); p = P[k].to_numpy(); lut = np.full(len(ok), 2, np.int16); lut[0] = 3
        lut[P.row.to_numpy()] = (p != y).astype(np.int16); ax.imshow(lut[sub], cmap=err_cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')
        acc = float(np.mean(p[inwin] == y[inwin])); meta[g]['window_acc'][k] = acc
        ax.text(0.03, 0.03, f"window correct {100 * acc:.0f}%\nBA {pk['ba'][k]:.3f}", transform=ax.transAxes, fontsize=6, va='bottom', ha='left',
                color=INK, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.2), linespacing=1.1)
        ax.set_title(title if i == 0 else '', fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([]); [ax.spines[s].set_visible(True) for s in ('top', 'right')]; tag(ax, 'abcdefghijklmno'[i * 5 + j + 1])
    bx = fig.add_subplot(gs[i, 4]); x = np.arange(7); w = 0.27
    for j, (k, title, c) in enumerate(METH):
        bx.bar(x + (j - 1) * w, recall(y, P[k].to_numpy()), w, color=c, lw=0, label=title)
    bx.set_xticks(x); bx.set_xticklabels(AB, fontsize=6); bx.set_ylim(0, 1); bx.set_yticks([0, 0.5, 1]); bx.grid(axis='y', color=GRID, lw=0.5)
    bx.set_ylabel('Recall', fontsize=7, labelpad=1); bx.yaxis.set_label_position('right'); bx.yaxis.tick_right()
    bx.spines['right'].set_visible(True); bx.spines['left'].set_visible(False)
    for t, c in zip(bx.get_xticklabels(), CC): t.set_color(c); t.set_fontweight('bold')
    bx.set_title('Per-class recall, all parcels' if i == 0 else '', fontsize=7, pad=2); tag(bx, 'abcdefghijklmno'[i * 5 + 4])
h = [Patch(color=c, label=f'{a}: {l}') for c, a, l in zip(CC, AB, CL)]
h2 = [Patch(color='#c9c6bc', label='Correct'), Patch(color='#d1352b', label='Misclassified'), Patch(fc='#eeede8', ec='#bbbbbb', lw=0.4, label='Not evaluated')] + \
     [Patch(color=c, label=f'{t} (recall)') for _, t, c in METH]
fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.035), handlelength=1.0, columnspacing=0.8)
fig.legend(handles=h2, loc='lower center', ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.005), handlelength=1.0, columnspacing=0.8)
name = 'fig_classmap' if MODE == 'main' else 'figS_classmap_api'
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'{name}.{e}'))
json.dump(meta, open(os.path.join(HERE, f'{name}_manifest.json'), 'w'), indent=1); print(json.dumps(meta))
