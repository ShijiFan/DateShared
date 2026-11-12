"""FR-A parcel features, same definitions as SAR/multiyear_20261017/build_features_my.py: parcel value = mean of LINEAR values over its pure
40 m pixels; VV/VH -> dB; coherence linear; class = class_strict in TARGET order (Other excluded); parcels need finite values for every layer.
Slots = FR plan_pairs (T88 slot calendar). Layout identical to features_{A,B}.npz so the frozen code can be reused (tag '<year>_T59').
Usage: python fr_features.py <years comma> -> D:\\sar_data\\fr_a\\features_FR.npz + features_FR_meta.json (merges with existing years)"""
import sys, os, csv, json
import numpy as np, rasterio
YEARS = [int(y) for y in sys.argv[1].split(',')]; ROOT = r'D:\sar_data\fr_a'; HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(os.path.dirname(HERE), 'FR-A_PaysDeCaux', 'plan_pairs.csv')
TARGET = ["PermGrass", "TempGrass", "Maize", "Potato", "Beet", "WinterCereal", "SpringCereal"]
idx = [json.loads(l) for l in open(os.path.join(ROOT, 'FR', 'product_index.jsonl'), encoding='utf-8')]
have = {(r['year'], r['key']) for r in idx}
plan = [r for r in csv.DictReader(open(PLAN, encoding='utf-8')) if r['scene1']]
fp, mp = os.path.join(ROOT, 'features_FR.npz'), os.path.join(ROOT, 'features_FR_meta.json')
out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}; meta = json.load(open(mp)) if os.path.exists(mp) else {}
def pmeans(path, lab, n):
    with rasterio.open(path) as s: a = s.read(1).astype('float64')
    ok = np.isfinite(a) & (lab > 0); sm = np.bincount(lab[ok], weights=a[ok], minlength=n); c = np.bincount(lab[ok], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'): return sm / c
for yr in YEARS:
    L = os.path.join(ROOT, 'FR', 'labels')
    with rasterio.open(os.path.join(L, f'parcel_raster_{yr}.tif')) as s: lab = s.read(1).astype('int64')
    n = int(lab.max()) + 1
    sp = [r for r in csv.DictReader(open(os.path.join(L, f'spatial_splits_{yr}.csv'), encoding='utf-8')) if r['eligible_ge2'] == '1' and r['class_strict'] in TARGET]
    rows = np.array([int(r['source_row']) for r in sp])
    P = sorted([r for r in plan if int(r['year']) == yr], key=lambda r: int(r['slot']))
    P = [r for r in P if (yr, r['date1']) in have and (yr, r['date2']) in have and (yr, f"{r['date1']}_{r['date2']}") in have]
    if not P: print('no products', yr); continue
    orbit = int(P[0]['orbit']); slots = [int(r['slot']) for r in P]; dates = sorted({d for r in P for d in (r['date1'], r['date2'])}); keys = [f"{r['date1']}_{r['date2']}" for r in P]
    D = os.path.join(ROOT, 'FR', 'aligned', str(yr), f'T{orbit}')
    vv = np.stack([pmeans(os.path.join(D, f'VV_{d}.tif'), lab, n)[rows] for d in dates], 1)
    vh = np.stack([pmeans(os.path.join(D, f'VH_{d}.tif'), lab, n)[rows] for d in dates], 1)
    co = np.stack([pmeans(os.path.join(D, f'COH_{k}.tif'), lab, n)[rows] for k in keys], 1)
    ok = np.isfinite(vv).all(1) & np.isfinite(vh).all(1) & np.isfinite(co).all(1) & (vv > 0).all(1) & (vh > 0).all(1)
    tag = f'{yr}_T{orbit}'
    out[f'{tag}_VV'] = (10 * np.log10(vv[ok])).astype(np.float32); out[f'{tag}_VH'] = (10 * np.log10(vh[ok])).astype(np.float32); out[f'{tag}_COH'] = co[ok].astype(np.float32)
    out[f'{tag}_y'] = np.array([TARGET.index(sp[i]['class_strict']) for i in np.flatnonzero(ok)])
    out[f'{tag}_folds'] = np.array([[sp[i][f'fold_{k}_role'] for k in range(6)] for i in np.flatnonzero(ok)]); out[f'{tag}_row'] = rows[ok]
    meta[tag] = dict(slots=slots, dates=dates, keys=keys, n=int(ok.sum()))
    print(tag, 'parcels', int(ok.sum()), 'of', len(rows), 'slots', slots, 'class counts', np.bincount(out[f'{tag}_y'], minlength=7).tolist(), flush=True)
np.savez_compressed(fp[:-4] + '.tmp.npz', **out); json.dump(meta, open(mp + '.tmp', 'w'), indent=1)
os.replace(fp[:-4] + '.tmp.npz', fp); os.replace(mp + '.tmp', mp); print('saved', sorted(meta))
