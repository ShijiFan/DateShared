"""Parcel feature dumps for the multi-year evaluation, same definitions as A1 (Year.parcel_set('primary') + features()):
parcel value = mean of LINEAR values over its pure 40 m pixels; VV/VH converted to dB; coherence linear; class = class_strict in
TARGET_CLASSES order; parcels need finite values for every used layer. Slots: plan_pairs_B.csv (2017-2023); output layout matches
SAR/shift_20261017/dump_{A,B}.npz so the frozen evaluation code can be reused. 2024/2025 are copied from those dumps.
Usage: python build_features_my.py <site>  -> D:\\sar_data\\multiyear\\features_<site>.npz (+ _meta.json)"""
import sys, os, csv, json
import numpy as np, rasterio
SITE = sys.argv[1]; ROOT = r'D:\sar_data\multiyear'; HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = ["PermGrass", "TempGrass", "Maize", "Potato", "Beet", "WinterCereal", "SpringCereal"]
TRACKS = (37, 88) if SITE == 'A' else (88,)
_ip = os.path.join(ROOT, SITE, 'product_index.jsonl')
idx = [json.loads(l) for l in open(_ip, encoding='utf-8')] if os.path.exists(_ip) else []
have = {(r['year'], r['track'], r['key']) for r in idx}
plan = [r for r in csv.DictReader(open(os.path.join(HERE, 'plan_pairs_B.csv'), encoding='utf-8')) if r['site'] == SITE or int(r['track']) == 88]
out, meta = {}, {}
def pmeans(path, lab, n):
    with rasterio.open(path) as s: a = s.read(1).astype('float64')
    ok = np.isfinite(a) & (lab > 0)
    sm = np.bincount(lab[ok], weights=a[ok], minlength=n); c = np.bincount(lab[ok], minlength=n)
    with np.errstate(invalid='ignore', divide='ignore'): return sm / c
for yr in range(2017, 2024):
    L = os.path.join(ROOT, SITE, 'labels')
    if not os.path.exists(os.path.join(L, f'spatial_splits_{yr}.csv')): print('no labels', yr); continue
    with rasterio.open(os.path.join(L, f'parcel_raster_{yr}.tif')) as s: lab = s.read(1).astype('int64')
    n = int(lab.max()) + 1
    sp = [r for r in csv.DictReader(open(os.path.join(L, f'spatial_splits_{yr}.csv'), encoding='utf-8')) if r['eligible_ge2'] == '1' and r['class_strict'] in TARGET]
    rows = np.array([int(r['source_row']) for r in sp])
    for trk in TRACKS:
        P = sorted([r for r in plan if int(r['year']) == yr and int(r['track']) == trk and r['site'] == SITE], key=lambda r: int(r['slot']))
        P = list({r['slot']: r for r in P}.values())
        if not P: continue
        if not all((yr, trk, f"{r['date1']}_{r['date2']}") in have and (yr, trk, r['date1']) in have and (yr, trk, r['date2']) in have for r in P):
            print('incomplete products', yr, trk); continue
        slots = [int(r['slot']) for r in P]; dates = sorted({d for r in P for d in (r['date1'], r['date2'])}); keys = [f"{r['date1']}_{r['date2']}" for r in P]
        D = os.path.join(ROOT, SITE, 'aligned', str(yr), f'T{trk}')
        vv = np.stack([pmeans(os.path.join(D, f'VV_{d}.tif'), lab, n)[rows] for d in dates], 1)
        vh = np.stack([pmeans(os.path.join(D, f'VH_{d}.tif'), lab, n)[rows] for d in dates], 1)
        co = np.stack([pmeans(os.path.join(D, f'COH_{k}.tif'), lab, n)[rows] for k in keys], 1)
        ok = np.isfinite(vv).all(1) & np.isfinite(vh).all(1) & np.isfinite(co).all(1) & (vv > 0).all(1) & (vh > 0).all(1)
        tag = f'{yr}_T{trk}'
        out[f'{tag}_VV'] = (10 * np.log10(vv[ok])).astype(np.float32); out[f'{tag}_VH'] = (10 * np.log10(vh[ok])).astype(np.float32)
        out[f'{tag}_COH'] = co[ok].astype(np.float32)
        out[f'{tag}_y'] = np.array([TARGET.index(sp[i]['class_strict']) for i in np.flatnonzero(ok)])
        out[f'{tag}_folds'] = np.array([[sp[i][f'fold_{k}_role'] for k in range(6)] for i in np.flatnonzero(ok)])
        out[f'{tag}_row'] = rows[ok]
        meta[tag] = dict(slots=slots, dates=dates, keys=keys, n=int(ok.sum()))
        print(SITE, tag, 'parcels', int(ok.sum()), 'slots', slots, flush=True)
# 2024/2025 from the development dumps (same definitions), slots 0..8
dev = np.load(os.path.join(r'E:\research\SAR\shift_20261017', f'dump_{SITE}.npz'), allow_pickle=True)
dm = json.load(open(os.path.join(r'E:\research\SAR\shift_20261017', f'dump_{SITE}_meta.json')))['meta']
for yr in (2024, 2025):
    for trk in TRACKS:
        k = f'{yr}_T{trk}_M9'
        if f'{k}_VV' not in dev.files: continue
        tag = f'{yr}_T{trk}'
        for lay in ('VV', 'VH', 'COH'): out[f'{tag}_{lay}'] = dev[f'{k}_{lay}']
        out[f'{tag}_y'] = dev[f'{yr}_y']; out[f'{tag}_folds'] = dev[f'{yr}_folds']; out[f'{tag}_row'] = dev[f'{yr}_row']
        meta[tag] = dict(slots=list(range(9)), dates=dm[k]['dates'], keys=dm[k]['keys'], n=int(len(dev[f'{yr}_y'])))
np.savez_compressed(os.path.join(ROOT, f'features_{SITE}.tmp.npz'), **out)
json.dump(meta, open(os.path.join(ROOT, f'features_{SITE}_meta.tmp.json'), 'w'), indent=1)
os.replace(os.path.join(ROOT, f'features_{SITE}.tmp.npz'), os.path.join(ROOT, f'features_{SITE}.npz'))
os.replace(os.path.join(ROOT, f'features_{SITE}_meta.tmp.json'), os.path.join(ROOT, f'features_{SITE}_meta.json')); print('saved', len(meta), 'year-tracks')
