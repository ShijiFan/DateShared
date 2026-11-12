"""Align plan-B HyP3 products to the existing Site A / Site B 40 m grids (identical resampling to A1 a1_s4_build_cube.py and site2
s2_05_stream_align.py: linear gamma0 VV/VH and coherence area-averaged with Resampling.average). T88 -> sites A and B; T37 -> site A.
Zips in D:\\sar_data\\multiyear\\downloads are only READ (deletion needs user consent). Loops until download_B reports all jobs finished
and every zip is aligned. Index: D:\\sar_data\\multiyear\\<site>\\product_index.jsonl"""
import os, csv, json, time, zipfile, hashlib
import numpy as np, rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
HERE = os.path.dirname(os.path.abspath(__file__)); DL = r'D:\sar_data\multiyear\downloads'; ROOT = r'D:\sar_data\multiyear'
SAR = r'E:\research\SAR'
GRIDS = {'A': os.path.join(SAR, r'A1_observation_budget_20260923\runs\20260923_G0L_COMMON_GRID\grid_spec.json'),
         'B': os.path.join(SAR, r'site2_20260930\runs\G0L_COMMON_GRID\grid_spec.json')}
G = {}
for s, p in GRIDS.items():
    gs = json.load(open(p)); G[s] = (gs['crs'], int(gs['width']), int(gs['height']), Affine.from_gdal(*gs['affine_gdal']))
cat = {}
for r in csv.DictReader(open(os.path.join(HERE, 'plan_pairs_B.csv'), encoding='utf-8')):
    for sc, d in ((r['scene1'], r['date1']), (r['scene2'], r['date2'])):
        cat[sc.split('_')[5]] = dict(year=int(r['year']), track=int(r['track']), date=d)
def parse(stem):
    p = stem.split('_')
    if stem.startswith('S1') and '_RTC' in stem: return dict(kind='RTC', key1=p[2])
    if '_INT40_' in stem: return dict(kind='INSAR', key1=p[1], key2=p[2])
    return None
def align(src, g):
    crs, W, H, TR = g
    with rasterio.open(src) as s:
        a = s.read(1).astype('float32')
        if s.nodata is not None: a[a == s.nodata] = np.nan
        a[~np.isfinite(a)] = np.nan
        d = np.full((H, W), np.nan, dtype='float32')
        reproject(a, d, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan, dst_transform=TR, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
    return d
def save(path, a, g):
    crs, W, H, TR = g; os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path[:-4] + '.tmp.tif'
    with rasterio.open(tmp, 'w', driver='GTiff', dtype='float32', nodata=np.nan, width=W, height=H, count=1, crs=crs, transform=TR, compress='deflate') as d: d.write(a, 1)
    os.replace(tmp, path)
def done_uids(site):
    p = os.path.join(ROOT, site, 'product_index.jsonl')
    return {json.loads(l)['uid'] for l in open(p, encoding='utf-8')} if os.path.exists(p) else set()
while True:
    done = {s: done_uids(s) for s in G}; n_new = 0
    for zf in sorted(f for f in os.listdir(DL) if f.endswith('.zip')):
        info = parse(zf[:-4]); s1 = cat.get(info['key1']) if info else None
        if s1 is None: continue
        if info['kind'] == 'INSAR':
            s2 = cat.get(info['key2']); key = f"{s1['date']}_{s2['date']}"; lays = (('COH', '_corr.tif'),)
        else:
            key = s1['date']; lays = (('VV', '_VV.tif'), ('VH', '_VH.tif'))
        sites = ('A', 'B') if s1['track'] == 88 else ('A',)
        uid = f"{info['kind']}|{s1['year']}|{s1['track']}|{key}"
        todo = [s for s in sites if uid not in done[s]]
        if not todo: continue
        zp = os.path.join(DL, zf)
        try:
            names = [m.filename for m in zipfile.ZipFile(zp).infolist() if not m.is_dir()]
        except zipfile.BadZipFile:
            continue                                   # still downloading
        for s in todo:
            rec = dict(uid=uid, product=zf[:-4], kind=info['kind'], year=s1['year'], track=s1['track'], key=key, date1=s1['date'],
                       date2=key.split('_')[1] if info['kind'] == 'INSAR' else '', layers='')
            got = []
            for lay, suf in lays:
                m = next((n for n in names if n.endswith(suf)), None)
                if m is None: continue
                out = os.path.join(ROOT, s, 'aligned', str(s1['year']), f"T{s1['track']}", f'{lay}_{key}.tif')
                a = align(f'/vsizip/{zp}/{m}', G[s]); save(out, a, G[s])
                v = a[np.isfinite(a)]; rec[f'{lay}_valid_frac'] = round(v.size / a.size, 4)
                if lay in ('VV', 'VH') and v.size: rec[f'{lay}_median_db'] = round(float(10 * np.log10(np.median(v[v > 0]))), 3)
                got.append(lay)
            rec['layers'] = ';'.join(got)
            with open(os.path.join(ROOT, s, 'product_index.jsonl'), 'a', encoding='utf-8') as f: f.write(json.dumps(rec) + '\n')
            done[s].add(uid); n_new += 1
        print(time.strftime('%H:%M:%S'), 'aligned', zf, todo, flush=True)
    st = json.load(open(os.path.join(HERE, 'JOB_STATUS_B.json'))) if os.path.exists(os.path.join(HERE, 'JOB_STATUS_B.json')) else {'counts': {}}
    pending = st['counts'].get('PENDING', 1) + st['counts'].get('RUNNING', 0)
    if pending == 0 and n_new == 0:
        print('ALL ALIGNED', {s: len(done[s]) for s in done}, flush=True); break
    time.sleep(300)
