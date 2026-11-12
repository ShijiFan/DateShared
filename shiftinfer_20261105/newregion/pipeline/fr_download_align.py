"""FR-A: poll submitted HyP3 jobs (never submits), download finished zips (parallel, size-checked, .part -> rename) to
D:\\sar_data\\fr_a\\downloads, and align every product to the FR 40 m grid with the NL resampling (linear gamma0 VV/VH and coherence,
Resampling.average; identical align() to SAR/multiyear_20261017/align_my.py). Loops until all jobs are finished and aligned.
Index: D:\\sar_data\\fr_a\\FR\\product_index.jsonl ; status: newregion/FR-A_PaysDeCaux/JOB_STATUS.json"""
import os, csv, json, time, re, shutil, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, rasterio, requests, hyp3_sdk
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
HERE = os.path.dirname(os.path.abspath(__file__)); PLAN = os.path.join(os.path.dirname(HERE), 'FR-A_PaysDeCaux')
ROOT = r'D:\sar_data\fr_a'; DL = os.path.join(ROOT, 'downloads'); os.makedirs(DL, exist_ok=True)
gs = json.load(open(os.path.join(ROOT, 'FR', 'grid_spec.json'))); GRID = (gs['crs'], int(gs['width']), int(gs['height']), Affine.from_gdal(*gs['affine_gdal']))
cat = {}
for r in csv.DictReader(open(os.path.join(PLAN, 'plan_pairs.csv'), encoding='utf-8')):
    if not r['scene1']: continue
    for sc, d in ((r['scene1'], r['date1']), (r['scene2'], r['date2'])): cat[sc.split('_')[5]] = dict(year=int(r['year']), orbit=int(r['orbit']), date=d)


def parse(stem):
    p = stem.split('_')
    if stem.startswith('S1') and '_RTC' in stem: return dict(kind='RTC', key1=p[2])
    if '_INT40_' in stem: return dict(kind='INSAR', key1=p[1], key2=p[2])
    return None


def align(src):
    crs, W, H, TR = GRID
    with rasterio.open(src) as s:
        a = s.read(1).astype('float32')
        if s.nodata is not None: a[a == s.nodata] = np.nan
        a[~np.isfinite(a)] = np.nan
        d = np.full((H, W), np.nan, dtype='float32')
        reproject(a, d, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan, dst_transform=TR, dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.average)
    return d


def save(path, a):
    crs, W, H, TR = GRID; os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path[:-4] + '.tmp.tif'
    with rasterio.open(tmp, 'w', driver='GTiff', dtype='float32', nodata=np.nan, width=W, height=H, count=1, crs=crs, transform=TR, compress='deflate') as d: d.write(a, 1)
    os.replace(tmp, path)


def get(fd):
    tgt = os.path.join(DL, fd['filename'])
    if os.path.exists(tgt) and os.path.getsize(tgt) == fd['size']: return 'skip'
    if shutil.disk_usage(DL).free / 1e9 < 40: return 'disk'
    part = tgt + '.part'
    for _ in range(3):
        try:
            with requests.get(fd['url'], stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(part, 'wb') as f:
                    for c in r.iter_content(8 << 20): f.write(c)
            if os.path.getsize(part) == fd['size']: os.replace(part, tgt); return 'ok'
        except Exception as e: print('retry', fd['filename'], repr(e), flush=True)
    return 'fail'


def align_all():
    ip = os.path.join(ROOT, 'FR', 'product_index.jsonl')
    done = {json.loads(l)['uid'] for l in open(ip, encoding='utf-8')} if os.path.exists(ip) else set(); n = 0
    for zf in sorted(f for f in os.listdir(DL) if f.endswith('.zip')):
        info = parse(zf[:-4]); s1 = cat.get(info['key1']) if info else None
        if s1 is None: continue
        if info['kind'] == 'INSAR': key = f"{s1['date']}_{cat[info['key2']]['date']}"; lays = (('COH', '_corr.tif'),)
        else: key = s1['date']; lays = (('VV', '_VV.tif'), ('VH', '_VH.tif'))
        uid = f"{info['kind']}|{s1['year']}|{s1['orbit']}|{key}"
        if uid in done: continue
        zp = os.path.join(DL, zf); names = [m.filename for m in zipfile.ZipFile(zp).infolist() if not m.is_dir()]
        rec = dict(uid=uid, product=zf[:-4], kind=info['kind'], year=s1['year'], track=s1['orbit'], key=key, date1=s1['date'], date2=key.split('_')[1] if info['kind'] == 'INSAR' else '')
        got = []
        for lay, suf in lays:
            m = next((x for x in names if x.endswith(suf)), None)
            if m is None: continue
            a = align(f'/vsizip/{zp}/{m}'); save(os.path.join(ROOT, 'FR', 'aligned', str(s1['year']), f"T{s1['orbit']}", f'{lay}_{key}.tif'), a)
            v = a[np.isfinite(a)]; rec[f'{lay}_valid_frac'] = round(v.size / a.size, 4)
            if lay in ('VV', 'VH') and v.size: rec[f'{lay}_median_db'] = round(float(10 * np.log10(np.median(v[v > 0]))), 3)
            if lay == 'COH' and v.size: rec['COH_median'] = round(float(np.median(v)), 3)
            got.append(lay)
        rec['layers'] = ';'.join(got)
        with open(ip, 'a', encoding='utf-8') as f: f.write(json.dumps(rec) + '\n')
        done.add(uid); n += 1; print(time.strftime('%H:%M:%S'), 'aligned', uid, flush=True)
    return n


h = hyp3_sdk.HyP3(); t_end = time.time() + 30 * 3600
while True:
    ids = [r['job_id'] for r in csv.DictReader(open(os.path.join(PLAN, 'SUBMITTED_JOBS.csv'), encoding='utf-8'))]
    jobs = [h.get_job_by_id(x) for x in ids]; counts = {}
    for j in jobs: counts[j.status_code] = counts.get(j.status_code, 0) + 1
    print(time.strftime('%H:%M:%S'), counts, flush=True)
    fds = [fd for j in jobs if j.status_code == 'SUCCEEDED' for fd in (j.files or [])]
    if os.environ.get('FR_LOWMEM') == '1':          # low-memory mode after memory-pressure stops: one download, then align, sequentially
        res, na = [], 0
        for fd in fds:
            res.append(get(fd))
            if res[-1] == 'ok': na += align_all()
        na += align_all()
    else:
        with ThreadPoolExecutor(int(os.environ.get('FR_NW', 6))) as ex: res = list(ex.map(get, fds))
        na = align_all()
    print(time.strftime('%H:%M:%S'), {k: res.count(k) for k in set(res)}, flush=True)
    json.dump(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), counts=counts, jobs=[dict(job_id=j.job_id, name=j.name, job_type=j.job_type, status=j.status_code,
              files=';'.join(f['filename'] for f in (j.files or []))) for j in jobs]), open(os.path.join(PLAN, 'JOB_STATUS.json'), 'w'), indent=1)
    if 'disk' in res: print('STOP disk'); raise SystemExit(1)
    if not any(j.status_code in ('PENDING', 'RUNNING') for j in jobs) and 'fail' not in res and na == 0: print('ALL DONE'); break
    if time.time() > t_end: print('timeout'); break
    time.sleep(300)
