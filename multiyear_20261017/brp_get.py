"""Robust single-year BRP download (PDOK has no HTTP Range support and drops long connections): retry the full download up to
12 times until the byte count equals Content-Length; 2017-2019 are zips (read via /vsizip/), 2020+ GeoPackages. Crop to the union bbox
of Site A and Site B, save brp_{year}_aoi.gpkg (skipped if present), delete the national file. Log: brp_get_{year}.json
Usage: python brp_get.py <year>"""
import os, sys, json, hashlib, time, zipfile
import requests, geopandas as gpd
from shapely.geometry import box
y = int(sys.argv[1]); OUT = os.path.dirname(os.path.abspath(__file__)); D = r'D:\sar_data\multiyear\brp'
out = os.path.join(D, f'brp_{y}_aoi.gpkg')
if os.path.exists(out): print('exists'); raise SystemExit(0)
ext = 'zip' if y <= 2019 else 'gpkg'
url = f'https://service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_{y}.{ext}'
nat = os.path.join(D, f'get_{y}.{ext}'); aoi = gpd.GeoSeries([box(6.00, 51.98, 7.05, 53.10)], crs='EPSG:4326').to_crs('EPSG:28992')
for attempt in range(12):
    h = hashlib.sha256(); n = 0
    try:
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status(); L = int(r.headers['Content-Length'])
            with open(nat, 'wb') as f:
                for c in r.iter_content(8 << 20): f.write(c); h.update(c); n += len(c)
        if n == L: break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'attempt', attempt, 'failed at', n, repr(e)[:120], flush=True)
    time.sleep(30)
else:
    raise SystemExit('gave up')
src = nat
if ext == 'zip':
    names = zipfile.ZipFile(nat).namelist()
    inner = next((m for m in names if m.lower().endswith('.gpkg')), None) or next(m for m in names if m.lower().endswith('.shp'))
    src = f'/vsizip/{nat}/{inner}'
gdf = gpd.read_file(src, bbox=tuple(aoi.total_bounds))
if gdf.crs is None: gdf = gdf.set_crs(28992)
gdf.to_file(out + '.tmp.gpkg', driver='GPKG'); os.replace(out + '.tmp.gpkg', out)
log = dict(url=url, national_sha256=h.hexdigest(), national_bytes=n, attempts=attempt + 1, source=src, aoi_rows=len(gdf), crs=str(gdf.crs),
           years_in_file=sorted(set(gdf['jaar'].astype(str))) if 'jaar' in gdf else None, columns=list(gdf.columns),
           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
os.remove(nat); json.dump(log, open(os.path.join(OUT, f'brp_get_{y}.json'), 'w'), indent=1); print(y, len(gdf), 'rows', list(gdf.columns))
