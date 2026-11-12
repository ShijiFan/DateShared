"""Single-year BRP download (same as brp_download.py, run in parallel because PDOK serves ~0.2 MB/s per connection).
Usage: python brp_one.py <year>  -> D:\\sar_data\\multiyear\\brp\\brp_{year}_aoi.gpkg, log brp_one_{year}.json"""
import os, sys, json, hashlib, time
import requests, geopandas as gpd
from shapely.geometry import box
y = int(sys.argv[1]); OUT = os.path.dirname(os.path.abspath(__file__)); D = r'D:\sar_data\multiyear\brp'
aoi = gpd.GeoSeries([box(6.00, 51.98, 7.05, 53.10)], crs='EPSG:4326').to_crs('EPSG:28992')
url = f'https://service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_{y}.gpkg'
nat = os.path.join(D, f'one_{y}_' + url.split('/')[-1]); out = os.path.join(D, f'brp_{y}_aoi.gpkg'); h = hashlib.sha256()
with requests.get(url, stream=True, timeout=600) as r:
    r.raise_for_status()
    with open(nat, 'wb') as f:
        for c in r.iter_content(8 << 20): f.write(c); h.update(c)
gdf = gpd.read_file(nat, bbox=tuple(aoi.total_bounds)); gdf.to_file(out + '.tmp.gpkg', driver='GPKG'); os.replace(out + '.tmp.gpkg', out)
log = dict(url=url, national_sha256=h.hexdigest(), national_bytes=os.path.getsize(nat), aoi_rows=len(gdf),
           years_in_file=sorted(set(gdf['jaar'].astype(str))) if 'jaar' in gdf else None,
           status_values=sorted(set(gdf['status'].astype(str))) if 'status' in gdf else None,
           columns=list(gdf.columns), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
os.remove(nat); json.dump(log, open(os.path.join(OUT, f'brp_one_{y}.json'), 'w'), indent=1); print(y, len(gdf), 'rows')
