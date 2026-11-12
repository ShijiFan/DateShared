"""BRP 2017-2019 zips (downloaded completely by brp_get.py, byte count = Content-Length) contain an Esri FileGDB, not gpkg/shp.
Read the polygon layer via /vsizip/ (OpenFileGDB), crop to the union bbox of Site A and B, save brp_{year}_aoi.gpkg, log sha256, delete zip.
Usage: python brp_gdb.py <year>  -> brp_get_{year}.json"""
import os, sys, json, hashlib, time, zipfile
import geopandas as gpd, pyogrio
from shapely.geometry import box
y = int(sys.argv[1]); OUT = os.path.dirname(os.path.abspath(__file__)); D = r'D:\sar_data\multiyear\brp'
zp = os.path.join(D, f'get_{y}.zip'); out = os.path.join(D, f'brp_{y}_aoi.gpkg')
h = hashlib.sha256()
with open(zp, 'rb') as f:
    for c in iter(lambda: f.read(8 << 20), b''): h.update(c)
gdb = sorted({n.split('/')[0] for n in zipfile.ZipFile(zp).namelist() if '.gdb/' in n})[0]
src = f'/vsizip/{zp}/{gdb}'; layers = pyogrio.list_layers(src); print(layers)
lay = next(str(l[0]) for l in layers if l[1] and 'Polygon' in str(l[1]))
aoi = gpd.GeoSeries([box(6.00, 51.98, 7.05, 53.10)], crs='EPSG:4326').to_crs('EPSG:28992')
gdf = gpd.read_file(src, layer=lay, bbox=tuple(aoi.total_bounds))
if gdf.crs is None: gdf = gdf.set_crs(28992)
gdf.columns = [c if c == 'geometry' else c.lower() for c in gdf.columns]
gdf.to_file(out + '.tmp.gpkg', driver='GPKG'); os.replace(out + '.tmp.gpkg', out)
log = dict(url=f'https://service.pdok.nl/rvo/gewaspercelen/atom/downloads/brpgewaspercelen_definitief_{y}.zip', zip_sha256=h.hexdigest(),
           zip_bytes=os.path.getsize(zp), gdb=gdb, layer=lay, layers=[[str(a), str(b)] for a, b in layers], aoi_rows=len(gdf), crs=str(gdf.crs),
           columns=list(gdf.columns), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
os.remove(zp); json.dump(log, open(os.path.join(OUT, f'brp_get_{y}.json'), 'w'), indent=1); print(y, len(gdf), 'rows', list(gdf.columns))
