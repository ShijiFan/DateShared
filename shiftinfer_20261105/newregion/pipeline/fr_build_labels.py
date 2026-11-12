"""FR-A label grids and spatial splits, identical algorithm to SAR/multiyear_20261017/build_labels_my.py (pure 40 m pixels, -sqrt(2)*20 m
buffer, conflicts removed, eligible >= 2 pure pixels, 5 km blocks, 6 folds, 400 m exclusion). Differences (CONFIRM_PREREG §3):
grid = new FR grid (EPSG:32631, same origin rule as NL grid_spec); class map = newregion/FR_MAPPING_FROZEN.csv (code_cultu -> 7 classes,
unmapped -> Other); no AMBIG list (exclusions are in the mapping); block -> fold: all blocks shuffled with seed 20261106 once (first call
writes FR_BLOCK_FOLDS.json; later years reuse it, unseen blocks appended with the same seed rule as NL).
Usage: python -I fr_build_labels.py <year> -> D:\\sar_data\\fr_a\\FR\\labels\\{parcel_raster_Y.tif, parcel_table_Y.csv, spatial_splits_Y.csv, LABEL_QC_Y.json}"""
import sys, os, csv, json, math, hashlib
from collections import Counter
import numpy as np, geopandas as gpd, rasterio, shapely
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer
YEAR = int(sys.argv[1]); HERE = os.path.dirname(os.path.abspath(__file__)); NR = os.path.dirname(HERE)
AOI = box(0.55, 49.55, 1.10, 49.90); CRS = 'EPSG:32631'; PIX = 40; ERO = math.sqrt(2) * PIX / 2; SEED = 20261106
ROOT = r'D:\sar_data\fr_a'; OUT = os.path.join(ROOT, 'FR', 'labels'); os.makedirs(OUT, exist_ok=True)
GRID = os.path.join(ROOT, 'FR', 'grid_spec.json'); BF = os.path.join(ROOT, 'FR', 'FR_BLOCK_FOLDS.json')
RPG = os.path.join(ROOT, 'rpg', f'rpg_{YEAR}_aoi.gpkg')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
aoi_m = transform(Transformer.from_crs('EPSG:4326', CRS, always_xy=True).transform, AOI)
if not os.path.exists(GRID):
    mnx, mny, mxx, mxy = aoi_m.bounds; left = math.floor(mnx / PIX) * PIX; top = math.ceil(mxy / PIX) * PIX
    W = int(math.ceil((mxx - left) / PIX)); H = int(math.ceil((top - mny) / PIX))
    json.dump(dict(crs=CRS, pixel_m=PIX, width=W, height=H, affine_gdal=[left, PIX, 0.0, top, 0.0, -PIX],
                   origin_rule='project AOI EPSG:4326 -> EPSG:32631; left=floor(minx/40)*40; top=ceil(maxy/40)*40 (same rule as NL grid_spec)'), open(GRID, 'w'), indent=1)
gs = json.load(open(GRID)); W, H = int(gs['width']), int(gs['height']); TR = Affine.from_gdal(*gs['affine_gdal'])
aoi_mask = rasterize([(aoi_m.buffer(-ERO), 1)], out_shape=(H, W), transform=TR, fill=0, dtype='uint8') != 0
rule = {r['code_cultu']: r['class'] for r in csv.DictReader(open(os.path.join(NR, 'FR_MAPPING_FROZEN.csv'), encoding='utf-8-sig'))}
gdf = gpd.read_file(RPG).to_crs(CRS); gdf = gdf[gdf.intersects(aoi_m)].reset_index(drop=True)
geoms = np.asarray(gdf.geometry.values, dtype=object)
valid = shapely.is_valid(geoms); empty = shapely.is_empty(geoms) | shapely.is_missing(geoms)
if (~valid & ~empty).any(): geoms[~valid & ~empty] = shapely.make_valid(geoms[~valid & ~empty])
codes = [str(v).strip() if v is not None else '' for v in gdf['code_cultu']]; labels = [rule.get(c, 'Other') for c in codes]
buf = shapely.buffer(geoms, -ERO); keep = ~(shapely.is_empty(buf) | shapely.is_missing(buf))
shapes = [(g, i + 1) for i, g in enumerate(buf) if keep[i]]
cnt = rasterize(((g, 1) for g, _ in shapes), out_shape=(H, W), transform=TR, fill=0, dtype='uint16', merge_alg=MergeAlg.add)
ids = rasterize(shapes, out_shape=(H, W), transform=TR, fill=0, dtype='uint32'); ids[(cnt > 1) | ~aoi_mask] = 0
per = np.bincount(ids.ravel(), minlength=len(gdf) + 1)
with rasterio.open(os.path.join(OUT, f'parcel_raster_{YEAR}.tif'), 'w', driver='GTiff', height=H, width=W, count=1, dtype='uint32', crs=CRS,
                   transform=TR, nodata=0, compress='LZW', tiled=True) as d: d.write(ids, 1)
cen = shapely.centroid(geoms); cx, cy = shapely.get_x(cen), shapely.get_y(cen); area = shapely.area(geoms) / 1e4
rows = []
for i, g in enumerate(geoms):
    fp = hashlib.sha256(shapely.to_wkb(shapely.normalize(g))).hexdigest()[:16]
    rows.append(dict(year=YEAR, parcel_id=f'{YEAR}_{i + 1:06d}_{fp}', source_row=i + 1, code_cultu=codes[i], **{'class': labels[i]}, area_ha=round(float(area[i]), 6),
                     pure_pixels_40m=int(per[i + 1]), eligible_ge2=int(per[i + 1] >= 2), block_5km=f'{math.floor(cx[i] / 5000)}_{math.floor(cy[i] / 5000)}'))
with open(os.path.join(OUT, f'parcel_table_{YEAR}.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
bmap = {k: int(v) for k, v in json.load(open(BF)).items()} if os.path.exists(BF) else {}
unseen = sorted({r['block_5km'] for r in rows if r['eligible_ge2'] and r['block_5km'] not in bmap})
base = len(bmap); rng = np.random.default_rng(SEED + base); sh = unseen.copy(); rng.shuffle(sh)
for i, b in enumerate(sh): bmap[b] = (base + i) % 6
json.dump(bmap, open(BF, 'w'), indent=1)
gdf['eligible'] = [r['eligible_ge2'] for r in rows]; gdf['fold'] = [bmap.get(r['block_5km'], -1) for r in rows]; gdf['geometry'] = geoms
el = gdf[gdf.eligible == 1].reset_index(); eg = np.asarray(el.geometry.values, dtype=object); tree = shapely.STRtree(eg)
roles = {}
for k in range(6):
    te = np.flatnonzero(el.fold.values == k); near = set(tree.query(eg[te], predicate='dwithin', distance=400.0)[1]) - set(te)
    for j in range(len(el)):
        roles[(k, int(el.loc[j, 'index']))] = 'TEST' if el.fold.values[j] == k else ('EXCLUDED_400M' if j in near else 'TRAIN')
srows = []
for i, r in enumerate(rows):
    e = dict(year=YEAR, parcel_id=r['parcel_id'], source_row=r['source_row'], code_cultu=r['code_cultu'], class_strict=r['class'], class_permissive=r['class'],
             pure_pixels_40m=r['pure_pixels_40m'], eligible_ge2=r['eligible_ge2'], block_5km=r['block_5km'], assigned_test_fold=bmap.get(r['block_5km'], -1))
    for k in range(6): e[f'fold_{k}_role'] = roles.get((k, i), 'INELIGIBLE_LT2') if r['eligible_ge2'] else 'INELIGIBLE_LT2'
    srows.append(e)
with open(os.path.join(OUT, f'spatial_splits_{YEAR}.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)
elig = [r for r in rows if r['eligible_ge2']]
qc = dict(site='FR', year=YEAR, rpg=RPG, rpg_sha256=sha(RPG), mapping_sha256=sha(os.path.join(NR, 'FR_MAPPING_FROZEN.csv')), rows=len(rows), eligible=len(elig),
          eligible_by_class=dict(Counter(r['class'] for r in elig)), unseen_blocks=len(unseen), pure_pixels=int((ids > 0).sum()), grid=gs)
json.dump(qc, open(os.path.join(OUT, f'LABEL_QC_{YEAR}.json'), 'w'), indent=1); print(json.dumps({k: v for k, v in qc.items() if k != 'grid'}))
