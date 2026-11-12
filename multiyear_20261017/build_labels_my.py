"""Multi-year label grids and spatial splits (2017-2023) for Site A and Site B, identical algorithm to
A1 build_g0l_common_grid_20260923.py / site2 s2_03_grid_labels.py (pure 40 m pixels, -sqrt(2)*20 m buffer, conflicts removed,
eligible >= 2 pure pixels, 5 km blocks) and s2_04_splits.py (6 folds, 400 m exclusion, strict/permissive classes).
Grid: the EXISTING site grid_spec.json (so SAR alignment matches 2024/2025). Block->fold map: the EXISTING
SPATIAL_SPLIT_REPORT.json of the site (cross-year consistency); unseen blocks are assigned with the site seed applied to the
sorted list of unseen blocks (rule fixed before any 2017-2023 data were inspected; PREREG deviation record).
Crop code -> class: A1 mapping crop_code_mapping_2024_2025.csv; unmapped codes -> Other (fraction reported).
Usage: python build_labels_my.py <site A|B> <year>
Outputs: D:\\sar_data\\multiyear\\<site>\\labels\\{parcel_raster_Y.tif, parcel_table_Y.csv, spatial_splits_Y.csv, LABEL_QC_Y.json}"""
import sys, os, csv, json, math, hashlib
from collections import Counter
import numpy as np, geopandas as gpd, rasterio, shapely
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer
SITE, YEAR = sys.argv[1], int(sys.argv[2])
SAR = r'E:\research\SAR'
CFG = {'A': dict(aoi=box(6.00, 51.98, 6.54, 52.38), grid=os.path.join(SAR, r'A1_observation_budget_20260923\runs\20260923_G0L_COMMON_GRID\grid_spec.json'),
                 report=os.path.join(SAR, r'A1_observation_budget_20260923\runs\20260923_G0L_COMMON_GRID\SPATIAL_SPLIT_REPORT.json'), seed=20260923),
       'B': dict(aoi=box(6.55, 52.75, 7.05, 53.10), grid=os.path.join(SAR, r'site2_20260930\runs\G0L_COMMON_GRID\grid_spec.json'),
                 report=os.path.join(SAR, r'site2_20260930\runs\G0L_COMMON_GRID\SPATIAL_SPLIT_REPORT.json'), seed=20260930)}[SITE]
MAPPING = os.path.join(SAR, r'A1_observation_budget_20260923\runs\20260923_0140_G0R\qc\crop_code_mapping_2024_2025.csv')
AMBIG = {238, 314, 382, 670, 2652, 6636, 7130}
OUT = os.path.join(r'D:\sar_data\multiyear', SITE, 'labels'); os.makedirs(OUT, exist_ok=True)
BRP = os.path.join(r'D:\sar_data\multiyear\brp', f'brp_{YEAR}_aoi.gpkg')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(4 << 20), b''): h.update(c)
    return h.hexdigest()
gs = json.load(open(CFG['grid'])); W, H = int(gs['width']), int(gs['height']); TR = Affine.from_gdal(*gs['affine_gdal']); CRS = gs['crs']
PIX = 40; ERO = math.sqrt(2) * PIX / 2
aoi_m = transform(Transformer.from_crs('EPSG:4326', CRS, always_xy=True).transform, CFG['aoi'])
aoi_mask = rasterize([(aoi_m.buffer(-ERO), 1)], out_shape=(H, W), transform=TR, fill=0, dtype='uint8') != 0
rule = {int(r['gewascode']): r['assigned_8class'] for r in csv.DictReader(open(MAPPING, encoding='utf-8-sig'))}
gdf = gpd.read_file(BRP)
code_col = next(c for c in gdf.columns if c.lower() in ('gewascode', 'gws_gewascode', 'gws_gewasc', 'gewas_code', 'gewascod'))
aoi_rd = gpd.GeoSeries([CFG['aoi']], crs='EPSG:4326').to_crs(gdf.crs)
gdf = gdf[gdf.intersects(aoi_rd.iloc[0])].reset_index(drop=True).to_crs(CRS)
geoms = np.asarray(gdf.geometry.values, dtype=object)
valid = shapely.is_valid(geoms); empty = shapely.is_empty(geoms) | shapely.is_missing(geoms)
if (~valid & ~empty).any(): geoms[~valid & ~empty] = shapely.make_valid(geoms[~valid & ~empty])
codes = [int(v) if v is not None and str(v).strip().isdigit() else -1 for v in gdf[code_col]]
labels = [rule.get(c, 'Other') for c in codes]
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
    rows.append(dict(year=YEAR, parcel_id=f'{YEAR}_{i + 1:06d}_{fp}', source_row=i + 1, gewascode=codes[i], **{'class': labels[i]}, area_ha=round(float(area[i]), 6),
                     pure_pixels_40m=int(per[i + 1]), eligible_ge2=int(per[i + 1] >= 2), block_5km=f'{math.floor(cx[i] / 5000)}_{math.floor(cy[i] / 5000)}'))
with open(os.path.join(OUT, f'parcel_table_{YEAR}.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# spatial splits with the existing block->fold map
bmap = {k: int(v) for k, v in json.load(open(CFG['report']))['block_assignments'].items()}
unseen = sorted({r['block_5km'] for r in rows if r['eligible_ge2'] and r['block_5km'] not in bmap})
rng = np.random.default_rng(CFG['seed']); sh = unseen.copy(); rng.shuffle(sh)
for i, b in enumerate(sh): bmap[b] = i % 6
gdf['eligible'] = [r['eligible_ge2'] for r in rows]; gdf['fold'] = [bmap.get(r['block_5km'], -1) for r in rows]; gdf['geometry'] = geoms
el = gdf[gdf.eligible == 1].reset_index(); eg = np.asarray(el.geometry.values, dtype=object); tree = shapely.STRtree(eg)
roles = {}
for k in range(6):
    te = np.flatnonzero(el.fold.values == k); near = set(tree.query(eg[te], predicate='dwithin', distance=400.0)[1]) - set(te)
    for j in range(len(el)):
        roles[(k, int(el.loc[j, 'index']))] = 'TEST' if el.fold.values[j] == k else ('EXCLUDED_400M' if j in near else 'TRAIN')
srows = []
for i, r in enumerate(rows):
    e = dict(year=YEAR, parcel_id=r['parcel_id'], source_row=r['source_row'], gewascode=r['gewascode'],
             class_strict='Other' if r['gewascode'] in AMBIG else r['class'], class_permissive=r['class'], pure_pixels_40m=r['pure_pixels_40m'],
             eligible_ge2=r['eligible_ge2'], block_5km=r['block_5km'], assigned_test_fold=bmap.get(r['block_5km'], -1))
    for k in range(6): e[f'fold_{k}_role'] = roles.get((k, i), 'INELIGIBLE_LT2') if r['eligible_ge2'] else 'INELIGIBLE_LT2'
    srows.append(e)
with open(os.path.join(OUT, f'spatial_splits_{YEAR}.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(srows[0])); w.writeheader(); w.writerows(srows)
elig = [r for r in rows if r['eligible_ge2']]
qc = dict(site=SITE, year=YEAR, brp=BRP, brp_sha256=sha(BRP), code_column=code_col, rows=len(rows), eligible=len(elig),
          eligible_by_class=dict(Counter(r['class'] for r in elig)), unmapped_code_frac_eligible=float(np.mean([r['gewascode'] not in rule for r in elig])),
          unseen_blocks=len(unseen), pure_pixels=int((ids > 0).sum()))
json.dump(qc, open(os.path.join(OUT, f'LABEL_QC_{YEAR}.json'), 'w'), indent=1); print(json.dumps(qc))
