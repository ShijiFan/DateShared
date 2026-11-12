"""AMEND_SPATIAL_20261021.md: cross-year 400 m exclusion lists for every group, ordered year pair (s != t, 2017-2025) and fold.
Geometries: original BRP polygons (EPSG:32632), matched to feature rows (source_row) and verified by the parcel_table geometry
fingerprints. Output D:\\sar_data\\multiyear\\xyexcl_<group>.npz with key '<s>_<t>_<k>' = positions (into the source-year feature
arrays) of TRAIN parcels to drop; XYEXCL_AUDIT.csv; assertion that kept TRAIN parcels are >= 400 m from all target TEST parcels."""
import json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, shapely, pyogrio
from shapely.geometry import box
HERE = Path(__file__).resolve().parent; ROOT = Path(r'D:\sar_data\multiyear'); SAR = Path(r'E:\research\SAR')
AOI = {'A': (6.00, 51.98, 6.54, 52.38), 'B': (6.55, 52.75, 7.05, 53.10)}
YEARS = list(range(2017, 2026)); D = 400.0; t0 = time.time(); inputs = {}
def reg(p):
    p = Path(p); inputs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]; return p
def fix(g):
    g = g.to_crs(32632); geoms = np.asarray(g.geometry.values, dtype=object)
    bad = ~shapely.is_valid(geoms) & ~shapely.is_empty(geoms) & ~shapely.is_missing(geoms); geoms[bad] = shapely.make_valid(geoms[bad]); return geoms
def fp(geoms): return [hashlib.sha256(w).hexdigest()[:16] for w in shapely.to_wkb(shapely.normalize(geoms))]
def load(site, y):
    if y <= 2023:
        gp = reg(ROOT / 'brp' / f'brp_{y}_aoi.gpkg'); crs = pyogrio.read_info(gp)['crs']; aoi = gpd.GeoSeries([box(*AOI[site])], crs=4326).to_crs(crs).iloc[0]
        g = gpd.read_file(gp, bbox=tuple(aoi.bounds), fid_as_index=True).sort_index(); g = g[g.intersects(aoi)].reset_index(drop=True)
        table = pd.read_csv(reg(ROOT / site / 'labels' / f'parcel_table_{y}.csv')); expected = table.parcel_id.str.rsplit('_', n=1).str[-1].tolist()
    elif site == 'A':
        src = {2024: 'brp_parcels_2024_aoi.geojson', 2025: 'brp_parcels_aoi.geojson'}[y]
        g = gpd.read_file(reg(SAR / 'sar_v2' / 'labels' / 'nl_brp' / src))
        table = pd.read_csv(reg(SAR / 'A1_observation_budget_20260923' / 'runs' / '20260923_G0L_COMMON_GRID' / f'parcel_table_{y}.csv')); expected = table.geometry_sha256_16.tolist()
    else:
        gp = reg(SAR / 'sar_v2' / 'labels' / 'nl_brp' / f'brpgewaspercelen_definitief_{y}.gpkg')
        aoi = gpd.GeoSeries([box(*AOI[site])], crs=4326).to_crs(28992).iloc[0]
        g = gpd.read_file(gp, bbox=tuple(aoi.bounds)); g = g[g.intersects(aoi)].reset_index(drop=True)      # same read order as s2_03_grid_labels.py
        table = pd.read_csv(reg(SAR / 'site2_20260930' / 'runs' / 'G0L_COMMON_GRID' / f'parcel_table_{y}.csv'))
        expected = table['geometry_sha256_16'].tolist() if 'geometry_sha256_16' in table else table.parcel_id.str.rsplit('_', n=1).str[-1].tolist()
    geoms = fix(g); got = fp(geoms)
    if got != expected and sorted(got) == sorted(expected) and len(set(got)) == len(got):    # same polygons, different order: map by unique fingerprint
        pos = {f: i for i, f in enumerate(got)}; geoms = geoms[[pos[f] for f in expected]]; got = expected; print(site, y, 'reordered by fingerprint', flush=True)
    assert len(geoms) == len(table) and got == expected, f'{site} {y}: geometry/table mismatch ({len(geoms)} vs {len(table)})'
    return geoms
GROUPS = [('A', 37), ('A', 88), ('B', 88)]; audit = []
for site in 'AB':
    Z = np.load(reg(ROOT / f'features_{site}.npz'), allow_pickle=True); G = {y: load(site, y) for y in YEARS}
    print(site, 'geometries verified', round(time.time() - t0), 's', flush=True)
    for s_, trk in [g for g in GROUPS if g[0] == site]:
        grp = f'{site}{trk}'; out = {}
        rows = {y: Z[f'{y}_T{trk}_row'] for y in YEARS}; F = {y: Z[f'{y}_T{trk}_folds'] for y in YEARS}
        for t in YEARS:
            for k in range(6):
                ti = np.flatnonzero(F[t][:, k] == 'TEST'); tg = G[t][rows[t][ti] - 1]; tree = shapely.STRtree(tg)
                for s in YEARS:
                    if s == t: continue
                    si = np.flatnonzero(F[s][:, k] == 'TRAIN'); sg = G[s][rows[s][si] - 1]
                    q = tree.query(sg, predicate='dwithin', distance=D)
                    near = np.unique(q[0]) if q.shape[1] else np.array([], dtype=int)
                    if q.shape[1]:                       # dwithin is inclusive; drop only strict < 400 m
                        dist = shapely.distance(sg[q[0]], tg[q[1]]); near = np.unique(q[0][dist < D])
                    drop = si[near]; out[f'{s}_{t}_{k}'] = drop.astype(np.int32)
                    keep = np.setdiff1d(si, drop)          # verification on the kept set
                    q2 = tree.query(G[s][rows[s][keep] - 1], predicate='dwithin', distance=D)
                    mind = float(shapely.distance(G[s][rows[s][keep[q2[0]]] - 1], tg[q2[1]]).min()) if q2.shape[1] else None
                    assert mind is None or mind >= D - 1e-6, (grp, s, t, k, mind)
                    ys = Z[f'{s}_T{trk}_y']
                    audit.append(dict(group=grp, src=s, tgt=t, fold=k, n_train_orig=len(si), n_drop=len(drop), frac_drop=round(len(drop) / max(len(si), 1), 5),
                                      min_kept_class_count=int(np.bincount(ys[keep], minlength=7).min()), min_kept_distance_m=mind))
        np.savez_compressed(ROOT / f'xyexcl_{grp}.npz', **out); print(grp, 'done', round(time.time() - t0), 's', flush=True)
A = pd.DataFrame(audit); A.to_csv(HERE / 'XYEXCL_AUDIT.csv', index=False)
summ = dict(run_at=time.strftime('%Y-%m-%d %H:%M:%S'), n_combinations=len(A), drop_frac_median=float(A.frac_drop.median()), drop_frac_max=float(A.frac_drop.max()),
            n_drop_total=int(A.n_drop.sum()), combos_with_class_below_20=int((A.min_kept_class_count < 20).sum()), by_group=A.groupby('group').frac_drop.describe().round(4).to_dict('index'),
            inputs_sha256=inputs)
(HERE / 'XYEXCL_SUMMARY.json').write_text(json.dumps(summ, indent=1)); print(json.dumps({k: v for k, v in summ.items() if k != 'inputs_sha256'}, indent=1))
