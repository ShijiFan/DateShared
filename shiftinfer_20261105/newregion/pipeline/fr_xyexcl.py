"""FR-A cross-year 400 m exclusion, identical algorithm to SAR/multiyear_20261017/build_xyexcl.py: for every ordered year pair (s != t) and
fold k, source-year TRAIN parcels strictly < 400 m from a target-year fold-k TEST parcel are dropped. Geometries are re-read in exactly the
order of fr_build_labels.py (RPG gpkg -> EPSG:32631 -> intersects AOI -> make_valid) and verified by the parcel_table fingerprints.
Usage: python -I fr_xyexcl.py -> D:\\sar_data\\fr_a\\xyexcl_FR.npz, XYEXCL_AUDIT_FR.csv; drop(s, t, k, n) helper below."""
import os, json, hashlib
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer
ROOT = r'D:\sar_data\fr_a'; CRS = 'EPSG:32631'; AOI = box(0.55, 49.55, 1.10, 49.90); D = 400.0
_C = {}


def drop(s, t, k, n):
    m = np.zeros(n, bool)
    if int(s) == int(t): return m
    if 'FR' not in _C: _C['FR'] = np.load(os.path.join(ROOT, 'xyexcl_FR.npz'))
    m[_C['FR'][f'{int(s)}_{int(t)}_{int(k)}']] = True
    return m


def load(y):
    aoi_m = transform(Transformer.from_crs('EPSG:4326', CRS, always_xy=True).transform, AOI)
    g = gpd.read_file(os.path.join(ROOT, 'rpg', f'rpg_{y}_aoi.gpkg')).to_crs(CRS); g = g[g.intersects(aoi_m)].reset_index(drop=True)
    geoms = np.asarray(g.geometry.values, dtype=object); bad = ~shapely.is_valid(geoms) & ~shapely.is_empty(geoms) & ~shapely.is_missing(geoms)
    geoms[bad] = shapely.make_valid(geoms[bad])
    got = [hashlib.sha256(w).hexdigest()[:16] for w in shapely.to_wkb(shapely.normalize(geoms))]
    exp = pd.read_csv(os.path.join(ROOT, 'FR', 'labels', f'parcel_table_{y}.csv')).parcel_id.str.rsplit('_', n=1).str[-1].tolist()
    assert got == exp, f'{y}: geometry/table mismatch'
    return geoms


if __name__ == '__main__':
    Z = np.load(os.path.join(ROOT, 'features_FR.npz'), allow_pickle=True); M = json.load(open(os.path.join(ROOT, 'features_FR_meta.json')))
    tags = sorted(M); years = [int(t.split('_')[0]) for t in tags]; G = {y: load(y) for y in years}; print('geometries verified', years, flush=True)
    rows = {y: Z[f'{t}_row'] for y, t in zip(years, tags)}; F = {y: Z[f'{t}_folds'] for y, t in zip(years, tags)}; out, audit = {}, []
    for t in years:
        for k in range(6):
            ti = np.flatnonzero(F[t][:, k] == 'TEST'); tg = G[t][rows[t][ti] - 1]; tree = shapely.STRtree(tg)
            for s in years:
                if s == t: continue
                si = np.flatnonzero(F[s][:, k] == 'TRAIN'); sg = G[s][rows[s][si] - 1]; q = tree.query(sg, predicate='dwithin', distance=D)
                near = np.unique(q[0][shapely.distance(sg[q[0]], tg[q[1]]) < D]) if q.shape[1] else np.array([], dtype=int)
                dr = si[near]; out[f'{s}_{t}_{k}'] = dr.astype(np.int32); keep = np.setdiff1d(si, dr)
                q2 = tree.query(G[s][rows[s][keep] - 1], predicate='dwithin', distance=D)
                mind = float(shapely.distance(G[s][rows[s][keep[q2[0]]] - 1], tg[q2[1]]).min()) if q2.shape[1] else None
                assert mind is None or mind >= D - 1e-6, (s, t, k, mind)
                audit.append(dict(src=s, tgt=t, fold=k, n_train=len(si), n_drop=len(dr), frac_drop=round(len(dr) / max(len(si), 1), 5), min_kept_distance_m=mind))
    np.savez_compressed(os.path.join(ROOT, 'xyexcl_FR.npz'), **out); A = pd.DataFrame(audit)
    A.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'XYEXCL_AUDIT_FR.csv'), index=False)
    print('done', len(out), 'drop frac median', round(float(A.frac_drop.median()), 4), 'max', round(float(A.frac_drop.max()), 4))
