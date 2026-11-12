"""P2 step 1: dump parcel feature blocks for one site using that site's own A1-style modules (same data access as TGRS).
Usage: D:\\anaconda\\envs\\sar\\python.exe p2_dump.py <scripts_dir> <site_tag>
Output: dump_<site>.npz with, per (year, track, M): VV, VH, COH blocks, y, source_row, block, fold roles; and pair keys."""
import sys, os, json, time
import numpy as np
SCRIPTS, SITE = sys.argv[1], sys.argv[2]
sys.path.insert(0, SCRIPTS)
from a1_s6_experiments import Year      # noqa: E402
out = {}; meta = {}
for yr in (2024, 2025):
    Y = Year(yr); ps = Y.parcel_set('primary')
    out[f'{yr}_y'] = ps['y'].values; out[f'{yr}_row'] = ps.index.values; out[f'{yr}_block'] = ps['block_5km'].astype(str).values
    out[f'{yr}_folds'] = np.stack([ps[f'fold_{k}_role'].astype(str).values for k in range(6)], 1)
    for trk in (37, 88):
        for M in (2, 9):
            keys = Y.pairs(trk, M)
            if not keys: continue
            dates = sorted({d for k in keys for d in k.split('_')})
            for lay, cols in (('VV', dates), ('VH', dates), ('COH', keys)):
                v = Y.tab[(trk, lay)].reindex(index=ps.index, columns=cols).values.astype(np.float32)
                out[f'{yr}_T{trk}_M{M}_{lay}'] = 10 * np.log10(v) if lay != 'COH' else v
            meta[f'{yr}_T{trk}_M{M}'] = dict(keys=keys, dates=dates)
    print(SITE, yr, len(ps), flush=True)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dump_{SITE}.npz')
np.savez_compressed(p, **out)
json.dump(dict(meta=meta, scripts=SCRIPTS, run_at=time.strftime('%Y-%m-%d %H:%M:%S')), open(p.replace('.npz', '_meta.json'), 'w'), indent=1)
