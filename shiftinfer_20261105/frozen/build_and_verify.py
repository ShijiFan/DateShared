"""Build frozen archive weights from the P9 archive caches and verify the portable module against P10 (p10_main_<grp>.csv).
archive_NL_A.npz (A37+A88; used when testing B88), archive_NL_B.npz (B88; used when testing A), archive_NL_all.npz (all NL; new-region use).
Verification: first N_CHECK transfers of every group, BA must equal p10 'k8_all|hybridD_lw'. -> VERIFY.json"""
import os, sys, pickle, json, itertools
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, UP)
import hybridD_lw as F
from p1_tasks import load, data, xydrop, dim_meta, DEV
from p9_lib import dim_dates
N_CHECK = int(sys.argv[1]) if len(sys.argv) > 1 else 6

arch = {g: pickle.load(open(os.path.join(UP, f'p9_arch_{g}.pkl'), 'rb')) for g in ('A37', 'A88', 'B88')}
A = {'A': F.Archive.build(arch['A37'] + arch['A88'], dict(source='NL site A, T37+T88, 2017-2022/2024/2025')),
     'B': F.Archive.build(arch['B88'], dict(source='NL site B, T88, 2017-2022/2024/2025')),
     'all': F.Archive.build(arch['A37'] + arch['A88'] + arch['B88'], dict(source='NL sites A+B, all tracks, 2017-2022/2024/2025'))}
for k, a in A.items():
    a.save(os.path.join(HERE, f'archive_NL_{k}.npz')); print(k, a.meta, {kk: round(v, 4) for kk, v in a.s_a.items()}, flush=True)
res = {}
for TG in ('B88', 'A37', 'A88'):
    ar = F.Archive.load(os.path.join(HERE, f"archive_NL_{'A' if TG[0] == 'B' else 'B'}.npz"))
    ref = pd.read_csv(os.path.join(UP, f'p10_main_{TG}.csv')).set_index(['src', 'tgt'])['k8_all|hybridD_lw']
    site, trk, Z, M = load(TG); years = [y for y in DEV if f'{y}_T{trk}' in M]; diffs = []
    for ya, yb in list(itertools.permutations(years, 2))[:N_CHECK]:
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = F.impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = F.impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); doy = np.array(dim_dates(M, trk, ya, S, ty, sl, sub).dayofyear)
        pred = np.full(len(Yb), -1)
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            pred[tem] = F.adapt(Ea[trm], Ya[trm], Eb[tem], tk, doy, ar)[0]
        ba = balanced_accuracy_score(Yb[pred >= 0], pred[pred >= 0]); diffs.append(abs(ba - ref.loc[(ya, yb)]))
        print(TG, ya, yb, round(ba, 6), round(ref.loc[(ya, yb)], 6), flush=True)
    res[TG] = dict(n=len(diffs), max_abs_diff=float(max(diffs)))
json.dump(dict(check=res, archives={k: dict(meta=a.meta, s_a=a.s_a) for k, a in A.items()}), open(os.path.join(HERE, 'VERIFY.json'), 'w'), indent=1)
print(json.dumps(res))
