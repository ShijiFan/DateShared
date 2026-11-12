"""EXPLORE_P9 archive cache: per (group, src, tgt, fold) the source TRAIN class means (gauss_fit, imputed, xydrop as in P1),
task dims, source-year doy per dim and the oracle per-class shift (same definition as p1_tasks; NaN when < 20 parcels).
Usage: python p9_archive.py <grp> -> p9_arch_<grp>.pkl"""
import os, sys, itertools, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, DEV
from p9_lib import dim_dates
HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    G = sys.argv[1]; site, trk, Z, M = load(G); out = []; t0 = time.time()
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    for ya, yb in itertools.permutations(years, 2):
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]
        doy = np.array(dim_dates(M, trk, ya, S, ty, sl, sub).dayofyear)
        for k in range(6):
            tr = (Fa[:, k] == 'TRAIN') & ~xydrop(G, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
            if not te.sum(): continue
            ys_, yt = Ya[tr], Yb[te]; Et = Eb[te]
            mu, prec, logdet, cls = gauss_fit(Ea[tr], ys_)
            orc = {c: (Et[yt == c].mean(0) - mu[c]) if ((yt == c).sum() >= 20 and (ys_ == c).sum() >= 20) else np.full(D, np.nan) for c in cls}
            out.append(dict(id=(G, ya, yb, k), cls=list(cls), tk=tk, mu={c: mu[c] for c in cls}, ty=ty, sl=sl, sub=sub, doy=doy, orc=orc))
    pickle.dump(out, open(os.path.join(HERE, f'p9_arch_{G}.pkl'), 'wb'))
    print('DONE', G, len(out), round(time.time() - t0), 's')
