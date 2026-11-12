"""Diagnostics of the FROZEN method (no method change): for every dev transfer/fold, run frozen hybridD_lw (portable module,
archive = other site), project the estimated shift on its discovered basis (LS) to get per-date shared offsets and per-class timing,
and record per-class shift error vs oracle and per-class recall. Usage: python diag_frozen.py <grp> -> diag_frozen_<grp>.csv / _coef_<grp>.csv"""
import os, sys, itertools, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'frozen'))
import hybridD_lw as F
from p1_tasks import load, data, xydrop, dim_meta, DEV
from p9_lib import dim_dates, api_at

TG = sys.argv[1]; t0 = time.time()
ar = F.Archive.load(os.path.join(HERE, 'frozen', f"archive_NL_{'A' if TG[0] == 'B' else 'B'}.npz"))
site, trk, Z, M = load(TG); years = [y for y in DEV if f'{y}_T{trk}' in M]; dg, cf = [], []
for ya, yb in itertools.permutations(years, 2):
    S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
    vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
    Ea = F.impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = F.impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
    ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]
    ds, dt_ = dim_dates(M, trk, ya, S, ty, sl, sub), dim_dates(M, trk, yb, S, ty, sl, sub); doy = np.array(ds.dayofyear)
    aps, apt = api_at(site, ds), api_at(site, dt_)
    for k in range(6):
        trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
        if not tem.sum(): continue
        Es, ys_, Et, yt = Ea[trm], Ya[trm], Eb[tem], Yb[tem]
        mu, prec, logdet, cls = F.gauss_fit(Es, ys_); m, Psi = ar.prior(cls, mu, tk, doy)
        lab, dl = F.map_em(Et, mu, prec, logdet, cls, m, Psi)
        for c in cls:
            mt = yt == c; ps = float((ys_ == c).mean())
            rec = float((lab[mt] == c).mean()) if mt.sum() else np.nan
            ok = mt.sum() >= 20 and (ys_ == c).sum() >= 20; e = (dl[c] - (Et[mt].mean(0) - mu[c])) if ok else np.full(D, np.nan)
            dg.append(dict(group=TG, src=ya, tgt=yb, fold=k, cls=c, psrc=ps, recall=rec, rmse_bs=float(np.sqrt(np.mean(e[ty < 2] ** 2))) if ok else np.nan))
        B, kinds = F.basis(cls, mu, ty, sl, sub, doy); th, *_ = np.linalg.lstsq(B, np.concatenate([dl[c] for c in cls]) - m, rcond=None)
        bsd = sorted(set((int(sl[d]), int(sub[d])) for d in range(D) if ty[d] < 2))
        for j, (s_, u_) in enumerate(bsd):
            d0 = np.where((ty == 0) & (sl == s_) & (sub == u_))[0][0]
            cf.append(dict(group=TG, src=ya, tgt=yb, fold=k, kind='wbs', slot=s_, sub=u_, coef=th[j], api_s=aps[d0], api_t=apt[d0]))
        nb = kinds.index('tau')
        for a, c in enumerate(cls): cf.append(dict(group=TG, src=ya, tgt=yb, fold=k, kind='tau', cls=c, coef=th[nb + a]))
    print(TG, ya, '->', yb, round(time.time() - t0), 's', flush=True)
    pd.DataFrame(dg).to_csv(os.path.join(HERE, f'diag_frozen_{TG}.csv'), index=False); pd.DataFrame(cf).to_csv(os.path.join(HERE, f'diag_frozen_coef_{TG}.csv'), index=False)
print('DONE', TG)
