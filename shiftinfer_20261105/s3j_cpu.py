"""S3j (EXPLORE_S3D_Q1_ZH.md, written before running): Dutch single-source results with the archive WITHOUT the transfer's seasons.
Template p11_c2.py / p10_lw.py; only the archive changes: for transfer (ya, yb) the priors are rebuilt from archive tasks whose two years
are both outside {ya, yb}.  Columns (suffix _x): hpj_cov01_x, hybridD_r*_x, hybridD_lw_x (Gaussian, tab_nl), hgb_lw_x (tab_plugin),
rf_lw_x (even transfers only, as P11 phase 2).
Usage: python s3j_cpu.py <A37|A88|B88> [--check]   (--check: full archive, first transfer only -> s3j_cpu_<grp>_check.csv)"""
import os, sys, itertools, pickle, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'frozen'))
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, DEV, H
from p9_lib import gen_em2, dim_dates
from p9_basis import build_priors, GROUPS
from p10_lw import add_lw, priors3
import hybridD_lw as F
hgb, assemble = H['hgb'], H['assemble']
rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt', random_state=20260923, n_jobs=4)
split = lambda E, nd: (E[:, :nd], E[:, nd:2 * nd], E[:, 2 * nd:])
GV = ['hpj_cov01', 'hybridD_r*', 'hybridD_lw']

TG = sys.argv[1]; CHECK = '--check' in sys.argv; t0 = time.time(); trs = 'B' if TG[0] == 'A' else 'A'
allr = pd.concat([pd.read_parquet(os.path.join(HERE, f'p1_rows_{g}.parquet')) for g in GROUPS], ignore_index=True); tr = allr[allr.site == trs]
arch = sum([pickle.load(open(os.path.join(HERE, f'p9_arch_{g}.pkl'), 'rb')) for g in GROUPS if g[0] == trs], [])
ty_all = sorted(set(tr.src))
OUT = os.path.join(HERE, f"s3j_cpu_{TG}{'_check' if CHECK else ''}.csv")


def priors_without(ex):
    ys = [y for y in ty_all if y not in ex]
    t_ = tr[tr.src.isin(ys) & tr.tgt.isin(ys)]; a_ = [a for a in arch if a['id'][1] in ys and a['id'][2] in ys]
    return add_lw(build_priors(t_, a_, with_fixed=False), t_, a_), len(a_)


site, trk, Z, M = load(TG)
years = [y for y in DEV if f'{y}_T{trk}' in M]; perm = list(itertools.permutations(years, 2)); even = set(perm[0::2])
pairs = sorted({tuple(sorted(p)) for p in perm})                       # both directions of a pair share one archive
todo = [p for q in pairs for p in (q, q[::-1])]
if CHECK: todo = perm[:1]
done = {(int(r.src), int(r.tgt)) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set()
cur = (None, None)
for ya, yb in todo:
    if (ya, yb) in done: continue
    ex = set() if CHECK else {ya, yb}
    if cur[0] != frozenset(ex): cur = (frozenset(ex), priors_without(ex))
    P, n_arch = cur[1]
    S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
    vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
    Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
    nd = vva.shape[1]; ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]; doy = np.array(dim_dates(M, trk, ya, S, ty, sl, sub).dayofyear)
    XB = assemble(*split(Eb, nd)); isev = (ya, yb) in even
    V = [g + '_x' for g in GV] + ['hgb_lw_x'] + (['rf_lw_x'] if isev else [])
    pred = {v: np.full(len(Yb), -1) for v in V}
    for k in range(6):
        trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
        if not tem.sum(): continue
        Es, ys_ = Ea[trm], Ya[trm]; Et = Eb[tem]; mu, prec, logdet, cls = gauss_fit(Es, ys_); C = len(cls)
        PR = priors3(P, cls, mu, tk, ty, sl, sub, doy)
        for g in GV:                                                     # as p10_lw.run_task
            m, Psi, Pi = PR[g]
            r = gen_em2(Et, mu, prec, logdet, cls, (m, Psi if Psi is not None else np.zeros((C * D, C * D))), 0.1, Pinv=Pi)
            pred[g + '_x'][tem] = r[0]
        d_lw = gen_em2(Et, mu, prec, logdet, cls, (PR['hybridD_lw'][0], PR['hybridD_lw'][1]), 0.1)[1]      # as p11_c2
        Xs = assemble(*split(F.resynthesize(Es, ys_, d_lw), nd))
        pred['hgb_lw_x'][tem] = hgb().fit(Xs, ys_).predict(XB[tem])
        if isev: pred['rf_lw_x'][tem] = rf().fit(Xs, ys_).predict(XB[tem])
    ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
    row = dict(group=TG, src=ya, tgt=yb, n_arch_tasks=n_arch, **{v: ba.get(v, np.nan) for v in [g + '_x' for g in GV] + ['hgb_lw_x', 'rf_lw_x']})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(TG, ya, '->', yb, {v: round(x, 4) for v, x in ba.items()}, round(time.time() - t0), 's', flush=True)
print('DONE', TG)
