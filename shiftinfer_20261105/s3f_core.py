"""S3f (EXPLORE_S3D_Q1_ZH.md, section S3f): shift-estimation accuracy, per-class predictions and parameter sensitivity of the frozen method.
Pipeline identical to frozen/build_and_verify.py (p1_tasks data, impute, gauss_fit, leave-site-out archive, map_em); transfers todo[0::2].
Usage: python s3f_core.py main <B88|A37|A88>        -> s3f_shift_<grp>.csv.gz, s3f_pred_<grp>.csv.gz, s3f_sens_<grp>.csv (config 'frozen')
       python s3f_core.py sens <grp> <b|lam|nit> <v>  -> appends config '<p>=<v>' to s3f_sens_<grp>.csv (proposed only)"""
import os, sys, itertools, pickle, time
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'frozen')); sys.path.insert(0, HERE)
import hybridD_lw as F
from p1_tasks import load, data, xydrop, dim_meta, DEV
from p9_lib import dim_dates, gen_em2
MODE, GRP = sys.argv[1], sys.argv[2]
PAR, VAL = (sys.argv[3], float(sys.argv[4])) if MODE == 'sens' else (None, None)
OTHER = 'A' if GRP[0] == 'B' else 'B'
if PAR == 'b':                       # rebuild the archive with the same tasks under attenuation b (sigma_a^2 consistent with the basis)
    orig = F.basis
    F.basis = lambda cls, mu, ty, sl, sub, doy, b=None: orig(cls, mu, ty, sl, sub, doy, b=VAL)
    caches = ['A37', 'A88'] if OTHER == 'A' else ['B88']
    tasks = sum((pickle.load(open(os.path.join(HERE, f'p9_arch_{g}.pkl'), 'rb')) for g in caches), [])
    ar = F.Archive.build(tasks)
else:
    ar = F.Archive.load(os.path.join(HERE, 'frozen', f'archive_NL_{OTHER}.npz'))
if PAR == 'nit': F.NIT = int(VAL)
LAM = VAL if PAR == 'lam' else F.LAM
CFG = 'frozen' if MODE == 'main' else f'{PAR}={VAL:g}'


def raw_pred(X, mu, prec, logdet, cls):
    L = np.stack([-0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', X - mu[c], prec[c], X - mu[c]) for c in cls], 1)
    return np.array(cls)[L.argmax(1)]


site, trk, Z, M = load(GRP); years = [y for y in DEV if f'{y}_T{trk}' in M]; pairs = list(itertools.permutations(years, 2))[0::2]
SENS = os.path.join(HERE, f's3f_sens_{GRP}.csv')
done = set()
if os.path.exists(SENS):
    s = pd.read_csv(SENS); done = {(r.src, r.tgt) for r in s[s.config == CFG].itertuples()}
shift_rows, pred_rows, t0 = [], [], time.time()
for ya, yb in pairs:
    if (ya, yb) in done: continue
    S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
    vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
    Ea = F.impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = F.impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
    ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); doy = np.array(dim_dates(M, trk, ya, S, ty, sl, sub).dayofyear)
    P = {v: np.full(len(Yb), -1) for v in (('raw', 'em', 'lw') if MODE == 'main' else ('lw',))}; fold = np.full(len(Yb), -1)
    for k in range(6):
        trm = (Fa[:, k] == 'TRAIN') & ~xydrop(GRP, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
        if not tem.sum(): continue
        mu, prec, logdet, cls = F.gauss_fit(Ea[trm], Ya[trm]); m, Psi = ar.prior(cls, mu, tk, doy)
        P['lw'][tem], d_lw = F.map_em(Eb[tem], mu, prec, logdet, cls, m, Psi, lam=LAM); fold[tem] = k
        if MODE != 'main': continue
        P['raw'][tem] = raw_pred(Eb[tem], mu, prec, logdet, cls)
        P['em'][tem], d_em = gen_em2(Eb[tem], mu, prec, logdet, cls, None, 0.1)
        for c in cls:
            mt = tem & (Yb == c); ns = int((Ya[trm] == c).sum()); nt = int(mt.sum())
            if nt < 20 or ns < 20: continue
            tru = Eb[mt].mean(0) - mu[c]
            for d in range(len(tk)):
                shift_rows.append((GRP, ya, yb, k, c, ns, nt, d, int(ty[d]), int(sl[d]), int(sub[d]), tru[d], d_em[c][d], d_lw[c][d]))
    ok = P['lw'] >= 0
    row = dict(group=GRP, src=ya, tgt=yb, config=CFG, **{f'ba_{v}': balanced_accuracy_score(Yb[ok], p[ok]) for v, p in P.items()})
    pd.DataFrame([row]).to_csv(SENS, mode='a', header=not os.path.exists(SENS), index=False)
    if MODE == 'main':
        pred_rows.append(pd.DataFrame(dict(group=GRP, src=ya, tgt=yb, fold=fold[ok], y=Yb[ok], p_raw=P['raw'][ok], p_em=P['em'][ok], p_lw=P['lw'][ok])))
    print(GRP, CFG, ya, '->', yb, {k: round(v, 4) for k, v in row.items() if k.startswith('ba')}, round(time.time() - t0), 's', flush=True)
if MODE == 'main' and shift_rows:
    pd.DataFrame(shift_rows, columns=['group', 'src', 'tgt', 'fold', 'cls', 'n_src', 'n_tgt', 'dim', 'ty', 'slot', 'sub', 'true', 'em', 'lw']).to_csv(
        os.path.join(HERE, f's3f_shift_{GRP}.csv.gz'), index=False)
    pd.concat(pred_rows).to_csv(os.path.join(HERE, f's3f_pred_{GRP}.csv.gz'), index=False)
print('DONE', GRP, CFG)
