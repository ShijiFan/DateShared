"""S3i amendment 2 (EXPLORE_S3D_Q1_ZH.md, 2026-11-12 06:50): TempCNN at K = 2 archive years (same draw/archive as s3i_curve.py).
tcn_pool_k2, tcn_poolsh_k2; construction as s3h_poolx.py deep (db_run.train unchanged, NaN-aware normalisation).
Usage: python s3i_deep.py <grp> [part n] [--smoke] -> s3i_deep_<grp>[_p<part>].csv (resumable)"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import s3h_lib as L
import s3i_lib as I
F = L.F
ARGS = [a for a in sys.argv[1:] if a != '--smoke']; SMOKE = '--smoke' in sys.argv
G = ARGS[0]; PART, NP = (int(ARGS[1]), int(ARGS[2])) if len(ARGS) > 2 else (0, 1)
DB = os.path.join(L.HERE, '..', 'multiyear_20261017', 'deep_baseline_2023'); sys.argv = [os.path.join(DB, 'db_run.py'), 'B88']; sys.path.insert(0, DB)
import db_run as D
OUT = os.path.join(L.HERE, f"s3i_deep_{G}{'' if NP == 1 else f'_p{PART}'}{'_smoke' if SMOKE else ''}.csv")
V = ['tcn_pool_k2', 'tcn_poolsh_k2']
tasks = L.tasks_for(G); todo = L.transfers(G)[PART::NP]
if SMOKE: todo = todo[:1]
done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set(); t0 = time.time()


def tcn(Xtr, ytr, Xte, k):
    mu_ = np.nanmean(Xtr, axis=(0, 2), keepdims=True); sd_ = np.nanstd(Xtr, axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda X: np.nan_to_num((X - mu_) / sd_).astype(np.float32)
    net, XT = D.train(norm(Xtr), ytr, norm(Xte), False, D.SEED + k); return D.predict(net, XT)


for ya, yb in todo:
    if (ya, yb) in done: continue
    S = L.common(G, ya, yb)
    Ea_all, Ya, Fa, _ = L.extract(G, ya, S); Eb_all, Yb, Fb, _ = L.extract(G, yb, S)
    Ea = F.impute(Ea_all, Ea_all); Eb = F.impute(Eb_all, Ea_all); tk, doy = L.layout(G, ya, S)
    s = I.sets(G, ya, yb, S, tasks, [2])[2]; PS, YP, AX = s['PS'], s['YP'], s['AX']
    XP = np.concatenate([L.to_seq(p['E']) for p in PS]); Xb = L.to_seq(Eb_all)
    pred = {v: np.full(len(Yb), -1) for v in V}
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~L.xydrop(G, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        if SMOKE and k > 0: break
        mu, prec, ld, cls = F.gauss_fit(Ea[tr], Ya[tr]); m, P = AX.prior(cls, mu, tk, doy); _, d = F.map_em(Eb[te], mu, prec, ld, cls, m, P)
        mv = [L.shift_season(p, Eb[te], AX)[0] for p in PS]
        pred['tcn_pool_k2'][te] = tcn(np.concatenate([L.to_seq(Ea_all[tr]), XP]), np.r_[Ya[tr], YP], Xb[te], k)
        Xs = L.to_seq(F.resynthesize(Ea_all[tr], Ya[tr], d))
        pred['tcn_poolsh_k2'][te] = tcn(np.concatenate([Xs] + [L.to_seq(E) for E in mv]), np.r_[Ya[tr], YP], Xb[te], k)
    ok = pred[V[0]] >= 0
    row = dict(group=G, src=ya, tgt=yb, years_k2=' '.join(map(str, s['yrs'])), n_pool_rows=len(YP), n_pool_seasons=len(PS),
               **{v: balanced_accuracy_score(Yb[ok], p[ok]) for v, p in pred.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(G, ya, '->', yb, {v: round(row[v], 3) for v in V}, round(time.time() - t0), 's', flush=True)
print('DONE', G, PART, NP)
