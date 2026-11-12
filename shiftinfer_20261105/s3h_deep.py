"""S3h GPU part (EXPLORE_S3D_Q1_ZH.md, S3h): TempCNN (db_run.train unchanged) trained on the resynthesised source (tcn_lw, p11d / fr_deep
convention, recomputed for an internal pairing), on the same-archive pool (tcn_pool), and on the pool with every season moved to the
target by the frozen MAP-EM (tcn_poolsh). Usage: python s3h_deep.py <B88|A37|A88|FR> [part n] [--smoke] -> s3h_deep_<grp>[_p<part>].csv"""
import os, sys, time
import numpy as np, pandas as pd
import s3h_lib as L
F = L.F
DB = os.path.join(L.HERE, '..', 'multiyear_20261017', 'deep_baseline_2023')
ARGS = [a for a in sys.argv[1:] if a != '--smoke']; SMOKE = '--smoke' in sys.argv
G = ARGS[0]; PART, NP = (int(ARGS[1]), int(ARGS[2])) if len(ARGS) > 2 else (0, 1)
sys.argv = [os.path.join(DB, 'db_run.py'), 'B88']; sys.path.insert(0, DB)
import db_run as D                                  # functions only (train, predict, bal_acc, SEED)
OUT = os.path.join(L.HERE, f"s3h_deep_{G}{'' if NP == 1 else f'_p{PART}'}{'_smoke' if SMOKE else ''}.csv")
V = ['tcn_lw', 'tcn_pool', 'tcn_poolsh']
AR = L.frozen_archive(G)
todo = L.transfers(G)[PART::NP]
if SMOKE: todo = todo[:1]
done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set(); t0 = time.time()


def fit_predict(Xtr, ytr, Xte, k, nanaware):
    if nanaware:
        mu_ = np.nanmean(Xtr, axis=(0, 2), keepdims=True); sd_ = np.nanstd(Xtr, axis=(0, 2), keepdims=True) + 1e-6
    else:
        mu_ = Xtr.mean(axis=(0, 2), keepdims=True); sd_ = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    norm = lambda X: np.nan_to_num((X - mu_) / sd_).astype(np.float32)
    net, XT = D.train(norm(Xtr), ytr, norm(Xte), False, D.SEED + k); return D.predict(net, XT)


for ya, yb in todo:
    if (ya, yb) in done: continue
    S = L.common(G, ya, yb)
    Ea_all, Ya, Fa, _ = L.extract(G, ya, S); Eb_all, Yb, Fb, _ = L.extract(G, yb, S)
    Ea = F.impute(Ea_all, Ea_all); Eb = F.impute(Eb_all, Ea_all); tk, doy = L.layout(G, ya, S)
    Xb = L.to_seq(Eb_all); PS = L.pool_seasons(G, S); XP = np.concatenate([L.to_seq(p['E']) for p in PS]); YP = np.concatenate([p['Y'] for p in PS])
    pred = {v: np.full(len(Yb), -1) for v in V}; tf = time.time()
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~L.xydrop(G, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        if SMOKE and k > 0: break
        mu, prec, ld, cls = F.gauss_fit(Ea[tr], Ya[tr]); m, P = AR.prior(cls, mu, tk, doy)
        _, d_lw = F.map_em(Eb[te], mu, prec, ld, cls, m, P)
        Xs_p11d = L.to_seq(F.resynthesize(np.nan_to_num(Ea_all[tr]), Ya[tr], d_lw))          # p11d: NaN -> 0 before the shift
        pred['tcn_lw'][te] = fit_predict(Xs_p11d, Ya[tr], Xb[te], k, False)
        Xs_nan = L.to_seq(F.resynthesize(Ea_all[tr], Ya[tr], d_lw))
        pred['tcn_pool'][te] = fit_predict(np.concatenate([L.to_seq(Ea_all[tr]), XP]), np.r_[Ya[tr], YP], Xb[te], k, True)
        XPS = np.concatenate([L.to_seq(L.shift_season(p, Eb[te], AR)[0]) for p in PS])
        pred['tcn_poolsh'][te] = fit_predict(np.concatenate([Xs_nan, XPS]), np.r_[Ya[tr], YP], Xb[te], k, True)
        if SMOKE: print('fold', k, 'seconds', round(time.time() - tf, 1), flush=True)
    ok = pred['tcn_lw'] >= 0
    row = dict(group=G, src=ya, tgt=yb, n_pool_rows=len(YP), **{v: D.bal_acc(Yb[ok], p[ok]) for v, p in pred.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(G, ya, '->', yb, {v: round(row[v], 3) for v in V}, round(time.time() - t0), 's', flush=True)
print('DONE', G, PART, NP)
