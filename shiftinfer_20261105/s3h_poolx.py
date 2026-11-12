"""S3h amendment (EXPLORE_S3D_Q1_ZH.md, 2026-11-11 12:10): pooled training without the transfer's source and target seasons.
cpu : hgb_pool_excl, hgb_poolsh_excl (archive seasons moved by the frozen MAP-EM with the archive rebuilt without those seasons)
deep: tcn_pool_excl, tcn_poolsh_excl (db_run.train unchanged; same convention as s3h_deep.py)
Usage: python s3h_poolx.py <cpu|deep> <grp> [part n] -> s3h_poolx_<kind>_<grp>[_p<part>].csv (resumable)"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import s3h_lib as L
F = L.F
KIND, G = sys.argv[1], sys.argv[2]; PART, NP = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
if KIND == 'deep':
    DB = os.path.join(L.HERE, '..', 'multiyear_20261017', 'deep_baseline_2023'); sys.argv = [os.path.join(DB, 'db_run.py'), 'B88']; sys.path.insert(0, DB)
    import db_run as D
OUT = os.path.join(L.HERE, f"s3h_poolx_{KIND}_{G}{'' if NP == 1 else f'_p{PART}'}.csv")
V = ['hgb_pool_excl', 'hgb_poolsh_excl'] if KIND == 'cpu' else ['tcn_pool_excl', 'tcn_poolsh_excl']
tasks = L.tasks_for(G); todo = L.transfers(G)[PART::NP]
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
    AX = F.Archive.build([t for t in tasks if not ({int(t['id'][1]), int(t['id'][2])} & {ya, yb})])
    PS = [p for p in L.pool_seasons(G, S) if p['y'] not in (ya, yb)]; YP = np.concatenate([p['Y'] for p in PS])
    if KIND == 'cpu': XP = np.vstack([L.assemble(*L.split(p['E'])) for p in PS]); XB = L.assemble(*L.split(Eb))
    else: XP = np.concatenate([L.to_seq(p['E']) for p in PS]); Xb = L.to_seq(Eb_all)
    pred = {v: np.full(len(Yb), -1) for v in V}
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~L.xydrop(G, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        mu, prec, ld, cls = F.gauss_fit(Ea[tr], Ya[tr]); m, P = AX.prior(cls, mu, tk, doy); _, d = F.map_em(Eb[te], mu, prec, ld, cls, m, P)
        mv = [L.shift_season(p, Eb[te], AX)[0] for p in PS]
        if KIND == 'cpu':
            pred['hgb_pool_excl'][te] = L.hgb().fit(np.vstack([L.assemble(*L.split(Ea[tr])), XP]), np.r_[Ya[tr], YP]).predict(XB[te])
            Xs = L.assemble(*L.split(F.resynthesize(Ea[tr], Ya[tr], d)))
            pred['hgb_poolsh_excl'][te] = L.hgb().fit(np.vstack([Xs] + [L.assemble(*L.split(E)) for E in mv]), np.r_[Ya[tr], YP]).predict(XB[te])
        else:
            pred['tcn_pool_excl'][te] = tcn(np.concatenate([L.to_seq(Ea_all[tr]), XP]), np.r_[Ya[tr], YP], Xb[te], k)
            Xs = L.to_seq(F.resynthesize(Ea_all[tr], Ya[tr], d))
            pred['tcn_poolsh_excl'][te] = tcn(np.concatenate([Xs] + [L.to_seq(E) for E in mv]), np.r_[Ya[tr], YP], Xb[te], k)
    ok = pred[V[0]] >= 0
    row = dict(group=G, src=ya, tgt=yb, n_pool_rows=len(YP), n_pool_seasons=len(PS), **{v: balanced_accuracy_score(Yb[ok], p[ok]) for v, p in pred.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(G, ya, '->', yb, {v: round(row[v], 3) for v in V}, round(time.time() - t0), 's', flush=True)
print('DONE', KIND, G, PART, NP)
