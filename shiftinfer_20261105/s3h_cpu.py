"""S3h CPU part (EXPLORE_S3D_Q1_ZH.md, S3h): Gaussian-classifier ablations / archive without the transfer's seasons, and HGB with
the same-archive pooled training. Usage: python s3h_cpu.py <B88|A37|A88|FR> [part n] [--smoke]  -> s3h_cpu_<grp>[_p<part>].csv (resumable)"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
import s3h_lib as L
F = L.F
ARGS = [a for a in sys.argv[1:] if a != '--smoke']; SMOKE = '--smoke' in sys.argv
G = ARGS[0]; PART, NP = (int(ARGS[1]), int(ARGS[2])) if len(ARGS) > 2 else (0, 1)
OUT = os.path.join(L.HERE, f"s3h_cpu_{G}{'' if NP == 1 else f'_p{PART}'}{'_smoke' if SMOKE else ''}.csv")
GV = ['g_em', 'g_lw', 'g_lw_excl', 'g_notau', 'g_nodate', 'g_shuffle']; HV = ['hgb_raw', 'hgb_lw', 'hgb_pool', 'hgb_poolsh']

tasks = L.tasks_for(G); print('archive tasks', len(tasks), 'full-mode max diff vs frozen', L.check_full(G, tasks), flush=True)
AR = L.frozen_archive(G); CTRL = {m: L.build_control(tasks, m) for m in ('no_tau', 'no_date')}
todo = L.transfers(G)[PART::NP]
if SMOKE: todo = todo[:1]
done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set(); t0 = time.time()
for ya, yb in todo:
    if (ya, yb) in done: continue
    S = L.common(G, ya, yb)
    Ea_all, Ya, Fa, _ = L.extract(G, ya, S); Eb_all, Yb, Fb, _ = L.extract(G, yb, S)
    Ea = F.impute(Ea_all, Ea_all); Eb = F.impute(Eb_all, Ea_all); tk, doy = L.layout(G, ya, S); XB = L.assemble(*L.split(Eb))
    ex = [t for t in tasks if not ({int(t['id'][1]), int(t['id'][2])} & {ya, yb})]; AX = F.Archive.build(ex)
    PS = L.pool_seasons(G, S); XP = np.vstack([L.assemble(*L.split(p['E'])) for p in PS]); YP = np.concatenate([p['Y'] for p in PS])
    pred = {v: np.full(len(Yb), -1) for v in GV + HV}; n_moved = 0; tf = time.time()
    for k in range(6):
        tr = (Fa[:, k] == 'TRAIN') & ~L.xydrop(G, ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        if not te.sum(): continue
        if SMOKE and k > 0: break
        Es, ys, Et = Ea[tr], Ya[tr], Eb[te]; mu, prec, ld, cls = F.gauss_fit(Es, ys)
        pred['g_em'][te] = gen = L.gen_em2(Et, mu, prec, ld, cls, None, 0.1)[0]
        m, P = AR.prior(cls, mu, tk, doy); pred['g_lw'][te], d_lw = F.map_em(Et, mu, prec, ld, cls, m, P)
        m, P = AX.prior(cls, mu, tk, doy); pred['g_lw_excl'][te] = F.map_em(Et, mu, prec, ld, cls, m, P)[0]
        for v, mode in (('g_notau', 'no_tau'), ('g_nodate', 'no_date')):
            m, P = L.control_prior(CTRL[mode], mode, cls, mu, tk, doy); pred[v][te] = F.map_em(Et, mu, prec, ld, cls, m, P)[0]
        m, P = L.shuffle_prior(AR, cls, mu, tk, doy, 20261111 + 1000 * (ya - 2000) + 10 * (yb - 2000) + k); pred['g_shuffle'][te] = F.map_em(Et, mu, prec, ld, cls, m, P)[0]
        Xs_lw = L.assemble(*L.split(F.resynthesize(Es, ys, d_lw)))
        pred['hgb_raw'][te] = L.hgb().fit(L.assemble(*L.split(Es)), ys).predict(XB[te])
        pred['hgb_lw'][te] = L.hgb().fit(Xs_lw, ys).predict(XB[te])
        pred['hgb_pool'][te] = L.hgb().fit(np.vstack([L.assemble(*L.split(Es)), XP]), np.r_[ys, YP]).predict(XB[te])
        mv = [L.shift_season(p, Et, AR) for p in PS]; n_moved = sum(f for _, f in mv)
        XPS = np.vstack([L.assemble(*L.split(E)) for E, _ in mv])
        pred['hgb_poolsh'][te] = L.hgb().fit(np.vstack([Xs_lw, XPS]), np.r_[ys, YP]).predict(XB[te])
        if SMOKE: print('fold', k, 'seconds', round(time.time() - tf, 1), flush=True)
    ok = pred['g_lw'] >= 0
    row = dict(group=G, src=ya, tgt=yb, n_slots=len(S), n_pool_rows=len(YP), n_pool_seasons=len(PS), n_pool_moved=n_moved,
               n_arch_tasks=len(tasks), n_excl_tasks=len(ex), **{v: balanced_accuracy_score(Yb[ok], p[ok]) for v, p in pred.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(G, ya, '->', yb, {v: round(row[v], 3) for v in GV + HV}, round(time.time() - t0), 's', flush=True)
print('DONE', G, PART, NP)
