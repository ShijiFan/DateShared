"""EXPLORE_P9: shift-basis priors (physics modes + archive-learned modes, incl. toy-B discovered water-cloud date basis).
All variants use the lam=0.1 covariance update. Prior from the other site (leave-site-out), as P3-P8.
Usage: python p9_basis.py <test group> -> p9_basis_<grp>.csv, p9_delta_<grp>.csv, p9_coef_<grp>.csv, p9_modes_train<site>.npz"""
import os, sys, itertools, pickle, time, json
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, DEV
from p3_joint import build_joint, joint_for
from p9_lib import (gen_em2, woodbury, dim_dates, hand_B, disc_B, mllr_B, api_at, season_api, fit_mw, mw_mean,
                    keyspace, key_type, ppca, map_lowrank, phys_fit, NOM)
HERE = os.path.dirname(os.path.abspath(__file__)); GROUPS = ('A37', 'A88', 'B88')
RFIX = (2, 4, 8)


def build_priors(tr_rows, arch, with_fixed=True):
    """Archive-level objects for one training archive (rows of the training site, archive cache list)."""
    keys, jm, jc = build_joint(tr_rows)
    _, V, ids, full = keyspace(tr_rows); kt = key_type(keys); Vf = V[full]; idf = [ids[i] for i in np.where(full)[0]]
    tot = np.array([Vf.var(0, ddof=1)[kt == t].mean() for t in range(3)])
    P = dict(keys=keys, jm=jm, jc=jc, tot=tot)
    P['learn'] = {'rstar': ppca(Vf, kt)}
    var_h, e0_h, RV_h = phys_fit(arch, keys, jm, jc, 'hand'); var_d, e0_d, RV_d = phys_fit(arch, keys, jm, jc, 'disc')
    Rh = np.array([RV_h[i] for i in idf]); Rd = np.array([RV_d[i] for i in idf])
    P['hand'] = dict(var=var_h, e0=e0_h, rstar=ppca(Rh, kt)); P['disc'] = dict(var=var_d, e0=e0_d, rstar=ppca(Rd, kt))
    if with_fixed:
        for r in RFIX:
            P['learn'][r] = ppca(Vf, kt, r); P['hand'][r] = ppca(Rh, kt, r)
    P['n_complete'] = int(full.sum()); P['n_arch'] = len(arch)
    return P


def task_priors(P, names, cls, mu, tk, ty, sl, sub, doy, m_mw=None):
    """returns {variant: (m, Psi or None, Pinv or None)}; Psi only for hpj (exact P5 path)."""
    C, D = len(cls), len(tk); m0 = np.zeros(C * D); out = {}
    m_arch, Psi_lw = joint_for(P['keys'], P['jm'], P['jc'], cls, tk)
    Bh, kh = hand_B(cls, mu, ty, doy); Bd, kd = disc_B(cls, mu, ty, sl, sub, doy)
    ety = lambda e3: np.tile(np.asarray(e3)[ty], C)
    sv = lambda kinds, var: np.array([var[k] for k in kinds])
    for n in names:
        if n == 'hpj_cov01': out[n] = (m_arch, Psi_lw, None)
        elif n == 'mllr_cov01':
            W = mllr_B(cls, mu, D); out[n] = (m0, None, woodbury(W, np.full(W.shape[1], 100.0), np.full(C * D, 0.05 ** 2)))
        elif n == 'phys_nom':
            s = np.array([NOM['L'][k[1]] ** 2 if k[0] == 'L' else NOM['tau'] ** 2 for k in kh])
            out[n] = (m0, None, woodbury(Bh, s, ety([v ** 2 for v in NOM['e']])))
        elif n == 'phys_arch': out[n] = (m_arch, None, woodbury(Bh, sv(kh, P['hand']['var']), ety(P['hand']['e0'])))
        elif n == 'physD_arch': out[n] = (m_arch, None, woodbury(Bd, sv(kd, P['disc']['var']), ety(P['disc']['e0'])))
        elif n.startswith('learn_r'):
            U, e, r, _ = P['learn']['rstar' if n == 'learn_r*' else int(n[7:])]
            Ut, et = map_lowrank(P['keys'], U, e, P['tot'], cls, tk); out[n] = (m_arch, None, woodbury(Ut, np.ones(Ut.shape[1]), et))
        elif n.startswith('hybridD_r*'):
            U, e, r, _ = P['disc']['rstar']; Ut, et = map_lowrank(P['keys'], U, e, P['disc']['e0'], cls, tk)
            Pi = woodbury(np.hstack([Bd, Ut]), np.concatenate([sv(kd, P['disc']['var']), np.ones(Ut.shape[1])]), et)
            out[n] = (m_mw if n.endswith('_mw') else m_arch, None, Pi)
        elif n.startswith('hybrid_r'):
            U, e, r, _ = P['hand']['rstar' if n == 'hybrid_r*' else int(n[8:])]; Ut, et = map_lowrank(P['keys'], U, e, P['hand']['e0'], cls, tk)
            out[n] = (m_arch, None, woodbury(np.hstack([Bh, Ut]), np.concatenate([sv(kh, P['hand']['var']), np.ones(Ut.shape[1])]), et))
    return out


VARS = ['em_cov01', 'hpj_cov01', 'mllr_cov01', 'phys_nom', 'phys_arch', 'learn_r*', 'hybrid_r*', 'physD_arch', 'hybridD_r*', 'hybridD_r*_mw'] + \
       [f'learn_r{r}' for r in RFIX] + [f'hybrid_r{r}' for r in RFIX]
DIAG = ['em_cov01', 'hpj_cov01', 'phys_nom', 'hybrid_r*', 'hybridD_r*']


if __name__ == '__main__':
    TG = sys.argv[1]; t0 = time.time(); trs = 'B' if TG[0] == 'A' else 'A'
    allr = pd.concat([pd.read_parquet(os.path.join(HERE, f'p1_rows_{g}.parquet')) for g in GROUPS], ignore_index=True)
    arch = sum([pickle.load(open(os.path.join(HERE, f'p9_arch_{g}.pkl'), 'rb')) for g in GROUPS if g[0] == trs], [])
    P = build_priors(allr[allr.site == trs], arch)
    st = pd.read_csv(os.path.join(HERE, 'toy_B', 'shift_table.csv'))
    for nm, ty_ in (('vh_s', 1),):
        L = st[st.dtype == ty_][['group', 'src', 'cls', 'slot', 'sub', 'mu_s']].drop_duplicates(['group', 'src', 'cls', 'slot', 'sub']).rename(columns={'mu_s': nm})
        st = st.merge(L, on=['group', 'src', 'cls', 'slot', 'sub'], how='left')
    st['vh_s'] = st.groupby(['group', 'src', 'cls', 'slot'])['vh_s'].transform(lambda s: s.fillna(s.mean()))
    thw = fit_mw(st, trs)
    info = dict(train_site=trs, n_arch=P['n_arch'], n_complete=P['n_complete'], mw_theta=thw.round(4).tolist(),
                r_learn=P['learn']['rstar'][2], r_hybrid=P['hand']['rstar'][2], r_hybridD=P['disc']['rstar'][2],
                var_hand={str(k): v for k, v in P['hand']['var'].items()}, var_disc={str(k): v for k, v in P['disc']['var'].items()},
                e0_hand=P['hand']['e0'].tolist(), e0_disc=P['disc']['e0'].tolist())
    print(json.dumps(info), flush=True)
    np.savez(os.path.join(HERE, f'p9_modes_train{trs}.npz'), U_learn=P['learn']['rstar'][0], evr_learn=P['learn']['rstar'][3],
             U_hand=P['hand']['rstar'][0], evr_hand=P['hand']['rstar'][3], U_disc=P['disc']['rstar'][0], evr_disc=P['disc']['rstar'][3],
             keys=np.array(P['keys']), info=json.dumps(info))
    site, trk, Z, M = load(TG); out, dg, cf = [], [], []; checked = False
    years = [y for y in DEV if f'{y}_T{trk}' in M]
    for ya, yb in itertools.permutations(years, 2):
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]
        ds, dt_ = dim_dates(M, trk, ya, S, ty, sl, sub), dim_dates(M, trk, yb, S, ty, sl, sub); doy = np.array(ds.dayofyear)
        aps, apt = api_at(site, ds), api_at(site, dt_); sapi = season_api(site, yb) - season_api(site, ya)
        pred = {v: np.full(len(Yb), -1) for v in VARS}
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            ys_, yt = Ya[trm], Yb[tem]; mu, prec, logdet, cls = gauss_fit(Ea[trm], ys_); Et = Eb[tem]; C = len(cls)
            m_mw = mw_mean(thw, cls, mu, ty, sl, sub, aps, apt)
            PR = task_priors(P, [v for v in VARS if v != 'em_cov01'], cls, mu, tk, ty, sl, sub, doy, m_mw)
            if not checked:     # Woodbury vs direct inverse (EXPLORE_P9 §2)
                Bh, kh = hand_B(cls, mu, ty, doy); U, e, r, _ = P['hand']['rstar']; Ut, et = map_lowrank(P['keys'], U, e, P['hand']['e0'], cls, tk)
                W = np.hstack([Bh, Ut]); s = np.concatenate([[P['hand']['var'][q] for q in kh], np.ones(Ut.shape[1])])
                Psi = W @ np.diag(s) @ W.T + np.diag(et); dI = np.linalg.inv(Psi)
                rel = float(np.abs(woodbury(W, s, et) - dI).max() / np.abs(dI).max()); print('woodbury rel err', rel, flush=True)
                assert rel < 1e-6; checked = True
            dl = {}
            for v in VARS:
                if v == 'em_cov01': lab, dl[v] = gen_em2(Et, mu, prec, logdet, cls, None, 0.1)
                else:
                    m, Psi, Pi = PR[v]; lab, dl[v] = gen_em2(Et, mu, prec, logdet, cls, (m, Psi if Psi is not None else np.zeros((C * D, C * D))), 0.1, Pinv=Pi)
                pred[v][tem] = lab
            for c in cls:
                if (yt == c).sum() < 20 or (ys_ == c).sum() < 20: continue
                o = Et[yt == c].mean(0) - mu[c]; ps = float((ys_ == c).mean())
                for v in DIAG:
                    e_ = dl[v][c] - o
                    dg.append(dict(group=TG, src=ya, tgt=yb, fold=k, cls=c, psrc=ps, var=v, rmse_bs=float(np.sqrt((e_[ty < 2] ** 2).mean())), rmse_coh=float(np.sqrt((e_[ty == 2] ** 2).mean()))))
            # coefficients of the hybridD estimate on the discovered basis (LS projection of delta_hat - m)
            Bd, kd = disc_B(cls, mu, ty, sl, sub, doy); m_arch = PR['hpj_cov01'][0]
            th, *_ = np.linalg.lstsq(Bd, np.concatenate([dl['hybridD_r*'][c] for c in cls]) - m_arch, rcond=None)
            bsd = sorted(set((int(sl[d]), int(sub[d])) for d in range(D) if ty[d] < 2))
            for j, (s_, u_) in enumerate(bsd):
                d0 = np.where((ty == 0) & (sl == s_) & (sub == u_))[0][0]
                cf.append(dict(group=TG, src=ya, tgt=yb, fold=k, kind='wbs', slot=s_, sub=u_, coef=th[j], api_s=aps[d0], api_t=apt[d0], season_dapi=sapi))
            nb = len(bsd) + len(set(int(sl[d]) for d in range(D) if ty[d] == 2))
            for a, c in enumerate(cls): cf.append(dict(group=TG, src=ya, tgt=yb, fold=k, kind='tau', cls=c, coef=th[nb + a], season_dapi=sapi))
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        out.append(dict(group=TG, src=ya, tgt=yb, **ba)); print(TG, ya, '->', yb, {v: round(x, 3) for v, x in ba.items()}, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(out).to_csv(os.path.join(HERE, f'p9_basis_{TG}.csv'), index=False)
        pd.DataFrame(dg).to_csv(os.path.join(HERE, f'p9_delta_{TG}.csv'), index=False); pd.DataFrame(cf).to_csv(os.path.join(HERE, f'p9_coef_{TG}.csv'), index=False)
    print('DONE', TG)
