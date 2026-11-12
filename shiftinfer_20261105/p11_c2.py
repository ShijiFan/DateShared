"""EXPLORE_P11_C2: classifier-agnostic resynthesis with the FROZEN hybridD_lw shifts (no method change).
Usage: python p11_c2.py <test group> [phase 1|2]
phase 1 (deviation 01:35, EXPLORE_P11 log): HGB raw/em/hpj/lw/oracle on all transfers -> p11_c2_<grp>.csv (resumes; keeps finished rows)
phase 2: RF raw/em/lw + short-archive HGB on todo[0::2] -> p11_c2b_<grp>.csv (skips transfers already complete in p11_c2_<grp>.csv)"""
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

if __name__ == '__main__':
    TG = sys.argv[1]; t0 = time.time(); trs = 'B' if TG[0] == 'A' else 'A'
    allr = pd.concat([pd.read_parquet(os.path.join(HERE, f'p1_rows_{g}.parquet')) for g in GROUPS], ignore_index=True); tr = allr[allr.site == trs]
    arch = sum([pickle.load(open(os.path.join(HERE, f'p9_arch_{g}.pkl'), 'rb')) for g in GROUPS if g[0] == trs], [])
    P8 = add_lw(build_priors(tr, arch, with_fixed=False), tr, arch)
    rng = np.random.default_rng(20261105); ty_all = sorted(set(tr.src))
    sub3 = {f'k3_{i}': sorted(rng.choice(ty_all, 3, replace=False).tolist()) for i in (0, 1)}; print('k3 subsets', sub3, flush=True)
    P3 = {}
    for nm, ys in sub3.items():
        t_ = tr[tr.src.isin(ys) & tr.tgt.isin(ys)]; a_ = [a for a in arch if a['id'][1] in ys and a['id'][2] in ys]
        P3[nm] = add_lw(build_priors(t_, a_, with_fixed=False), t_, a_)
    PH = sys.argv[2] if len(sys.argv) > 2 else '1'
    f1 = os.path.join(HERE, f'p11_c2_{TG}.csv'); fo = f1 if PH == '1' else os.path.join(HERE, f'p11_c2b_{TG}.csv')
    prev = pd.read_csv(f1) if os.path.exists(f1) else pd.DataFrame(columns=['src', 'tgt'])
    full_done = {(r.src, r.tgt) for r in prev.itertuples()} if PH == '1' else {(int(r.src), int(r.tgt)) for _, r in prev.iterrows() if 'rf_lw' in prev and pd.notna(r.get('rf_lw'))}
    out = prev.to_dict('records') if PH == '1' else (pd.read_csv(fo).to_dict('records') if os.path.exists(fo) else [])
    done = full_done | {(int(r['src']), int(r['tgt'])) for r in out}
    site, trk, Z, M = load(TG)
    years = [y for y in DEV if f'{y}_T{trk}' in M]; todo = list(itertools.permutations(years, 2)); even = set(todo[0::2])
    if PH == '2': todo = todo[0::2]
    for ya, yb in todo:
        if (ya, yb) in done: continue
        S = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S)
        Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        nd = vva.shape[1]; ty, sl, sub = dim_meta(S); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]; doy = np.array(dim_dates(M, trk, ya, S, ty, sl, sub).dayofyear)
        XB = assemble(*split(Eb, nd)); isev = (ya, yb) in even and PH == '2'
        V = ['hgb_raw', 'hgb_em', 'hgb_hpj', 'hgb_lw', 'hgb_oracle'] if PH == '1' else ['rf_raw', 'rf_em', 'rf_lw'] + [f'hgb_{m}_{nm}' for nm in P3 for m in ('hpj', 'lw')]
        pred = {v: np.full(len(Yb), -1) for v in V}
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            Es, ys_ = Ea[trm], Ya[trm]; Et, yt = Eb[tem], Yb[tem]; mu, prec, logdet, cls = gauss_fit(Es, ys_); C = len(cls)
            dl = {'raw': None, 'em': gen_em2(Et, mu, prec, logdet, cls, None, 0.1)[1]}
            PR = priors3(P8, cls, mu, tk, ty, sl, sub, doy)
            dl['hpj'] = gen_em2(Et, mu, prec, logdet, cls, (PR['hpj_cov01'][0], PR['hpj_cov01'][1]), 0.1)[1]
            dl['lw'] = gen_em2(Et, mu, prec, logdet, cls, (PR['hybridD_lw'][0], PR['hybridD_lw'][1]), 0.1)[1]
            dl['oracle'] = {c: (Et[yt == c].mean(0) - mu[c]) if ((yt == c).sum() >= 20 and (ys_ == c).sum() >= 20) else np.zeros(D) for c in cls}
            if isev:
                for nm, Pk in P3.items():
                    PRk = priors3(Pk, cls, mu, tk, ty, sl, sub, doy)
                    dl[f'hpj_{nm}'] = gen_em2(Et, mu, prec, logdet, cls, (PRk['hpj_cov01'][0], PRk['hpj_cov01'][1]), 0.1)[1]
                    dl[f'lw_{nm}'] = gen_em2(Et, mu, prec, logdet, cls, (PRk['hybridD_lw'][0], PRk['hybridD_lw'][1]), 0.1)[1]
            for v in V:
                clf, key = v.split('_', 1)
                E2 = Es if dl[key] is None else F.resynthesize(Es, ys_, dl[key])
                pred[v][tem] = (hgb() if clf == 'hgb' else rf()).fit(assemble(*split(E2, nd)), ys_).predict(XB[tem])
        ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in pred.items()}
        out.append(dict(group=TG, src=ya, tgt=yb, **ba)); print(TG, ya, '->', yb, {v: round(x, 3) for v, x in ba.items() if v in ('hgb_em', 'hgb_hpj', 'hgb_lw')}, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(out).to_csv(fo, index=False)
    print('DONE', TG)
