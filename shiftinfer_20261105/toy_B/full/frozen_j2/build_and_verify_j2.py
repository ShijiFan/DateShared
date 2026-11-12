"""Freeze j2 weights with the exact B-full evaluator code (evalba.archive / cands_r3.f_wc / evalba.fit_joint) and verify the
portable j2_wc.py against the B-full j2 BA (runs/ba_cands_final_confirm_<G>.csv, 'j2_hpj_wc_cond'). No new variant, no
selection: verification only. archive_j2_NL_A / _B (leave-site-out, for verification), archive_j2_NL_all (new-region use:
both NL sites, each with its own KNMI station). -> VERIFY_J2.json"""
import os, sys, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); FULL = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, FULL)
import evalba as E
from cands_r3 import CANDS
import j2_wc as J
from p1_tasks import load, data, xydrop, dim_meta
from p3_joint import build_joint
N_CHECK = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CD = [c for c in CANDS if c['name'] == 'j2_hpj_wc_cond'][0]


def archive_sites(sites, allr):
    """evalba.archive generalised to several sites (identical for one site)."""
    rows = allr[allr.site.isin(sites)]; keys, jm, jc = build_joint(rows); P0 = dict(keys=keys, jm=jm, jc=jc); A = []
    for G in E.GROUPS:
        if G[0] not in sites: continue
        for t in pickle.load(open(os.path.join(E.UP, f'p9_arch_{G}.pkl'), 'rb')):
            _, ya, yb, k = t['id']
            c = E.make_ctx(G, E.meta(G), int(G[1:]), ya, yb, t['cls'], t['mu'], t['ty'], t['sl'], t['sub'], t['tk'], t['doy'], P0)
            c['orc'] = t['orc']; c['id'] = t['id']; A.append(c)
    return P0, A


allr = pd.concat([pd.read_parquet(os.path.join(E.UP, f'p1_rows_{g}.parquet')) for g in E.GROUPS], ignore_index=True)
ARCH = {}
for name, sites in (('A', ('A',)), ('B', ('B',)), ('all', ('A', 'B'))):
    P0, A = archive_sites(sites, allr); gs = CD['fit'](A, P0); S = E.fit_joint(CD, P0, A, gs)
    ar = J.J2Archive(P0['keys'], S['jm'], S['jc'], gs['th'], dict(source=f'NL sites {"+".join(sites)}', n_tasks=len(A), n_complete=int(S['n_complete']),
                     weather='KNMI RH daily, station A=283 B=280; API decay 0.9, 60 d'))
    ar.save(os.path.join(HERE, f'archive_j2_NL_{name}.npz')); ARCH[name] = ar
    print(name, ar.meta, 'th', np.round(gs['th'], 4).tolist(), flush=True)

res = {}
for TG in ('B88', 'A37', 'A88'):
    ar = J.J2Archive.load(os.path.join(HERE, f"archive_j2_NL_{'A' if TG[0] == 'B' else 'B'}.npz"))
    ref = pd.read_csv(os.path.join(FULL, 'runs', f'ba_cands_final_confirm_{TG}.csv'))
    ref = ref[ref.cand == 'j2_hpj_wc_cond'].set_index(['src', 'tgt'])['ba']
    site, trk, Z, M = load(TG); diffs = []
    for ya, yb in list(ref.index)[:N_CHECK]:
        S_ = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
        vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S_); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S_)
        Ea = E.impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = E.impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
        ty, sl, sub = dim_meta(S_); tk = list(zip(ty, sl, sub))
        Pw = E.wser(TG[0])[0]
        a_s = J.api(Pw, E.pair_dates(E.meta(TG), trk, ya, ty, sl, sub)[1]); a_t = J.api(Pw, E.pair_dates(E.meta(TG), trk, yb, ty, sl, sub)[1])
        pred = np.full(len(Yb), -1)
        for k in range(6):
            trm = (Fa[:, k] == 'TRAIN') & ~xydrop(TG, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
            if not tem.sum(): continue
            pred[tem] = J.adapt(Ea[trm], Ya[trm], Eb[tem], tk, a_s, a_t, ar)[0]
        ba = balanced_accuracy_score(Yb[pred >= 0], pred[pred >= 0]); diffs.append(abs(ba - ref.loc[(ya, yb)]))
        print(TG, ya, yb, round(ba, 6), round(ref.loc[(ya, yb)], 6), flush=True)
    res[TG] = dict(n=len(diffs), max_abs_diff=float(max(diffs)))
json.dump(dict(check=res, archives={k: dict(meta=a.meta, th=a.th.tolist()) for k, a in ARCH.items()}), open(os.path.join(HERE, 'VERIFY_J2.json'), 'w'), indent=1)
print(json.dumps(res))
