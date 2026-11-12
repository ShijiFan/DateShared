"""NEW-REGION CONFIRMATION (France FR-A), single pre-registered run: CONFIRM_PREREG_DRAFT_ZH.md (N1-N5) + toy_B/full/frozen_j2/N6_ADDENDUM_ZH.md (N6).
Frozen method frozen/hybridD_lw.py with archive_NL_all.npz (main) and archive_NL_T88.npz (secondary, built from p9_arch_A88 + p9_arch_B88 by the frozen
Archive.build before this run); baselines em_cov01, hpj_cov01 (NL-all LW joint prior, p3_joint.build_joint over all NL p1_rows), raw and oracle
(p1_tasks.classify_fixed); j2 (frozen_j2/j2_wc.py, archive_j2_NL_all.npz, Meteo-France RR 76679001); HGB resynthesis (frozen HGB, mean shift);
N5 local 3-year FR archive (3 of the other 6 FR years, rng 20261106 in transfer order); N6c date coefficients vs FR dAPI.
Transfers: all ordered pairs of the FR years; 6 spatial folds; source TRAIN minus cross-year 400 m exclusion (fr_xyexcl.drop).
Usage: python confirm_fr.py -> CONFIRM_FR_rows.csv, CONFIRM_FR_coef.csv (resumable per transfer); confirm_fr_summary.py computes the criteria."""
import os, sys, json, itertools, time
import numpy as np, pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
for p in (UP, os.path.join(UP, 'frozen'), os.path.join(UP, 'toy_B', 'full', 'frozen_j2'), os.path.join(HERE, 'pipeline')): sys.path.insert(0, p)
import hybridD_lw as F
import j2_wc as J
from fr_xyexcl import drop as xydrop
from p1_tasks import classify_fixed, H
from p3_joint import build_joint, joint_for
from p9_lib import gen_em2
hgb, assemble = H['hgb'], H['assemble']
ROOT = r'D:\sar_data\fr_a'; Z = np.load(os.path.join(ROOT, 'features_FR.npz'), allow_pickle=True); M = json.load(open(os.path.join(ROOT, 'features_FR_meta.json')))
TAG = {int(t.split('_')[0]): t for t in M}; YEARS = sorted(TAG)
AR_ALL = F.Archive.load(os.path.join(UP, 'frozen', 'archive_NL_all.npz')); AR_T88 = F.Archive.load(os.path.join(UP, 'frozen', 'archive_NL_T88.npz'))
AJ2 = J.J2Archive.load(os.path.join(UP, 'toy_B', 'full', 'frozen_j2', 'archive_j2_NL_all.npz'))
RR = pd.read_csv(os.path.join(UP, 'toy_B', 'full', 'frozen_j2', 'fr_rr_daily.csv'), parse_dates=['date']).set_index('date')
P = RR[[c for c in RR.columns if c != 'date'][0]].clip(lower=0).astype(float)
allr = pd.concat([pd.read_parquet(os.path.join(UP, f'p1_rows_{g}.parquet')) for g in ('A37', 'A88', 'B88')], ignore_index=True)
KEYS, JM, JC = build_joint(allr)
OUT = os.path.join(HERE, 'CONFIRM_FR_rows.csv'); OUTC = os.path.join(HERE, 'CONFIRM_FR_coef.csv')


def data(y, S):
    m = M[TAG[y]]; pos = [m['slots'].index(s) for s in S]; dc = [c for p in pos for c in (2 * p, 2 * p + 1)]
    g = lambda l: Z[f'{TAG[y]}_{l}'].astype(float)
    return g('VV')[:, dc], g('VH')[:, dc], g('COH')[:, pos], Z[f'{TAG[y]}_y'], Z[f'{TAG[y]}_folds']


def dims(S, y):
    m = M[TAG[y]]; dt = pd.to_datetime(m['dates']); tk, doy = [], []
    for ty in (0, 1):
        for s in S:
            p = m['slots'].index(s)
            for u in (0, 1): tk.append((ty, s, u)); doy.append(dt[2 * p + u].dayofyear)
    for s in S:
        p = m['slots'].index(s); tk.append((2, s, 0)); doy.append((dt[2 * p] + (dt[2 * p + 1] - dt[2 * p]) / 2).dayofyear)
    return tk, np.array(doy)


def prep(ya, yb):
    S = tuple(sorted(set(M[TAG[ya]]['slots']) & set(M[TAG[yb]]['slots'])))
    vva, vha, coa, Ya, Fa = data(ya, S); vvb, vhb, cob, Yb, Fb = data(yb, S)
    Ea = F.impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = F.impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
    return S, Ea, Eb, Ya, Yb, Fa, Fb, vva.shape[1]


# ---- FR archive tasks for N5 (oracle shifts of FR year pairs; labels of archive years are history, never the evaluated target year) ----
print('building FR task cache', flush=True); CACHE = {}
for ya, yb in itertools.permutations(YEARS, 2):
    S, Ea, Eb, Ya, Yb, Fa, Fb, nd = prep(ya, yb); tk, doy = dims(S, ya)
    for k in range(6):
        trm = (Fa[:, k] == 'TRAIN') & ~xydrop(ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
        mu, prec, logdet, cls = F.gauss_fit(Ea[trm], Ya[trm]); Et, yt = Eb[te], Yb[te]
        orc = {c: (Et[yt == c].mean(0) - mu[c]) if ((yt == c).sum() >= 20 and (Ya[trm] == c).sum() >= 20) else np.full(len(tk), np.nan) for c in cls}
        ty, sl, sub = (np.array([t[i] for t in tk]) for i in range(3))
        CACHE[(ya, yb, k)] = dict(id=('FR', ya, yb, k), cls=list(cls), tk=tk, mu={c: mu[c] for c in cls}, ty=ty, sl=sl, sub=sub, doy=doy, orc=orc)


def n_complete(tasks):
    keys = sorted({tuple(int(x) for x in kk) for t in tasks for c in t['cls'] if np.isfinite(t['orc'][c]).all() for kk in t['tk']})
    if not keys: return 0
    kix = {kk: i for i, kk in enumerate(keys)}; nk = len(keys); n = 0
    for t in tasks:
        v = np.full(7 * nk, np.nan)
        for c in t['cls']:
            for i, kk in enumerate(t['tk']):
                if np.isfinite(t['orc'][c][i]): v[c * nk + kix[tuple(int(x) for x in kk)]] = t['orc'][c][i]
        n += int(~np.isnan(v).any())
    return n


def local_lw_joint(tasks):
    """hpj analogue on a local archive (same construction as p3_joint.build_joint: complete tasks, mean + Ledoit-Wolf)."""
    keys = sorted({tuple(int(x) for x in kk) for t in tasks for c in t['cls'] if np.isfinite(t['orc'][c]).all() for kk in t['tk']})
    kix = {kk: i for i, kk in enumerate(keys)}; nk = len(keys); V = []
    for t in tasks:
        v = np.full(7 * nk, np.nan)
        for c in t['cls']:
            for i, kk in enumerate(t['tk']):
                if np.isfinite(t['orc'][c][i]): v[c * nk + kix[tuple(int(x) for x in kk)]] = t['orc'][c][i]
        V.append(v)
    V = np.array(V); full = ~np.isnan(V).any(1)
    return keys, V[full].mean(0), LedoitWolf().fit(V[full]).covariance_


rng = np.random.default_rng(20261106); todo = list(itertools.permutations(YEARS, 2))
SUB3 = {tr: sorted(rng.choice([y for y in YEARS if y not in tr], 3, replace=False).tolist()) for tr in todo}
json.dump({f'{a}_{b}': v for (a, b), v in SUB3.items()}, open(os.path.join(HERE, 'CONFIRM_FR_N5_subsets.json'), 'w'), indent=1)
if os.path.exists(OUT):                                  # deviation 22:40: migrate rows written before the N5 fix (values not printed)
    R0 = pd.read_csv(OUT)
    if 'n_local_complete' not in R0:
        nl = [n_complete([CACHE[(a, b, k)] for a in SUB3[(int(r.src), int(r.tgt))] for b in SUB3[(int(r.src), int(r.tgt))] if a != b for k in range(6)]) for r in R0.itertuples()]
        R0.insert(3, 'n_local_complete', nl); R0.loc[R0.n_local_complete < 10, ['lw_local3', 'hpj_local3']] = np.nan
        R0.to_csv(OUT, index=False); print('migrated', len(R0), 'rows; n_local_complete', nl, flush=True)
done = {(r.src, r.tgt) for r in pd.read_csv(OUT).itertuples()} if os.path.exists(OUT) else set(); t0 = time.time()
GV = ['g_raw', 'em_cov01', 'hpj_cov01', 'lw_all', 'lw_T88', 'j2', 'g_oracle', 'lw_local3', 'hpj_local3']; HV = ['hgb_raw', 'hgb_em', 'hgb_hpj', 'hgb_lw', 'hgb_oracle']
for ya, yb in todo:
    if (ya, yb) in done: continue
    S, Ea, Eb, Ya, Yb, Fa, Fb, nd = prep(ya, yb); tk, doy = dims(S, ya); D = Ea.shape[1]
    api_s, api_t = J.api_for_task(P, M[TAG[ya]], tk), J.api_for_task(P, M[TAG[yb]], tk)
    loc = [CACHE[(a, b, k)] for a in SUB3[(ya, yb)] for b in SUB3[(ya, yb)] if a != b for k in range(6)]
    n_loc = n_complete(loc); LOCAL_OK = n_loc >= 10                   # deviation 22:40 (prereg §9): degenerate local archive -> N5 NaN
    if LOCAL_OK: AR_L = F.Archive.build(loc, dict(source=f'FR local {SUB3[(ya, yb)]}')); KL, MLm, CL = local_lw_joint(loc)
    split = lambda E: (E[:, :nd], E[:, nd:2 * nd], E[:, 2 * nd:]); XB = assemble(*split(Eb))
    pred = {v: np.full(len(Yb), -1) for v in GV + HV}; coef = []
    for k in range(6):
        trm = (Fa[:, k] == 'TRAIN') & ~xydrop(ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
        if not tem.sum(): continue
        Es, ys_, Et, yt = Ea[trm], Ya[trm], Eb[tem], Yb[tem]; mu, prec, logdet, cls = F.gauss_fit(Es, ys_); C = len(cls)
        dOR = {c: (Et[yt == c].mean(0) - mu[c]) if ((yt == c).sum() >= 20 and (ys_ == c).sum() >= 20) else np.zeros(D) for c in cls}
        pred['g_raw'][tem] = classify_fixed(Et, mu, prec, logdet, cls, {c: np.zeros(D) for c in cls})
        pred['g_oracle'][tem] = classify_fixed(Et, mu, prec, logdet, cls, dOR)
        lab, d_em = gen_em2(Et, mu, prec, logdet, cls, None, 0.1); pred['em_cov01'][tem] = lab
        m, Ps = joint_for(KEYS, JM, JC, cls, tk); lab, d_hpj = gen_em2(Et, mu, prec, logdet, cls, (m, Ps), 0.1); pred['hpj_cov01'][tem] = lab
        m, Ps = AR_ALL.prior(cls, mu, tk, doy); lab, d_lw = F.map_em(Et, mu, prec, logdet, cls, m, Ps); pred['lw_all'][tem] = lab
        m_all = m
        m, Ps = AR_T88.prior(cls, mu, tk, doy); pred['lw_T88'][tem] = F.map_em(Et, mu, prec, logdet, cls, m, Ps)[0]
        m, Ps = AJ2.prior(cls, mu, tk, api_s, api_t); pred['j2'][tem] = F.map_em(Et, mu, prec, logdet, cls, m, Ps)[0]
        if LOCAL_OK:
            m, Ps = AR_L.prior(cls, mu, tk, doy); pred['lw_local3'][tem] = F.map_em(Et, mu, prec, logdet, cls, m, Ps)[0]
            m, Ps = joint_for(KL, MLm, CL, cls, tk); pred['hpj_local3'][tem] = gen_em2(Et, mu, prec, logdet, cls, (m, Ps), 0.1)[0]
        for v, dl in (('hgb_raw', None), ('hgb_em', d_em), ('hgb_hpj', d_hpj), ('hgb_lw', d_lw), ('hgb_oracle', dOR)):
            E2 = Es if dl is None else F.resynthesize(Es, ys_, dl); pred[v][tem] = hgb().fit(assemble(*split(E2)), ys_).predict(XB[tem])
        ty = np.array([t[0] for t in tk]); sl = np.array([t[1] for t in tk]); sub = np.array([t[2] for t in tk])     # N6c
        B, kinds = F.basis(cls, mu, ty, sl, sub, doy); th, *_ = np.linalg.lstsq(B, np.concatenate([d_lw[c] for c in cls]) - m_all, rcond=None)
        bsd = sorted(set((int(sl[d]), int(sub[d])) for d in range(D) if ty[d] < 2))
        for j, (s_, u_) in enumerate(bsd):
            d0 = np.where((ty == 0) & (sl == s_) & (sub == u_))[0][0]
            coef.append(dict(src=ya, tgt=yb, fold=k, slot=s_, sub=u_, coef=th[j], api_s=api_s[d0], api_t=api_t[d0]))
    row = dict(src=ya, tgt=yb, n_slots=len(S), n_local_complete=n_loc,
               **{v: (balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) if (p >= 0).any() else np.nan) for v, p in pred.items()})
    pd.DataFrame([row]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    pd.DataFrame(coef).to_csv(OUTC, mode='a', header=not os.path.exists(OUTC), index=False)
    print(ya, '->', yb, round(time.time() - t0), 's', flush=True)                       # no BA printed: results are read once by the summary
print('DONE')
