"""P3f (NOTE_P3f_em.md): observable-constrained conditional-shift EM.
Target = mixture of source class Gaussians with class-specific mean shifts on BACKSCATTER dims only (coherence = anchor, delta=0).
Variants: em (anchor), em_smooth (3-point temporal smoothing of delta), em_all (all dims may shift), stable_pl (P3e best).
Diagnostic only: correlation of EM delta with oracle class offsets (uses target labels).
Run: D:\\anaconda\\envs\\sar\\python.exe p3f_em.py -> p3f_em.csv, p3f_summary.json"""
import os, json, time
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
G = os.path.dirname(os.path.abspath(__file__)); SEED = 20260923; t0 = time.time(); K = 7; TAU = 0.6; NIT = 30
Dm = {s: np.load(os.path.join(G, f'dump_{s}.npz'), allow_pickle=True) for s in ('A', 'B')}
def hgb(): return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
                                                 l2_regularization=1.0, early_stopping=False, random_state=SEED)
def assemble(vv, vh, coh): return np.hstack([vv, vh, vh - vv, coh])
def fullp(m, X):
    P = np.zeros((len(X), K)); P[:, m.classes_] = m.predict_proba(X); return P
def impute(X, ref):  # fill NaN with source column medians (very few NaNs)
    X = X.copy(); m = np.isnan(X); X[m] = np.take(np.nanmedian(ref, 0), np.nonzero(m)[1]); return X

def gauss_fit(Xs, ys):
    mu, prec, logdet, cls = {}, {}, {}, []
    for c in range(K):
        Xc = Xs[ys == c]
        if len(Xc) < 30: continue
        lw = LedoitWolf().fit(Xc); S = lw.covariance_
        sign, ld = np.linalg.slogdet(S); mu[c] = Xc.mean(0); prec[c] = np.linalg.inv(S); logdet[c] = ld; cls.append(c)
    return mu, prec, logdet, cls

def em(Xt, mu, prec, logdet, cls, shift_mask, smooth_groups=None):
    d = Xt.shape[1]; delta = {c: np.zeros(d) for c in cls}; pi = {c: 1.0 / len(cls) for c in cls}
    for _ in range(NIT):
        L = np.stack([np.log(pi[c]) - 0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', Xt - mu[c] - delta[c], prec[c], Xt - mu[c] - delta[c]) for c in cls], 1)
        R = np.exp(L - logsumexp(L, 1, keepdims=True))
        for j, c in enumerate(cls):
            w = R[:, j]; pi[c] = max(w.mean(), 1e-6)
            new = (w[:, None] * (Xt - mu[c])).sum(0) / max(w.sum(), 1e-9)
            new[~shift_mask] = 0.0
            if smooth_groups is not None:
                for grp in smooth_groups:
                    v = new[grp]; new[grp] = np.convolve(np.r_[v[0], v, v[-1]], np.ones(3) / 3, 'valid')
            delta[c] = new
    return delta

rows, diag = [], []
for s in ('A', 'B'):
    for trk in (37, 88):
        for ya, yb in ((2024, 2025), (2025, 2024)):
            z = Dm[s]; g = lambda yr, l: z[f'{yr}_T{trk}_M9_{l}'].astype(float)
            vva, vha, coa, vvb, vhb, cob = g(ya, 'VV'), g(ya, 'VH'), g(ya, 'COH'), g(yb, 'VV'), g(yb, 'VH'), g(yb, 'COH')
            Ya, Yb = z[f'{ya}_y'], z[f'{yb}_y']; Fa, Fb = z[f'{ya}_folds'], z[f'{yb}_folds']
            nd, nc = vva.shape[1], coa.shape[1]
            Ea_full = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb_full = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
            mask_anchor = np.r_[np.ones(2 * nd, bool), np.zeros(nc, bool)]; mask_all = np.ones(2 * nd + nc, bool)
            groups = [np.arange(0, nd), np.arange(nd, 2 * nd)]
            XB = assemble(vvb, vhb, cob)
            out = {v: np.full(len(Yb), -1) for v in ('raw', 'stable_pl', 'em', 'em_smooth', 'em_all')}
            for k in range(6):
                tr, te = Fa[:, k] == 'TRAIN', Fb[:, k] == 'TEST'
                if not te.sum(): continue
                ya_ = Ya[tr]
                out['raw'][te] = hgb().fit(assemble(vva[tr], vha[tr], coa[tr]), ya_).predict(XB[te])
                # stable_pl (P3e)
                smd = lambda A, B: np.abs(np.nanmean(B, 0) - np.nanmean(A, 0)) / np.sqrt((np.nanvar(A, 0) + np.nanvar(B, 0)) / 2)
                stable = np.argsort((smd(vva, vvb) + smd(vha, vhb)) / 2)[:nd // 2]
                P = fullp(hgb().fit(assemble(vva[tr][:, stable], vha[tr][:, stable], coa[tr]), ya_), assemble(vvb[te][:, stable], vhb[te][:, stable], cob[te]))
                pl, keep = P.argmax(1), P.max(1) > TAU; va2, ha2 = vva[tr].copy(), vha[tr].copy()
                for c in range(K):
                    mt, ms = (pl == c) & keep, ya_ == c
                    if mt.sum() >= 20 and ms.sum() >= 20:
                        va2[ms] += np.nanmedian(vvb[te][mt], 0) - np.nanmedian(vva[tr][ms], 0); ha2[ms] += np.nanmedian(vhb[te][mt], 0) - np.nanmedian(vha[tr][ms], 0)
                out['stable_pl'][te] = hgb().fit(assemble(va2, ha2, coa[tr]), ya_).predict(XB[te])
                # EM variants
                mu, prec, logdet, cls = gauss_fit(Ea_full[tr], ya_)
                for v, mask, sm in (('em', mask_anchor, None), ('em_smooth', mask_anchor, groups), ('em_all', mask_all, None)):
                    delta = em(Eb_full[te], mu, prec, logdet, cls, mask, sm)
                    va3, ha3, ca3 = vva[tr].copy(), vha[tr].copy(), coa[tr].copy()
                    for c in cls:
                        ms = ya_ == c; va3[ms] += delta[c][:nd]; ha3[ms] += delta[c][nd:2 * nd]; ca3[ms] += delta[c][2 * nd:]
                    out[v][te] = hgb().fit(assemble(va3, ha3, ca3), ya_).predict(XB[te])
                    if v == 'em' and k == 0:   # diagnostic: EM delta vs oracle class offsets (target labels)
                        for c in cls:
                            mb = Yb[te] == c
                            if mb.sum() < 20: continue
                            orc = np.r_[np.nanmedian(vvb[te][mb], 0) - np.nanmedian(vva[tr][ya_ == c], 0), np.nanmedian(vhb[te][mb], 0) - np.nanmedian(vha[tr][ya_ == c], 0)]
                            diag.append(dict(site=s, track=trk, src=ya, cls=c, corr=float(np.corrcoef(orc, delta[c][:2 * nd])[0, 1]),
                                             mae=float(np.nanmean(np.abs(orc - delta[c][:2 * nd]))), mean_oracle=float(np.nanmean(orc)), mean_em=float(delta[c][:2 * nd].mean())))
            predw = np.full(len(Yb), -1)
            for k in range(6):
                tr, te = Fb[:, k] == 'TRAIN', Fb[:, k] == 'TEST'
                if te.sum(): predw[te] = hgb().fit(XB[tr], Yb[tr]).predict(XB[te])
            bw = balanced_accuracy_score(Yb[predw >= 0], predw[predw >= 0])
            ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in out.items()}
            for v in ba:
                rows.append(dict(site=s, track=trk, src=ya, tgt=yb, variant=v, ba=ba[v], ba_within=bw, recovery=(ba[v] - ba['raw']) / (bw - ba['raw'])))
            print(s, trk, ya, '->', yb, {v: round(ba[v], 3) for v in ba}, 'within', round(bw, 3), round(time.time() - t0), 's', flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(G, 'p3f_em.csv'), index=False); Dg = pd.DataFrame(diag); Dg.to_csv(os.path.join(G, 'p3f_delta_diag.csv'), index=False)
piv = T.pivot_table(index=['site', 'track', 'src'], columns='variant', values='recovery')
summ = dict(median_recovery=T.groupby('variant').recovery.median().round(3).to_dict(), mean_recovery=T.groupby('variant').recovery.mean().round(3).to_dict(),
            wins_vs_stable_pl={v: int((piv[v] > piv['stable_pl']).sum()) for v in ('em', 'em_smooth', 'em_all')},
            anchor_helps_wins=int((piv['em'] > piv['em_all']).sum()), n=int(len(piv)),
            delta_vs_oracle=dict(median_corr=float(Dg['corr'].median()), median_mae_dB=float(Dg['mae'].median())),
            script=os.path.abspath(__file__), run_at=time.strftime('%Y-%m-%d %H:%M:%S'), runtime_s=round(time.time() - t0))
json.dump(summ, open(os.path.join(G, 'p3f_summary.json'), 'w'), indent=1); print(json.dumps(summ, indent=1))
