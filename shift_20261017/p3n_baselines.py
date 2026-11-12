"""P3n: required baselines from the novelty check, same transfers/folds.
saerens: prior-only EM on source HGB probabilities (Saerens et al. 2002);
bp2001: Bruzzone & Prieto 2001 unsupervised ML retraining - EM updating priors, means AND covariances on all dims, Gaussian ML classifier;
em_all_post: mean-shift EM on all dims (no anchor), posterior classifier;
ours_gate: gated ensemble (P3m gate_transfer). Run: D:\\anaconda\\envs\\sar\\python.exe p3n_baselines.py -> p3n_baselines.csv"""
import os, json, time
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p3k_target_side.py'), encoding='utf-8').read()
exec(_src.split('rows = []')[0])
G = os.path.dirname(os.path.abspath(__file__)); t0 = time.time(); TH_T = 0.6

def saerens(P, p_src, n_it=100):
    pt = p_src.copy()
    for _ in range(n_it):
        Q = P * (pt / np.maximum(p_src, 1e-9)); Q /= Q.sum(1, keepdims=True); pt = Q.mean(0)
    return Q

def bp2001(Xt, mu, prec, logdet, cls, n_it=30, reg=1e-3):
    d = Xt.shape[1]; m = {c: mu[c].copy() for c in cls}; S = {c: np.linalg.inv(prec[c]) for c in cls}; pi = {c: 1.0 / len(cls) for c in cls}
    for _ in range(n_it):
        P = {c: np.linalg.inv(S[c]) for c in cls}; LD = {c: np.linalg.slogdet(S[c])[1] for c in cls}
        L = np.stack([np.log(pi[c]) - 0.5 * LD[c] - 0.5 * np.einsum('ij,jk,ik->i', Xt - m[c], P[c], Xt - m[c]) for c in cls], 1)
        R = np.exp(L - logsumexp(L, 1, keepdims=True))
        for j, c in enumerate(cls):
            w = R[:, j]; sw = max(w.sum(), 1e-9); pi[c] = max(w.mean(), 1e-6); m[c] = (w[:, None] * Xt).sum(0) / sw
            Xc = Xt - m[c]; S[c] = (w[:, None] * Xc).T @ Xc / sw + reg * np.eye(d)
    P = {c: np.linalg.inv(S[c]) for c in cls}; LD = {c: np.linalg.slogdet(S[c])[1] for c in cls}
    L = np.stack([np.log(pi[c]) - 0.5 * LD[c] - 0.5 * np.einsum('ij,jk,ik->i', Xt - m[c], P[c], Xt - m[c]) for c in cls], 1)
    return np.exp(L - logsumexp(L, 1, keepdims=True))

rows = []
for s in ('A', 'B'):
    for trk in (37, 88):
        for ya, yb in ((2024, 2025), (2025, 2024)):
            z = Dm[s]; g = lambda yr, l: z[f'{yr}_T{trk}_M9_{l}'].astype(float)
            vva, vha, coa, vvb, vhb, cob = g(ya, 'VV'), g(ya, 'VH'), g(ya, 'COH'), g(yb, 'VV'), g(yb, 'VH'), g(yb, 'COH')
            Ya, Yb = z[f'{ya}_y'], z[f'{yb}_y']; Fa, Fb = z[f'{ya}_folds'], z[f'{yb}_folds']; XB = assemble(vvb, vhb, cob)
            nd, nc = vva.shape[1], coa.shape[1]
            Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
            mask = np.r_[np.ones(2 * nd, bool), np.zeros(nc, bool)]; mall = np.ones(2 * nd + nc, bool)
            out = {v: np.full(len(Yb), -1) for v in ('raw', 'saerens', 'bp2001', 'gauss_src', 'em_all_post', 'ours_gate')}
            for k in range(6):
                tr, te = Fa[:, k] == 'TRAIN', Fb[:, k] == 'TEST'
                if not te.sum(): continue
                ya_ = Ya[tr]; m0 = hgb().fit(assemble(vva[tr], vha[tr], coa[tr]), ya_); P0 = fullp(m0, XB[te]); out['raw'][te] = P0.argmax(1)
                out['saerens'][te] = saerens(P0, np.bincount(ya_, minlength=K) / len(ya_)).argmax(1)
                mu, prec, logdet, cls = gauss_fit(Ea[tr], ya_)
                # Gaussian ML classifier without adaptation (reference for bp2001)
                Lg = np.stack([-0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', Eb[te] - mu[c], prec[c], Eb[te] - mu[c]) + np.log((ya_ == c).mean()) for c in cls], 1)
                pg = np.zeros((te.sum(), K)); pg[:, cls] = np.exp(Lg - logsumexp(Lg, 1, keepdims=True)); out['gauss_src'][te] = pg.argmax(1)
                Rb = bp2001(Eb[te], mu, prec, logdet, cls); pb = np.zeros((te.sum(), K)); pb[:, cls] = Rb; out['bp2001'][te] = pb.argmax(1)
                _, Ra = em_full(Eb[te], mu, prec, logdet, cls, mall); pa = np.zeros((te.sum(), K)); pa[:, cls] = Ra; out['em_all_post'][te] = pa.argmax(1)
                delta, R = em_full(Eb[te], mu, prec, logdet, cls, mask)
                D = np.stack([delta[c][:2 * nd] for c in cls]); sh = R @ D
                Pt = fullp(m0, assemble(vvb[te] - sh[:, :nd], vhb[te] - sh[:, nd:], cob[te])); post = np.zeros((te.sum(), K)); post[:, cls] = R
                a = float((Pt.argmax(1) == post.argmax(1)).mean())
                out['ours_gate'][te] = (0.5 * Pt + 0.5 * post).argmax(1) if a >= TH_T else Pt.argmax(1)
            predw = np.full(len(Yb), -1)
            for k in range(6):
                tr, te = Fb[:, k] == 'TRAIN', Fb[:, k] == 'TEST'
                if te.sum(): predw[te] = hgb().fit(XB[tr], Yb[tr]).predict(XB[te])
            bw = balanced_accuracy_score(Yb[predw >= 0], predw[predw >= 0])
            ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in out.items()}
            for v in ba:
                rows.append(dict(site=s, track=trk, src=ya, tgt=yb, variant=v, ba=ba[v], ba_within=bw, recovery=(ba[v] - ba['raw']) / (bw - ba['raw'])))
            print(s, trk, ya, '->', yb, {v: round(ba[v], 3) for v in ba}, 'within', round(bw, 3), round(time.time() - t0), 's', flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(G, 'p3n_baselines.csv'), index=False)
summ = dict(median_recovery=T.groupby('variant').recovery.median().round(3).to_dict(), mean_recovery=T.groupby('variant').recovery.mean().round(3).to_dict(),
            n_positive={v: int((g.recovery > 0).sum()) for v, g in T.groupby('variant')}, min_recovery=T.groupby('variant').recovery.min().round(3).to_dict(),
            median_ba=T.groupby('variant').ba.median().round(3).to_dict(), script=os.path.abspath(__file__), run_at=time.strftime('%Y-%m-%d %H:%M:%S'))
json.dump(summ, open(os.path.join(G, 'p3n_summary.json'), 'w'), indent=1); print(json.dumps(summ, indent=1))
