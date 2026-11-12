"""P3k (next branch after P3j): EM offsets are accurate (MAE ~0.35 dB) but source-shift+retrain loses accuracy.
Use EM on the TARGET side instead, with EM responsibilities (no labels):
tshift_soft: x_t' = x_t - sum_c r_tc delta_c (backscatter dims), classify with the source HGB;
tshift_hard: de-shift by delta of argmax_c r_tc;
em_post: EM mixture posterior (shifted Gaussians) as the classifier;
ens: average of source-shift HGB probabilities (P3f 'em') and EM posterior.
Run: D:\\anaconda\\envs\\sar\\python.exe p3k_target_side.py -> p3k_target_side.csv"""
import os, json, time
import numpy as np, pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'p3f_em.py'), encoding='utf-8').read()
exec(_src.split('rows, diag = [], []')[0])           # gauss_fit, em, impute, hgb, assemble, K, Dm, NIT, fullp
G = os.path.dirname(os.path.abspath(__file__)); t0 = time.time()

def em_full(Xt, mu, prec, logdet, cls, mask):
    d = Xt.shape[1]; delta = {c: np.zeros(d) for c in cls}; pi = {c: 1.0 / len(cls) for c in cls}
    for _ in range(NIT):
        L = np.stack([np.log(pi[c]) - 0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', Xt - mu[c] - delta[c], prec[c], Xt - mu[c] - delta[c]) for c in cls], 1)
        R = np.exp(L - logsumexp(L, 1, keepdims=True))
        for j, c in enumerate(cls):
            w = R[:, j]; pi[c] = max(w.mean(), 1e-6); new = (w[:, None] * (Xt - mu[c])).sum(0) / max(w.sum(), 1e-9); new[~mask] = 0; delta[c] = new
    L = np.stack([np.log(pi[c]) - 0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', Xt - mu[c] - delta[c], prec[c], Xt - mu[c] - delta[c]) for c in cls], 1)
    R = np.exp(L - logsumexp(L, 1, keepdims=True))
    return delta, R

rows = []
for s in ('A', 'B'):
    for trk in (37, 88):
        for ya, yb in ((2024, 2025), (2025, 2024)):
            z = Dm[s]; g = lambda yr, l: z[f'{yr}_T{trk}_M9_{l}'].astype(float)
            vva, vha, coa, vvb, vhb, cob = g(ya, 'VV'), g(ya, 'VH'), g(ya, 'COH'), g(yb, 'VV'), g(yb, 'VH'), g(yb, 'COH')
            Ya, Yb = z[f'{ya}_y'], z[f'{yb}_y']; Fa, Fb = z[f'{ya}_folds'], z[f'{yb}_folds']; XB = assemble(vvb, vhb, cob)
            nd, nc = vva.shape[1], coa.shape[1]
            Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
            mask = np.r_[np.ones(2 * nd, bool), np.zeros(nc, bool)]
            out = {v: np.full(len(Yb), -1) for v in ('raw', 'em_srcshift', 'tshift_soft', 'tshift_hard', 'em_post', 'ens')}
            for k in range(6):
                tr, te = Fa[:, k] == 'TRAIN', Fb[:, k] == 'TEST'
                if not te.sum(): continue
                ya_ = Ya[tr]; m0 = hgb().fit(assemble(vva[tr], vha[tr], coa[tr]), ya_); out['raw'][te] = m0.predict(XB[te])
                mu, prec, logdet, cls = gauss_fit(Ea[tr], ya_); delta, R = em_full(Eb[te], mu, prec, logdet, cls, mask)
                D = np.stack([delta[c][:2 * nd] for c in cls])                     # (C, 2nd)
                # target-side soft / hard de-shift
                sh_soft = R @ D; sh_hard = D[R.argmax(1)]
                for v, sh in (('tshift_soft', sh_soft), ('tshift_hard', sh_hard)):
                    vb2, hb2 = vvb[te] - sh[:, :nd], vhb[te] - sh[:, nd:]
                    out[v][te] = m0.predict(assemble(vb2, hb2, cob[te]))
                # EM posterior classifier
                post = np.zeros((te.sum(), K)); post[:, cls] = R; out['em_post'][te] = post.argmax(1)
                # source-shift retrain (P3f em) + ensemble
                va2, ha2 = vva[tr].copy(), vha[tr].copy()
                for c in cls:
                    ms = ya_ == c; va2[ms] += delta[c][:nd]; ha2[ms] += delta[c][nd:2 * nd]
                m1 = hgb().fit(assemble(va2, ha2, coa[tr]), ya_); P1 = fullp(m1, XB[te]); out['em_srcshift'][te] = P1.argmax(1)
                out['ens'][te] = (0.5 * P1 + 0.5 * post).argmax(1)
            predw = np.full(len(Yb), -1)
            for k in range(6):
                tr, te = Fb[:, k] == 'TRAIN', Fb[:, k] == 'TEST'
                if te.sum(): predw[te] = hgb().fit(XB[tr], Yb[tr]).predict(XB[te])
            bw = balanced_accuracy_score(Yb[predw >= 0], predw[predw >= 0])
            ba = {v: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for v, p in out.items()}
            for v in ba:
                rows.append(dict(site=s, track=trk, src=ya, tgt=yb, variant=v, ba=ba[v], ba_within=bw, recovery=(ba[v] - ba['raw']) / (bw - ba['raw'])))
            print(s, trk, ya, '->', yb, {v: round(ba[v], 3) for v in ba}, 'within', round(bw, 3), round(time.time() - t0), 's', flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(G, 'p3k_target_side.csv'), index=False)
summ = dict(median_recovery=T.groupby('variant').recovery.median().round(3).to_dict(), mean_recovery=T.groupby('variant').recovery.mean().round(3).to_dict(),
            dry_to_wet_positive={v: int((g.recovery > 0).sum()) for v, g in T[T.src == 2025].groupby('variant')},
            script=os.path.abspath(__file__), run_at=time.strftime('%Y-%m-%d %H:%M:%S'))
json.dump(summ, open(os.path.join(G, 'p3k_summary.json'), 'w'), indent=1); print(json.dumps(summ, indent=1))
