"""B-full evaluator (PROTOCOL_B_FULL_ZH.md): score candidate shift equations by cross-year adaptation BA through the P9 MAP-EM.
A candidate = prior mean m(ctx) + shift basis B(ctx) (+ optional residual PPCA modes); global parameters fitted on the archive
site only; coefficient / residual scales estimated on archive tasks exactly as p9_lib.phys_fit but around the candidate mean.
Usage: python evalba.py <round module, e.g. cands_r0> [search|confirm] -> ba_<cand>.csv rows appended, results.jsonl"""
import os, sys, json, time, pickle, itertools, importlib
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UP); sys.path.insert(0, HERE)
from p1_tasks import load, data, impute, gauss_fit, xydrop, dim_meta, DEV, K
from p3_joint import build_joint, joint_for
from p9_lib_snap import gen_em2, woodbury, dim_dates, keyspace, key_type, ppca, map_lowrank
GROUPS = ('A37', 'A88', 'B88'); KNMI = os.path.join(UP, '..', 'multiyear_20261017', 'knmi_daily_my.csv'); STN = {'A': 283, 'B': 280}
OUT = os.path.join(HERE, 'runs'); os.makedirs(OUT, exist_ok=True)

# ---------------- weather (KNMI RH, EV24 only) ----------------
_W = {}
def wser(site):
    if site not in _W:
        w = pd.read_csv(KNMI, parse_dates=['date']); w = w[w.station_code == STN[site]].set_index('date').sort_index()
        _W[site] = (w.RH.clip(lower=0) / 10.0, w.EV24 / 10.0)
    return _W[site]


def api(site, dates, decay=0.9, win=60):
    P = wser(site)[0]; out = []
    for t in dates:
        p = P.loc[t - pd.Timedelta(days=win - 1):t].to_numpy()[::-1]; out.append(float((p * decay ** np.arange(len(p))).sum()))
    return np.array(out)


def psum(site, dates, n, which=0):
    S = wser(site)[which]; return np.array([float(S.loc[t - pd.Timedelta(days=n - 1):t].sum()) for t in dates])


def rain_between(site, d0, d1):
    """rain in (d0, d1] (mm); 0 when d0 == d1."""
    P = wser(site)[0]; return np.array([float(P.loc[a + pd.Timedelta(days=1):b].sum()) if b > a else 0.0 for a, b in zip(d0, d1)])


def pair_dates(M, trk, yr, ty, sl, sub):
    """(start, end) acquisition dates per dim; backscatter dims start == end."""
    m = M[f'{yr}_T{trk}']; dt = pd.to_datetime(m['dates']); s, e = [], []
    for t, q, u in zip(ty, sl, sub):
        p = m['slots'].index(q)
        if t < 2: s.append(dt[2 * p + u]); e.append(dt[2 * p + u])
        else: s.append(dt[2 * p]); e.append(dt[2 * p + 1])
    return pd.DatetimeIndex(s), pd.DatetimeIndex(e)


class Ctx(dict):
    """task context; weather features cached lazily: ctx.w('api_s'), ctx.w('api_t'), ctx.w('rainpair_s') ..."""
    def w(self, name, **kw):
        key = (name, tuple(sorted(kw.items())))
        if key not in self['_wc']:
            yr, side = (self['ya'], 's') if name.endswith('_s') else (self['yb'], 't')
            d0, d1 = self['pd_' + side]; base = name[:-2]
            if base == 'api': v = api(self['site'], d1, **kw)
            elif base.startswith('P'): v = psum(self['site'], d1, int(base[1:]))
            elif base.startswith('E'): v = psum(self['site'], d1, int(base[1:]), 1)
            elif base == 'rainpair': v = rain_between(self['site'], d0, d1)
            elif base in ('r60', 'e60'):      # (dims x 60) daily series ending at the acquisition, most recent first
                S = wser(self['site'])[0 if base == 'r60' else 1]
                v = np.stack([S.loc[t - pd.Timedelta(days=59):t].to_numpy()[::-1] for t in d1])
            else: raise KeyError(name)
            self['_wc'][key] = v
        return self['_wc'][key]


def make_ctx(G, M, trk, ya, yb, cls, mu, ty, sl, sub, tk, doy, P0):
    c = Ctx(site=G[0], group=G, ya=ya, yb=yb, cls=list(cls), mu=mu, ty=ty, sl=sl, sub=sub, tk=tk, doy=doy, _wc={})
    c['pd_s'] = pair_dates(M, trk, ya, ty, sl, sub); c['pd_t'] = pair_dates(M, trk, yb, ty, sl, sub)
    c['dd_s'] = dim_dates(M, trk, ya, None, ty, sl, sub); c['dd_t'] = dim_dates(M, trk, yb, None, ty, sl, sub)   # COH = pair midpoint (P9)
    c['m_arch'] = joint_for(P0['keys'], P0['jm'], P0['jc'], c['cls'], tk)[0]
    return c


# ---------------- archive ----------------
_META = {}
def meta(G):
    if G not in _META: _META[G] = json.load(open(os.path.join(r'D:\sar_data\multiyear', f'features_{G[0]}_meta.json')))
    return _META[G]


def archive(trs, allr, years=None):
    """P0 (keys, joint mean/cov, keyspace ids) and archive contexts with oracle shifts for the training site
    (optionally only archive tasks with both years in `years`, as p9_curve)."""
    rows = allr[allr.site == trs]
    if years is not None: rows = rows[rows.src.isin(years) & rows.tgt.isin(years)]
    keys, jm, jc = build_joint(rows)
    _, V, ids, full = keyspace(rows); kt = key_type(keys); idf = [ids[i] for i in np.where(full)[0]]
    P0 = dict(keys=keys, jm=jm, jc=jc, kt=kt, idf=idf, Psi_full=None)
    A = []
    for G in GROUPS:
        if G[0] != trs: continue
        for t in pickle.load(open(os.path.join(UP, f'p9_arch_{G}.pkl'), 'rb')):
            _, ya, yb, k = t['id']
            if years is not None and not (ya in years and yb in years): continue
            c = make_ctx(G, meta(G), int(G[1:]), ya, yb, t['cls'], t['mu'], t['ty'], t['sl'], t['sub'], t['tk'], t['doy'], P0)
            c['orc'] = t['orc']; c['id'] = t['id']; A.append(c)
    return P0, A


def fit_scales(cand, P0, A, gs):
    """per archive task LS of (oracle - m) on B; coefficient variance by kind, residual var by dtype, residual PPCA."""
    keys = P0['keys']; kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); coef = {}; res_by_t = {0: [], 1: [], 2: []}; RV = {}
    for t in A:
        cls, tk, ty = t['cls'], t['tk'], t['ty']; D = len(tk)
        B, kinds = cand['basis'](t, gs); m = cand.get('scale_mean', cand['mean'])(t, gs)
        y = np.concatenate([t['orc'][c] for c in cls]) - m; ok = ~np.isnan(y)
        if B.shape[1]:
            th, *_ = np.linalg.lstsq(B[ok], y[ok], rcond=None); r = np.full(len(y), np.nan); r[ok] = y[ok] - B[ok] @ th
            for kd, v in zip(kinds, th): coef.setdefault(kd, []).append(v)
        else: r = y
        for a in range(len(cls)):
            for i in range(D):
                if ok[a * D + i]: res_by_t[ty[i]].append(r[a * D + i] ** 2)
        v = np.full(K * nk, np.nan)
        for a, c in enumerate(cls):
            for i, k in enumerate(tk):
                if k in kix: v[c * nk + kix[k]] = r[a * D + i]
        RV[t['id']] = v
    S = dict(var={kd: float(np.mean(np.square(v))) for kd, v in coef.items()}, e0=np.array([np.mean(res_by_t[q]) for q in range(3)]))
    if cand.get('ppca'):
        R = np.array([RV[i] for i in P0['idf']]); S['pp'] = ppca(R, P0['kt'], cand.get('r'))
    return S


def fit_joint(cand, P0, A, gs):
    """'joint' candidates: hpj-style full-rank LW prior, optionally re-estimated on archive residuals after the weather term
    (cond), with prior mean m_joint + w(ctx); optional extra variance along basis directions (addB)."""
    S = dict(jm=P0['jm'], jc=P0['jc'])
    if cand.get('cond'):
        from sklearn.covariance import LedoitWolf
        keys = P0['keys']; kix = {k: i for i, k in enumerate(keys)}; nk = len(keys); V = []
        for t in A:
            r = np.concatenate([t['orc'][c] for c in t['cls']]) - cand['wfun'](t, gs); D = len(t['tk']); v = np.full(K * nk, np.nan)
            for a, c in enumerate(t['cls']):
                for i, k in enumerate(t['tk']):
                    if k in kix: v[c * nk + kix[k]] = r[a * D + i]
            V.append(v)
        V = np.array(V); V = V[~np.isnan(V).any(1)]; S['jm'] = V.mean(0); S['jc'] = LedoitWolf().fit(V).covariance_; S['n_complete'] = len(V)
    if cand.get('basis'):
        sub = dict(cand, mean=lambda t, g: joint_for(P0['keys'], S['jm'], S['jc'], t['cls'], t['tk'])[0] + cand['wfun'](t, g), ppca=False)
        S['var'] = fit_scales(sub, P0, A, gs)['var']
    return S


def joint_prior(cand, ctx, gs, S, P0):
    m, Psi = joint_for(P0['keys'], S['jm'], S['jc'], ctx['cls'], ctx['tk']); m = m + cand['wfun'](ctx, gs)
    if cand.get('basis'):
        B, kinds = cand['basis'](ctx, gs); Psi = Psi + (B * np.array([S['var'][k] for k in kinds])) @ B.T
    return m, Psi


def task_prior(cand, ctx, gs, S, P0):
    cls, tk, ty = ctx['cls'], ctx['tk'], ctx['ty']; C = len(cls)
    B, kinds = cand['basis'](ctx, gs); s = np.array([S['var'][k] for k in kinds])
    if cand.get('ppca'):
        U, e, r, _ = S['pp']; Ut, et = map_lowrank(P0['keys'], U, e, S['e0'], cls, tk); B = np.hstack([B, Ut]); s = np.concatenate([s, np.ones(Ut.shape[1])])
    else: et = np.tile(S['e0'][ty], C)
    return cand['mean'](ctx, gs), woodbury(B, s, et)


# ---------------- transfers ----------------
def transfers(G, which):
    site, trk = G[0], int(G[1:]); M = meta(G); years = [y for y in DEV if f'{y}_T{trk}' in M]
    pr = list(itertools.permutations(years, 2))
    if which not in ('search', 'confirm', 'smoke'): raise ValueError(which)
    if which == 'confirm' and os.environ.get('CONFIRM_OK') != 'frozen': raise RuntimeError('confirm split only after freezing (PROTOCOL §4)')
    out = [p for i, p in enumerate(pr) if (i % 3 == 0) == (which != 'confirm')]          # smoke = search subset
    if which == 'smoke': out = out[:1]
    return out[:int(os.environ['LIMIT'])] if os.environ.get('LIMIT') else out


def run(cands, which='search', tag=''):
    t0 = time.time()
    allr = pd.concat([pd.read_parquet(os.path.join(UP, f'p1_rows_{g}.parquet')) for g in GROUPS], ignore_index=True)
    for trs in ('B', 'A'):
        P0, A = archive(trs, allr)
        fitted = {}
        for cd in cands:
            if cd['name'] in ('em_cov01', 'hpj_cov01'): continue
            gs = cd['fit'](A, P0) if cd.get('fit') else {}
            fitted[cd['name']] = (gs, fit_joint(cd, P0, A, gs) if cd.get('joint') else fit_scales(cd, P0, A, gs))
            info = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in gs.items() if not callable(v)}
            print('fit', trs, cd['name'], json.dumps(info, default=str)[:300], 'r', fitted[cd['name']][1].get('pp', [0, 0, None])[2], round(time.time() - t0), 's', flush=True)
        for G in [g for g in GROUPS if g[0] != trs]:
            site, trk, Z, M = load(G); rows = []
            for ya, yb in transfers(G, which):
                S_ = tuple(sorted(set(M[f'{ya}_T{trk}']['slots']) & set(M[f'{yb}_T{trk}']['slots'])))
                vva, vha, coa, Ya, Fa = data(Z, M, trk, ya, S_); vvb, vhb, cob, Yb, Fb = data(Z, M, trk, yb, S_)
                Ea = impute(np.hstack([vva, vha, coa]), np.hstack([vva, vha, coa])); Eb = impute(np.hstack([vvb, vhb, cob]), np.hstack([vva, vha, coa]))
                ty, sl, sub = dim_meta(S_); tk = list(zip(ty, sl, sub)); D = Ea.shape[1]
                doy = np.array(dim_dates(M, trk, ya, S_, ty, sl, sub).dayofyear)
                pred = {cd['name']: np.full(len(Yb), -1) for cd in cands}
                for k in range(6):
                    trm = (Fa[:, k] == 'TRAIN') & ~xydrop(G, ya, yb, k, len(Fa)); tem = Fb[:, k] == 'TEST'
                    if not tem.sum(): continue
                    mu, prec, logdet, cls = gauss_fit(Ea[trm], Ya[trm]); Et = Eb[tem]; C = len(cls)
                    ctx = make_ctx(G, M, trk, ya, yb, cls, mu, ty, sl, sub, tk, doy, P0)
                    for cd in cands:
                        n = cd['name']
                        if n == 'em_cov01': lab, _ = gen_em2(Et, mu, prec, logdet, cls, None, 0.1)
                        elif n == 'hpj_cov01':
                            m, Psi = joint_for(P0['keys'], P0['jm'], P0['jc'], cls, tk); lab, _ = gen_em2(Et, mu, prec, logdet, cls, (m, Psi), 0.1)
                        elif cd.get('joint'):
                            gs, S = fitted[n]; m, Psi = joint_prior(cd, ctx, gs, S, P0); lab, _ = gen_em2(Et, mu, prec, logdet, cls, (m, Psi), 0.1)
                        else:
                            gs, S = fitted[n]; m, Pi = task_prior(cd, ctx, gs, S, P0)
                            lab, _ = gen_em2(Et, mu, prec, logdet, cls, (m, np.zeros((C * D, C * D))), 0.1, Pinv=Pi)
                        pred[n][tem] = lab
                ba = {n: balanced_accuracy_score(Yb[p >= 0], p[p >= 0]) for n, p in pred.items()}
                for n, v in ba.items(): rows.append(dict(cand=n, group=G, src=ya, tgt=yb, split=which, ba=v, tag=tag))
                print(G, ya, '->', yb, {n: round(v, 3) for n, v in ba.items()}, round(time.time() - t0), 's', flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, f'ba_{tag}_{which}_{G}.csv'), index=False)
    print('DONE', tag, which, round(time.time() - t0), 's')


if __name__ == '__main__':
    mod = importlib.import_module(sys.argv[1]); which = sys.argv[2] if len(sys.argv) > 2 else 'search'
    run(mod.CANDS, which, sys.argv[1])
