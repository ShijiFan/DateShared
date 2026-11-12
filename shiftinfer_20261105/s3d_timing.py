"""S3e timing (EXPLORE_S3D_Q1_ZH.md, S3e): wall-clock cost per fold of the proposed method vs deep / recent baselines. B88 2019->2020, fold 0."""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 's3d_q1.py'), encoding='utf-8').read().split("if MODE == 'smoke':")[0]
sys.argv = ['s3d_q1.py', 'timing']; exec(compile(src, 's3d_q1.py(defs)', 'exec'))
sys.path.insert(0, os.path.join(HERE, 'frozen')); import hybridD_lw as F
G = NLData('B88'); D = G.D; ya, yb, k = 2019, 2020, 0


def tk_doy(ya, S):        # = p11d_deep.tk_doy
    m = D.M[f'{ya}_T{D.TRK}']; dt = pd.to_datetime(m['dates']); tk, doy = [], []
    for ty in (0, 1):
        for s in S:
            p = m['slots'].index(s)
            for u in (0, 1): tk.append((ty, s, u)); doy.append(dt[2 * p + u].dayofyear)
    for s in S:
        p = m['slots'].index(s); tk.append((2, s, 0)); doy.append((dt[2 * p] + (dt[2 * p + 1] - dt[2 * p]) / 2).dayofyear)
    return tk, np.array(doy)


def flat(X): return np.hstack([X[:, 0], X[:, 1], X[:, 3][:, ::2]]).astype(float)      # = p11d_deep.flat


def shift_seq(X, y, delta, S):    # = p11d_deep.shift_seq
    X = X.copy(); n2 = 2 * len(S)
    for c, d in delta.items():
        m = y == c; X[m, 0] += d[:n2]; X[m, 1] += d[n2:2 * n2]; X[m, 3] += np.repeat(d[2 * n2:], 2)
    X[:, 2] = X[:, 1] - X[:, 0]; return X.astype(np.float32)


ar = F.Archive.load(os.path.join(HERE, 'frozen', 'archive_NL_A.npz'))
S = tuple(sorted(set(G.slots(ya)) & set(G.slots(yb)))); Xa, Ya, Fa = G.data(ya, S); Xb, Yb, Fb = G.data(yb, S); tkk, doy = tk_doy(ya, S)
tr = (Fa[:, k] == 'TRAIN') & ~G.drop(ya, yb, k, len(Fa)); te = Fb[:, k] == 'TEST'
sync = lambda: torch.cuda.synchronize() if torch.cuda.is_available() else None
R = dict(transfer=f'B88 {ya}->{yb} fold {k}', n_source=int(tr.sum()), n_target=int(te.sum()), D=int(flat(Xa[:1]).shape[1]), device=str(DEV),
         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
t = time.perf_counter()
Ea_all = flat(Xa); Ea = F.impute(Ea_all, Ea_all); Eb = F.impute(flat(Xb), Ea_all)
mu, prec, logdet, cls = F.gauss_fit(Ea[tr], Ya[tr]); m, Psi = ar.prior(cls, mu, tkk, doy); lab, d_lw = F.map_em(Eb[te], mu, prec, logdet, cls, m, Psi)
R['proposed_mapem_s'] = time.perf_counter() - t; R['proposed_mapem_ba'] = D.bal_acc(Yb[te], lab)
Xs_raw = shift_seq(np.nan_to_num(Xa[tr]), Ya[tr], d_lw, S); mu_ = Xs_raw.mean(axis=(0, 2), keepdims=True); sd_ = Xs_raw.std(axis=(0, 2), keepdims=True) + 1e-6
nrm = lambda X: np.nan_to_num((X - mu_) / sd_).astype(np.float32)
for name, dann in (('resynth_tempcnn_s', False), ('resynth_dann_s', True)):
    sync(); t = time.perf_counter(); net, XT = D.train(nrm(Xs_raw), Ya[tr], nrm(Xb[te]), dann, D.SEED + k); D.predict(net, XT); sync(); R[name] = time.perf_counter() - t
mu0 = np.nanmean(Xa[tr], axis=(0, 2), keepdims=True); sd0 = np.nanstd(Xa[tr], axis=(0, 2), keepdims=True) + 1e-6
norm = lambda X: np.nan_to_num((X - mu0) / sd0).astype(np.float32)
sync(); t = time.perf_counter(); net, XT0 = D.train(norm(Xa[tr]), Ya[tr], norm(Xb[te]), False, D.SEED + k); sync(); R['tempcnn_source_s'] = time.perf_counter() - t
sync(); t = time.perf_counter(); net, XT0 = D.train(norm(Xa[tr]), Ya[tr], norm(Xb[te]), True, D.SEED + k); sync(); R['dann_s'] = time.perf_counter() - t
sync(); t = time.perf_counter(); net, XT0 = D.train(norm(Xa[tr]), Ya[tr], norm(Xb[te]), False, D.SEED + k); sfuda(net, XT0, D.SEED + k, lr=1e-3); sync(); R['sfuda_s'] = time.perf_counter() - t
rng = np.random.default_rng(D.SEED + k); itr = np.flatnonzero(tr); hold = np.zeros(len(Fa), bool); hold[rng.choice(itr, len(itr) // 5, replace=False)] = True; fit = tr & ~hold
seed = D.SEED + k; torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); alg = srs_model(Xa.shape[2], 5e-4)
XS = torch.tensor(norm(Xa[fit]), device=DEV); YS = torch.tensor(Ya[fit], device=DEV); XV = torch.tensor(norm(Xa[hold]), device=DEV); XT = torch.tensor(norm(Xb[te]), device=DEV)
nb = max(1, len(XS) // 128); sync(); t = time.perf_counter(); best = (-1, None)
for phase in ('update', 'correct'):
    for _ in range(40):
        alg.train(); perm = torch.randperm(len(XS), generator=g).to(DEV)
        for i in range(nb):
            j = perm[i * 128:(i + 1) * 128]; jt = torch.randint(0, len(XT), (len(j),), generator=g).to(DEV); getattr(alg, phase)(XS[j], YS[j], XT[jt])
        bv = D.bal_acc(Ya[hold], predict(alg, XV))
        if bv >= best[0]: best = (bv, predict(alg, XT))
sync(); R['raincoat_srs_s'] = time.perf_counter() - t
print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in R.items()}, indent=1))
json.dump(R, open(os.path.join(HERE, 's3d_timing.json'), 'w'), indent=1)
