"""FR-A pipeline smoke check (single year; no cross-year result is computed):
(1) class-mean VV/VH/COH trajectories of FR 2023 vs NL 2024 T88 (same slot calendar): Pearson r per class and layer;
(2) within-year 6-fold Gaussian (frozen gauss_fit, LW) BA on FR 2023 - pipeline sanity only (NL within-year reference ~0.81-0.88);
(3) coherence dates = 12-day pair, key order = slot order. -> SMOKE_CHECK_2023.json"""
import os, sys, json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(UP, 'frozen'))
import hybridD_lw as F
Zf = np.load(r'D:\sar_data\fr_a\features_FR.npz', allow_pickle=True); Mf = json.load(open(r'D:\sar_data\fr_a\features_FR_meta.json'))
Zn = np.load(r'D:\sar_data\multiyear\features_B.npz', allow_pickle=True); Mn = json.load(open(r'D:\sar_data\multiyear\features_B_meta.json'))
CL = F.CLASSES; out = {}
tf, tn = '2023_T59', '2024_T88'
def cm(Z, t):
    y = Z[f'{t}_y']; return {c: {l: np.nanmean(Z[f'{t}_{l}'][y == c], 0) for l in ('VV', 'VH', 'COH')} for c in range(7) if (y == c).sum() >= 20}
a, b = cm(Zf, tf), cm(Zn, tn)
out['traj_r'] = {CL[c]: {l: round(float(np.corrcoef(a[c][l], b[c][l])[0, 1]), 3) for l in ('VV', 'VH', 'COH')} for c in a if c in b}
out['class_mean_db_FR'] = {CL[c]: {l: round(float(np.mean(a[c][l])), 2) for l in ('VV', 'VH')} for c in a}
out['class_mean_db_NL2024B'] = {CL[c]: {l: round(float(np.mean(b[c][l])), 2) for l in ('VV', 'VH')} for c in b}
E = np.hstack([Zf[f'{tf}_VV'], Zf[f'{tf}_VH'], Zf[f'{tf}_COH']]).astype(float); y = Zf[f'{tf}_y']; Fo = Zf[f'{tf}_folds']; pred = np.full(len(y), -1)
for k in range(6):
    tr, te = Fo[:, k] == 'TRAIN', Fo[:, k] == 'TEST'
    mu, prec, logdet, cls = F.gauss_fit(E[tr], y[tr])
    L = np.stack([-0.5 * logdet[c] - 0.5 * np.einsum('ij,jk,ik->i', E[te] - mu[c], prec[c], E[te] - mu[c]) for c in cls], 1); pred[te] = np.array(cls)[L.argmax(1)]
out['within_year_gauss_BA'] = round(float(balanced_accuracy_score(y[pred >= 0], pred[pred >= 0])), 4)
out['per_class_recall'] = {CL[c]: round(float((pred[y == c] == c).mean()), 3) for c in range(7) if (y == c).sum()}
d = [np.datetime64(x) for x in Mf[tf]['dates']]; out['pair_gaps_days'] = [int((d[2 * i + 1] - d[2 * i]).astype(int)) for i in range(len(d) // 2)]
out['keys'] = Mf[tf]['keys']; out['n'] = int(len(y)); out['class_counts'] = np.bincount(y, minlength=7).tolist()
json.dump(out, open(os.path.join(os.path.dirname(HERE), 'FR-A_PaysDeCaux', 'SMOKE_CHECK_2023.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
