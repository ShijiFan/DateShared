"""Supplementary Table SII: RAINCOAT-SRS under the four checkpoint rules, and ClimID-UDA, against the proposed method.
Sources: ../../s3d_{B88,A37,A88,FR}.csv (S3d), ../../climid/climid_*.csv (ClimID), ../../s3j_deep_*.csv (S3j: archive without the transfer's years), ../../newregion/FR_DEEP_rows.csv,
../../s3d_tune_parts.csv + s3d_tune.csv (tuning grid). Same transfers as Table tab:sota."""
import os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(H, '..', '..'); R = lambda *p: pd.read_csv(os.path.join(U, *p))
REG = {'Site B': ['B88'], 'Site A': ['A37', 'A88'], 'Pays de Caux': ['FR']}
def load(gs):
    out = []
    for g in gs:
        d = R(f's3d_{g}.csv').drop_duplicates(['src', 'tgt']); c = R('climid', f'climid_{g}.csv')[['src', 'tgt', 'hgb_raw', 'hgb_climid', 'tcn_raw', 'tcn_climid']]
        p = R('newregion', 'FR_DEEP_rows.csv') if g == 'FR' else R(f's3j_deep_{g}.csv').drop_duplicates(['src', 'tgt']).rename(columns={'tcn_lw_x': 'tcn_lw'})
        out.append(d.merge(c, on=['src', 'tgt']).merge(p[['src', 'tgt', 'tcn_lw']], on=['src', 'tgt']))
    return pd.concat(out)
T = {k: load(v) for k, v in REG.items()}
ROWS = [('RAINCOAT-SRS, rule A: held-out source parcels (main text)', 'srs_A'), ('RAINCOAT-SRS, rule B: last epoch', 'srs_B'),
        ('RAINCOAT-SRS, rule C: 2000 labelled target-season parcels', 'srs_C'), ('RAINCOAT-SRS, rule D: target test labels (official code)', 'srs_D'),
        ('ClimID-UDA, HGB: none / +ClimID', ('hgb_raw', 'hgb_climid')), ('ClimID-UDA, TempCNN: none / +ClimID', ('tcn_raw', 'tcn_climid')),
        ('Proposed (TempCNN)', 'tcn_lw')]
lines = []
for lab, k in ROWS:
    cells = []
    for r, d in T.items():
        if isinstance(k, tuple): cells.append(f'{d[k[0]].median():.3f} / {d[k[1]].median():.3f}')
        else:
            dd = d['tcn_lw'] - d[k]; cells.append(f'{d[k].median():.3f}' + ('' if k == 'tcn_lw' else f' (${dd.median():+.3f}$)'))
    lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
tune = pd.concat([R('s3d_tune_parts.csv')[['lr', 'srsA']],
                  pd.DataFrame([dict(lr=x, srsA=v) for _, t in R('s3d_tune.csv').iterrows() for x, v in ((5e-4, t['srsA_0.0005']), (1e-3, t['srsA_0.001']), (2e-3, t['srsA_0.002']))])])
tm = tune.groupby('lr').srsA.mean()
tex = r"""\begin{table*}[!t]
\caption{RAINCOAT-SRS under four checkpoint rules fixed before the run, and ClimID-UDA ported to Sentinel-1, on the transfers of the main
comparison. Median BA; in brackets the paired median difference of the proposed method. Rule A is used in the main text; rule D reads the
target test labels at every epoch, as the official RAINCOAT trainer does, and is an upper bound for that method. Learning rate of
RAINCOAT-SRS selected by rule A on four development transfers (mean BA %.3f / %.3f / %.3f for $5\times10^{-4}$ / $10^{-3}$ / $2\times10^{-3}$).}
\label{tab:s_recent}
\centering\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule
Method & Site B & Site A & Pays de Caux\\
\midrule
""" % (tm[5e-4], tm[1e-3], tm[2e-3])
tex += '\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
open(os.path.join(H, 'tab_s_recent.tex'), 'w', encoding='utf-8').write(tex); print(tex)
