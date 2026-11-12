"""Table: comparison with deep and recent (2024-2025) unsupervised adaptation methods. Source: ../../s3d_main_table.csv (s3d_table.py).
Proposed = TempCNN trained on the season resynthesised with the frozen-method shifts (one version for every column; user 2026-11-10)."""
import os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, '..', '..', 's3d_main_table.csv'))
LAB = {'No adaptation (TempCNN)': 'No adaptation', 'AdaBN': r'AdaBN \cite{Li2018adabn}', 'DANN': r'DANN \cite{Ganin2016}',
       'SFUDA (RSE 2024)': r'SFUDA \cite{Mohammadi2024}', 'RAINCOAT-SRS (ISPRS 2025), rule A': r'RAINCOAT-SRS$^{\ddagger}$ \cite{Gackstetter2025}',
       'Proposed (TempCNN)': r'\textbf{Proposed (TempCNN)}', 'Proposed + DANN': r'\textbf{Proposed + DANN}'}
COLS = ['NL site B', 'NL site A', 'France']
n = {c: int(S[f'{c} n'].iloc[0]) for c in COLS}
best = {c: S[S.method.isin(LAB)][f'{c} median'].max() for c in COLS}
rows = []
for _, r in S.iterrows():
    if r.method not in LAB: continue
    cells = []
    for c in COLS:
        v = r[f'{c} median']; ba = f'\\textbf{{{v:.3f}}}' if abs(v - best[c]) < 5e-4 else f'{v:.3f}'
        dv = r[f'{c} d_med']; dv = 0.0 if not pd.isna(dv) and abs(dv) < 5e-4 else dv
        d = '--' if pd.isna(dv) else (f"$0.000$ ({100 * r[f'{c} win']:.0f}\\%)" if dv == 0 else f"${dv:+.3f}$ ({100 * r[f'{c} win']:.0f}\\%)")
        cells += [ba, d]
    rows.append(LAB[r.method] + ' & ' + ' & '.join(cells) + r'\\')
tex = r"""\begin{table*}[!t]
\caption{Comparison with deep and recent unsupervised adaptation methods. Median balanced accuracy over transfers and paired median
difference of the proposed method to each method (share of transfers in which the proposed method is better). All methods except
RAINCOAT-SRS use the same temporal CNN, normalisation, folds and 400-m exclusion; the proposed method trains it on the source season
shifted by the estimated shifts. Dutch sites: every other ordered season pair (%d and %d transfers); France: all %d.
$^{\ddagger}$Self-attention encoder of \cite{Gackstetter2025} on the official RAINCOAT code; checkpoint selected on held-out source parcels.}
\label{tab:sota}
\centering\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{Site B} & \multicolumn{2}{c}{Site A} & \multicolumn{2}{c}{Pays de Caux}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Method & BA & $\Delta$ & BA & $\Delta$ & BA & $\Delta$\\
\midrule
""" % (n['NL site B'], n['NL site A'], n['France'])
tex += '\n'.join(rows[:5]) + '\n\\midrule\n' + '\n'.join(rows[5:]) + '\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
open(os.path.join(H, 'tab_sota.tex'), 'w', encoding='utf-8').write(tex); print(tex)
