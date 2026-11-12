"""Supplementary tables generated from result files (traceability):
S-I  structure ablation on the Dutch sites (P9_SUMMARY.json + P10: median BA and paired difference to the proposed prior);
S-II recent deep adaptation methods with official code and uniform tuning (S3C_SUMMARY.json, s3c_tune.csv);
S-III France: local three-season archive and track-restricted archive (CONFIRM_FR_SUMMARY.json);
S-IV black-box amortised regression (P1, p1_summary/FINAL_SUMMARY.json)."""
import os, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE))
J = lambda p: json.load(open(os.path.join(UP, p), encoding='utf-8')); D = ('testB88_priorA', 'testA_priorB'); G = {'testB88_priorA': ['B88'], 'testA_priorB': ['A37', 'A88']}
f3 = lambda x: f'{x:.3f}'
K = ['group', 'src', 'tgt']
T = pd.concat([pd.read_csv(os.path.join(UP, f'p9_basis_{g}.csv')) for g in ('A37', 'A88', 'B88')]).merge(
    pd.concat([pd.read_csv(os.path.join(UP, f'p10_main_{g}.csv')) for g in ('A37', 'A88', 'B88')])[K + ['k8_all|hybridD_lw']], on=K)
def cell(d, c):
    x = T[T.group.isin(G[d])]; z = (x['k8_all|hybridD_lw'] - x[c]).dropna()
    return f"{x[c].median():.3f} & ${z.median():+.3f}$ ({100 * (z > 0).mean():.0f}\\%)"
rows = [('Prior-free mean-shift EM', 'em_cov01'), ('Hand physics basis, nominal scales (no archive)', 'phys_nom'), ('Hand physics basis, archive scales', 'phys_arch'),
        ('Learned modes only (PPCA, $r^*$)', 'learn_r*'), ('Hand physics + learned modes', 'hybrid_r*'), ('Date-shared basis, archive scales', 'physD_arch'),
        ('Date-shared basis + low-rank residual', 'hybridD_r*'), ('Date-shared basis + low-rank residual + rain mean', 'hybridD_r*_mw'),
        ('MLLR-type shared affine shift', 'mllr_cov01'), ('Empirical full-rank prior', 'hpj_cov01')]
L = ['\\begin{table*}[!t]', '\\caption{Structure ablation (Dutch sites, all transfers, full archive of the other site). Median BA and paired median difference of the proposed prior (date-shared basis + full-rank residual) to each variant (share of transfers in which the proposed prior is better).}',
     '\\label{tab:s_ablation}', '\\centering\\footnotesize', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule', ' & \\multicolumn{2}{c}{Test site B} & \\multicolumn{2}{c}{Test site A}\\\\',
     '\\cmidrule(lr){2-3}\\cmidrule(l){4-5}', 'Prior & BA & $\\Delta$ & BA & $\\Delta$\\\\', '\\midrule']
for n, c in rows: L.append(f'{n} & {cell(D[0], c)} & {cell(D[1], c)}\\\\')
L += ['\\midrule', f"\\textbf{{Proposed}} & {T[T.group.isin(G[D[0]])]['k8_all|hybridD_lw'].median():.3f} & -- & {T[T.group.isin(G[D[1]])]['k8_all|hybridD_lw'].median():.3f} & --\\\\", '\\bottomrule', '\\end{tabular}', '\\end{table*}']
open(os.path.join(HERE, 'tab_s_ablation.tex'), 'w', encoding='utf-8').write('\n'.join(L))
S = J('S3C_SUMMARY.json'); tune = pd.read_csv(os.path.join(UP, 's3c_tune.csv')); mt = tune.drop(columns=K).mean()
L = ['\\begin{table}[!t]', f"\\caption{{Recent adaptation methods run with the official code (RAINCOAT) or the published algorithm (SFUDA), learning rate tuned uniformly on four fixed development transfers (RAINCOAT {S['lr']['raincoat']:g}, SFUDA {S['lr']['sfuda']:g}); every other transfer (84). Median BA.}}",
     '\\label{tab:s_recent}', '\\centering\\footnotesize', '\\begin{tabular}{@{}lcc@{}}', '\\toprule', 'Method & Test site B & Test site A\\\\', '\\midrule']
for n, c in (('TempCNN, no adaptation', 'dl_raw'), ('RAINCOAT \\cite{He2023}', 'raincoat_off'), ('SFUDA \\cite{Mohammadi2024}', 'sfuda_tuned'), ('AdaBN', 'dl_adabn'), ('DANN', 'dl_dann'), ('Proposed', 'proposed')):
    L.append(f"{n} & {f3(S[D[0]]['median_ba'][c])} & {f3(S[D[1]]['median_ba'][c])}\\\\")
L += ['\\midrule', 'Tuning grid mean BA, RAINCOAT & \\multicolumn{2}{c}{' + ' / '.join(f"{mt[f'rc_{x}']:.3f}" for x in (0.0005, 0.001, 0.002)) + '}\\\\',
      'Tuning grid mean BA, SFUDA & \\multicolumn{2}{c}{' + ' / '.join(f"{mt[f'sf_{x}']:.3f}" for x in (0.001, 0.01, 0.05)) + '}\\\\', '\\bottomrule', '\\end{tabular}', '\\end{table}']
open(os.path.join(HERE, 'tab_s_recent_s3c_old.tex'), 'w', encoding='utf-8').write('\n'.join(L))
F = J('newregion/CONFIRM_FR_SUMMARY.json'); r = F['report']
def pr(x): return f"${x['median']:+.4f}$ ({100 * x['frac_pos']:.0f}\\%, $n={x['n']}$)"
L = ['\\begin{table}[!t]', '\\caption{France: secondary analyses registered with the confirmation. Paired median difference (share positive, number of transfers).}', '\\label{tab:s_fr}',
     '\\centering\\footnotesize', '\\begin{tabular}{@{}p{4.6cm}l@{}}', '\\toprule', 'Comparison & Result\\\\', '\\midrule',
     f"Local 3-season archive vs.\\ Dutch archive (proposed) & {pr(r['N5 lw_local3 vs lw_all'])}\\\\",
     f"Local 3-season archive vs.\\ Dutch archive (empirical) & {pr(r['N5 hpj_local3 vs hpj_cov01'])}\\\\",
     f"Proposed vs.\\ empirical, both on the local 3-season archive & {pr(r['N5 lw_local3 vs hpj_local3'])}\\\\",
     f"Ascending-track (T88) Dutch archive vs.\\ full Dutch archive & {pr(r['secondary lw_T88 vs lw_all'])}\\\\",
     f"Weather-conditioned prior vs.\\ proposed & {pr(r['N6b j2 vs lw_all'])}\\\\", '\\bottomrule', '\\end{tabular}', '\\end{table}']
open(os.path.join(HERE, 'tab_s_fr.tex'), 'w', encoding='utf-8').write('\n'.join(L)); print('supp tables written')
