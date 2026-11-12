"""Generate LaTeX tables for the TGRS manuscript directly from result files (traceability): Table I (NL, Gaussian framework, median BA and
paired differences), Table II (classifier plug-in: HGB, RF, TempCNN, DANN), Table III (France confirmation N1-N6).
2026-11-12 (S3j): Tables I and II read results/EXCL.json (results/excl_tables.py): archive-based entries recomputed with the archive WITHOUT
the transfer's source and target seasons (s3j_cpu_*, s3j_deep_*); archive-free entries unchanged from the original files.
Sources: results/EXCL.json, newregion/CONFIRM_FR_SUMMARY.json."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE))
J = lambda p: json.load(open(os.path.join(UP, p), encoding='utf-8'))
EX, FR = J('results/EXCL.json'), J('newregion/CONFIRM_FR_SUMMARY.json')
D = ('testB88_priorA', 'testA_priorB')
f3 = lambda x: f'{x:.3f}'
def dd(r): return f"${r['median']:+.3f}$ ({100 * r['frac_pos']:.0f}\\%)"

# ---- Table I ----
rows = [('No adaptation (TempCNN)', 'dl_raw'), ('AdaBN (TempCNN)', 'dl_adabn'), ('DANN (TempCNN)', 'dl_dann'),
        ('Mean-shift EM + cov.\\ update', 'em_cov01'), ('MLLR-type shared affine shift', 'mllr_cov01'),
        ('Empirical full-rank prior (EMAP-type)', 'hpj_cov01_x'), ('Date-shared basis + low-rank residual', 'hybridD_r*_x'),
        ('\\textbf{Date-shared prior (proposed)}', 'hybridD_lw_x')]
L = ['\\begin{table*}[!t]', '\\caption{Cross-year transfer on the two Dutch sites (all 168 ordered year pairs; archive from the other site without the source and target years). Median balanced accuracy (BA) and paired median difference to the proposed method (share of transfers in which the proposed method is better).}',
     '\\label{tab:nl}', '\\centering\\footnotesize', '\\begin{tabular}{@{}lcccc@{}}', '\\toprule',
     ' & \\multicolumn{2}{c}{Test site B} & \\multicolumn{2}{c}{Test site A}\\\\', '\\cmidrule(lr){2-3}\\cmidrule(l){4-5}', 'Method & BA & $\\Delta$ & BA & $\\Delta$\\\\', '\\midrule']
for name, g in rows:
    cells = []
    for d in D:
        r = EX[d]['tab_nl'][g]; cells += [f3(r['median_ba']), '--' if r['diff'] is None else dd(r['diff'])]
    L.append(f"{name} & " + ' & '.join(cells) + '\\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
open(os.path.join(HERE, 'tab_nl.tex'), 'w', encoding='utf-8').write('\n'.join(L))

# ---- Table II (classifier plug-in, NL) ----
P = {d: EX[d]['plugin'] for d in D}
R = [('HGB', 'hgb_raw', 'hgb_em', 'hgb_lw_x'), ('RF$^{\\dagger}$', 'rf_raw', 'rf_em', 'rf_lw_x'), ('TempCNN', 'dl_raw', 'tcn_em', 'tcn_lw_x'),
     ('DANN (TempCNN)', 'dl_dann', None, 'dann_lw_x')]
L = ['\\begin{table*}[!t]', '\\caption{The estimated shifts as a classifier-agnostic plug-in (Dutch sites; archive without the source and target years). Median BA when the source season is left unchanged, shifted by the prior-free EM estimate, or shifted by the proposed estimate before training. $^{\\dagger}$Every other transfer (84).}',
     '\\label{tab:plugin}', '\\centering\\footnotesize', '\\begin{tabular}{@{}lcccccc@{}}', '\\toprule',
     ' & \\multicolumn{3}{c}{Test site B} & \\multicolumn{3}{c}{Test site A}\\\\', '\\cmidrule(lr){2-4}\\cmidrule(l){5-7}',
     'Classifier & none & EM & proposed & none & EM & proposed\\\\', '\\midrule']
for name, a, b, c in R:
    L.append(f'{name} & ' + ' & '.join([f3(P[d][a]) for d in D[:1]] + [f3(P[D[0]][b]) if b else '--', f3(P[D[0]][c]), f3(P[D[1]][a]),
                                        f3(P[D[1]][b]) if b else '--', f3(P[D[1]][c])]) + '\\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
open(os.path.join(HERE, 'tab_plugin.tex'), 'w', encoding='utf-8').write('\n'.join(L))

# ---- Table III (France) ----
c, r, m = FR['criteria'], FR['report'], FR['median_ba']
def ci(x): return f"${x['median']:+.4f}$ ({100 * x['frac_pos']:.0f}\\%) [${x['boot90'][0]:+.3f}$, ${x['boot90'][1]:+.3f}$]"
L = ['\\begin{table}[!t]', '\\caption{Independent confirmation in Pays de Caux, France (frozen method and Dutch archive; 56 ordered year pairs of 2017--2024). Paired median difference (share positive) [90\\% year-block bootstrap interval]; criteria fixed before the run.}',
     '\\label{tab:fr}', '\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}p{2.8cm}p{3.7cm}l@{}}', '\\toprule', 'Comparison & Result & Met\\\\', '\\midrule',
     f"Proposed vs.\\ mean-shift EM & {ci(c['N1 lw_all vs em_cov01 (median>=+0.01 & >=2/3 pos)']['res'])} & yes\\\\",
     f"Proposed vs.\\ empirical full-rank prior & {ci(c['N2 lw_all vs hpj_cov01 (median>=-0.003)']['res'])} & yes\\\\",
     f"HGB: proposed vs.\\ EM shifts & {ci(c['N3 hgb_lw vs hgb_em (median>=+0.01 & >=2/3 pos)']['res'])} & yes\\\\",
     f"Weather-conditioned prior vs.\\ empirical prior & {ci(c['N6 j2 vs hpj_cov01']['res'])} & {c['N6 j2 vs hpj_cov01']['grade']}\\\\",
     '\\midrule', f"Median BA: none / EM / empirical / proposed & {f3(m['g_raw'])} / {f3(m['em_cov01'])} / {f3(m['hpj_cov01'])} / {f3(m['lw_all'])} & \\\\",
     f"Spearman($w_t$, $\\Delta$API) & {r['N6c spearman(date coef, dAPI)']['rho']:.2f} ($n={r['N6c spearman(date coef, dAPI)']['n']:,}$) & \\\\",
     '\\bottomrule', '\\end{tabular}', '\\end{table}']
open(os.path.join(HERE, 'tab_fr.tex'), 'w', encoding='utf-8').write('\n'.join(L))
print('tables written')
