"""Table IV (rain-driven prior, Section res_physics) generated from result files only (traceability):
- archive-length rows: toy_B/full/runs/curve_<G>.csv (j2) merged with p10_curve_<G>.csv (hpj_cov01, hybridD_lw) on the same subsets
  (k3_0/k3_1, k5_0/k5_1, k8_all; L=3 and L=5 average the two subsets per transfer, as curve_summary.py) and transfers;
- NL confirmation split: toy_B/full/runs/ba_cands_final_confirm_<G>.csv (111 transfers, leave-site-out archive) and p10_main_<G>.csv;
- France: newregion/CONFIRM_FR_rows.csv (N6, N6b).
Also prints the numbers quoted in the text (fitted parameters from frozen_j2/VERIFY_J2.json and toy_B/full/runs/r3.log)."""
import os, json, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE)); FULL = os.path.join(UP, 'toy_B', 'full')
GR = {'B': ['B88'], 'A': ['A37', 'A88']}; L = {3: ('k3_0', 'k3_1'), 5: ('k5_0', 'k5_1'), 8: ('k8_all',)}
J2, HPJ, LW = 'j2_hpj_wc_cond', 'hpj_cov01', 'hybridD_lw'


def stat(x):
    x = pd.Series(x).dropna(); return float(x.median()), float((x > 0).mean()), len(x)


def cell(s): return f"${'0.000' if abs(s[0]) < 0.0005 else f'{s[0]:+.3f}'}$ ({100 * s[1]:.0f}\\%)"


rows = {}
for site, gs in GR.items():
    a = pd.concat([pd.read_csv(os.path.join(FULL, 'runs', f'curve_{g}.csv')).assign(group=g) for g in gs])
    b = pd.concat([pd.read_csv(os.path.join(UP, f'p10_curve_{g}.csv')) for g in gs])
    m = a.merge(b, on=['group', 'src', 'tgt'], suffixes=('', '_p10'))
    for l, subs in L.items():
        j2 = np.mean([m[f'{s}|{J2}'] for s in subs], 0); hp = np.mean([m[f'{s}|{HPJ}_p10' if f'{s}|{HPJ}_p10' in m else f'{s}|{HPJ}'] for s in subs], 0)
        lw = np.mean([m[f'{s}|{LW}'] for s in subs], 0)
        rows[(site, f'{l} seasons')] = (stat(j2 - hp), stat(j2 - lw))
    c = pd.concat([pd.read_csv(os.path.join(FULL, 'runs', f'ba_cands_final_confirm_{g}.csv')) for g in gs])
    c = c.pivot_table(index=['group', 'src', 'tgt'], columns='cand', values='ba').reset_index()
    p = pd.concat([pd.read_csv(os.path.join(UP, f'p10_main_{g}.csv')) for g in gs])
    c = c.merge(p[['group', 'src', 'tgt', f'k8_all|{LW}']], on=['group', 'src', 'tgt'])
    rows[(site, 'confirmation split')] = (stat(c[J2] - c[HPJ]), stat(c[J2] - c[f'k8_all|{LW}']))
fr = pd.read_csv(os.path.join(UP, 'newregion', 'CONFIRM_FR_rows.csv'))
rows[('FR', 'Dutch archive')] = (stat(fr['j2'] - fr['hpj_cov01']), stat(fr['j2'] - fr['lw_all']))

T = ['\\begin{table}[!t]',
     '\\caption{Rain-driven prior: paired median difference in balanced accuracy to the empirical full-rank prior and to the proposed '
     'date-shared prior (share of transfers with a positive difference). Netherlands: archive of the other site, restricted to three, '
     'five or eight seasons (84 transfers), and the confirmation split with the full archive (111 transfers). France: frozen Dutch '
     'archive and parameters, Météo-France gauge (56 transfers).}',
     '\\label{tab:rain}', '\\centering\\footnotesize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular}{@{}llcc@{}}', '\\toprule',
     'Test & Archive & vs.\\ empirical prior & vs.\\ proposed\\\\', '\\midrule']
name = {'B': 'NL site B', 'A': 'NL site A', 'FR': 'France'}
for site in ('B', 'A', 'FR'):
    ks = [k for k in rows if k[0] == site]
    for i, k in enumerate(ks):
        T.append(f"{name[site] if i == 0 else ''} & {k[1]} & {cell(rows[k][0])} & {cell(rows[k][1])}\\\\")
    if site != 'FR': T.append('\\midrule')
T += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
open(os.path.join(HERE, 'tab_rain.tex'), 'w', encoding='utf-8').write('\n'.join(T) + '\n')
for k, v in rows.items(): print(k, 'vs hpj %+.4f (%.2f, n=%d)' % v[0], 'vs lw %+.4f (%.2f, n=%d)' % v[1])

# numbers quoted in the text
V = json.load(open(os.path.join(FULL, 'frozen_j2', 'VERIFY_J2.json')))
print('j2 th (aVV aVH aCOH b k):', {k: np.round(a['th'], 3).tolist() for k, a in V['archives'].items()})
log = open(os.path.join(FULL, 'runs', 'r3.log'), encoding='utf-8').read()
for site in ('A', 'B'):
    k1 = json.loads(re.search(rf'fit {site} k1_decay_cond (\{{.*?\}})', log).group(1))['th']; d = 1 / (1 + np.exp(-k1[5]))
    k2 = json.loads(re.search(rf'fit {site} k2_clsamp_cond (\{{.*?\}})', log).group(1))['th']
    print(f'archive {site}: k1 decay {d:.3f} half-life {np.log(0.5) / np.log(d):.2f} d, e-fold {-1 / np.log(d):.2f} d; '
          f'k2 grass amp {np.round(k2[:2], 2).tolist()} other {np.round(k2[2:7], 2).tolist()} ratio {np.mean(k2[:2]) / np.mean(k2[2:7]):.2f}; b {k2[8]:.2f}')
