"""Consolidated results for the frozen method (read-only over existing CSVs; no new runs).
-> results/CONSOLIDATED.json, results/fig_archive_curve.png, results/fig_coef_vs_api.png
Sources: p9_basis_*, p10_main_*, p10_curve_*, toy_B/full/runs/curve_* (j2, k1), toy_B/full/runs/ba_cands_final_confirm_*, p9_coef_*."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE); G = ['A37', 'A88', 'B88']; KEY = ['group', 'src', 'tgt']
DIRS = {'testB88_priorA': ['B88'], 'testA_priorB': ['A37', 'A88']}
rd = lambda f: pd.concat([pd.read_csv(os.path.join(UP, f.format(g))) for g in G], ignore_index=True)


def pair(df, a, b):
    d = (df[a] - df[b]).dropna(); return dict(n=int(len(d)), median=round(float(d.median()), 4), frac_pos=round(float((d > 0).mean()), 3))


out = {}
# ---- T1 main comparison (all 168 transfers, full archive) ----
P9 = rd('p9_basis_{}.csv'); P10 = rd('p10_main_{}.csv')[KEY + ['k8_all|hybridD_lw']].rename(columns={'k8_all|hybridD_lw': 'hybridD_lw'})
T = P9.merge(P10, on=KEY); meth = ['em_cov01', 'mllr_cov01', 'phys_nom', 'learn_r*', 'hybrid_r*', 'hybridD_r*', 'hpj_cov01', 'hybridD_lw']
out['T1_main'] = {dn: {m: dict(median_ba=round(float(T[T.group.isin(gs)][m].median()), 4), vs_em=pair(T[T.group.isin(gs)], m, 'em_cov01'),
                                vs_hpj=pair(T[T.group.isin(gs)], m, 'hpj_cov01')) for m in meth} for dn, gs in DIRS.items()}
# ---- T2 archive-length curve: P10 (4 draws) and, on draws 0-1, j2/k1 vs lw ----
C10 = rd('p10_curve_{}.csv'); Cb = pd.concat([pd.read_csv(os.path.join(UP, 'toy_B', 'full', 'runs', f'curve_{g}.csv')) for g in G], ignore_index=True)
Cb = Cb[[c for c in Cb.columns if c in KEY or 'j2' in c or 'k1' in c]]; CC = C10.merge(Cb, on=KEY, how='inner')
def lvl(df, L, v, draws):
    return df[[f'k{L}_{i}|{v}' for i in draws]].mean(axis=1) if L != 8 else df[f'k8_all|{v}']
curve = {}
for dn, gs in DIRS.items():
    d4 = C10[C10.group.isin(gs)]; d2 = CC[CC.group.isin(gs)]; curve[dn] = {}
    for L in (3, 5, 8):
        r = {f'{v}': round(float(lvl(d4, L, v, range(4)).median()), 4) for v in ('hpj_cov01', 'hybridD_r*', 'hybridD_lw')}
        x = pd.DataFrame({v: lvl(d2, L, v, (0, 1)) for v in ('hpj_cov01', 'hybridD_lw', 'j2_hpj_wc_cond', 'k1_decay_cond')})
        r.update({f'{v}@draws01': round(float(x[v].median()), 4) for v in x}); r['n_common_draws01'] = int(len(x))
        r['lw_vs_hpj@4draws'] = pair(pd.DataFrame({'a': lvl(d4, L, 'hybridD_lw', range(4)), 'b': lvl(d4, L, 'hpj_cov01', range(4))}), 'a', 'b')
        r['lw_vs_j2@draws01'] = pair(x, 'hybridD_lw', 'j2_hpj_wc_cond'); r['lw_vs_k1@draws01'] = pair(x, 'hybridD_lw', 'k1_decay_cond')
        curve[dn][L] = r
out['T2_curve'] = curve
# ---- T3 lw vs j2/k1 on the B-full confirm transfers (full archive) ----
Bc = pd.concat([pd.read_csv(os.path.join(UP, 'toy_B', 'full', 'runs', f'ba_cands_final_confirm_{g}.csv')) for g in G], ignore_index=True)
Bw = Bc.pivot_table(index=KEY, columns='cand', values='ba').reset_index().merge(P10, on=KEY)
out['T3_confirm_split'] = {dn: {c: pair(Bw[Bw.group.isin(gs)], 'hybridD_lw', c) for c in [c for c in Bw.columns if c not in KEY + ['hybridD_lw']]} for dn, gs in DIRS.items()}
out['T3_n'] = {dn: int(Bw.group.isin(gs).sum()) for dn, gs in DIRS.items()}
# ---- T4 interpretability (P9 hybridD_r* estimates; frozen-method coefficients not yet extracted) ----
Cf = rd('p9_coef_{}.csv'); w = Cf[Cf.kind == 'wbs'].copy(); w['dapi'] = w.api_t - w.api_s
out['T4_coef'] = {dn: dict(spearman=round(float(spearmanr(x.coef, x.dapi)[0]), 3), n=int(len(x))) for dn, gs in DIRS.items() for x in [w[w.group.isin(gs)]]}
json.dump(out, open(os.path.join(HERE, 'CONSOLIDATED.json'), 'w'), indent=1)

# ---- figures (draft; validated categorical slots 1-4, direct labels + distinct markers because slots 3-4 < 3:1 contrast) ----
COL = {'hpj_cov01': '#2a78d6', 'hybridD_lw': '#eb6834', 'j2_hpj_wc_cond': '#1baf7a', 'hybridD_r*': '#eda100'}
LAB = {'hpj_cov01': 'Full-rank joint prior (hpj)', 'hybridD_lw': 'Frozen: water-cloud basis + LW residual', 'j2_hpj_wc_cond': 'Weather-conditioned prior (j2)', 'hybridD_r*': 'Water-cloud basis + low-rank residual'}
MK = {'hpj_cov01': 'o', 'hybridD_lw': 's', 'j2_hpj_wc_cond': '^', 'hybridD_r*': 'D'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e'})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), facecolor='#fcfcfb')
for ax, (dn, title) in zip(axs, (('testB88_priorA', 'Test site B (archive: site A)'), ('testA_priorB', 'Test site A (archive: site B)'))):
    ax.set_facecolor('#fcfcfb'); ax.grid(axis='y', color='#e6e5e0', lw=0.6)
    for v in ('hpj_cov01', 'hybridD_r*', 'hybridD_lw'):
        y = [curve[dn][L][v] for L in (3, 5, 8)]
        ax.plot([3, 5, 8], y, color=COL[v], lw=2, marker=MK[v], ms=6, mec='#fcfcfb', mew=1.5, label=LAB[v])
    yj = [curve[dn][L]['j2_hpj_wc_cond@draws01'] for L in (3, 5, 8)]
    ax.plot([3, 5, 8], yj, color=COL['j2_hpj_wc_cond'], lw=2, ls='--', marker=MK['j2_hpj_wc_cond'], ms=6, mec='#fcfcfb', mew=1.5, label=LAB['j2_hpj_wc_cond'] + ' (2 draws)')
    ax.set_xticks([3, 5, 8]); ax.set_xlabel('Archive length (years)'); ax.set_title(title, fontsize=9, color='#0b0b0b', loc='left')
axs[0].set_ylabel('Median balanced accuracy')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(os.path.join(HERE, 'fig_archive_curve.png'), dpi=200, facecolor='#fcfcfb'); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), facecolor='#fcfcfb', sharey=True)
for ax, (dn, title) in zip(axs, (('testB88_priorA', 'Test site B'), ('testA_priorB', 'Test site A'))):
    x = w[w.group.isin(DIRS[dn])]; ax.set_facecolor('#fcfcfb')
    ax.scatter(x.dapi, x.coef, s=4, color='#2a78d6', alpha=0.25, lw=0)
    ax.axhline(0, color='#c3c2b7', lw=0.6); ax.axvline(0, color='#c3c2b7', lw=0.6)
    ax.set_title(f"{title}: Spearman {out['T4_coef'][dn]['spearman']:.2f} (n={out['T4_coef'][dn]['n']})", fontsize=9, loc='left', color='#0b0b0b')
    ax.set_xlabel('ΔAPI at acquisition date (mm)')
axs[0].set_ylabel('Estimated shared date offset (label-free)')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_coef_vs_api.png'), dpi=200, facecolor='#fcfcfb'); plt.close(fig)
print(json.dumps(out['T2_curve'], indent=1)); print(json.dumps(out['T3_confirm_split'], indent=1)); print(json.dumps(out['T3_n']))
