"""Criteria of the FR confirmation exactly as pre-registered (CONFIRM_PREREG_DRAFT_ZH.md §5, N6_ADDENDUM §4) -> CONFIRM_FR_SUMMARY.json
N1 lw_all vs em_cov01: median >= +0.01 and >= 2/3 positive; N2 lw_all vs hpj_cov01: median >= -0.003; N3 hgb_lw vs hgb_em: median >= +0.01 and
>= 2/3 positive; N4 (report) lw vs hpj per transfer; N5 (report) local 3-year vs NL 8-year archives; N6 j2 vs hpj_cov01: non-inferior
(median >= -0.003) / improvement (median > 0 and >= 55% positive); N6b (report) j2 vs lw_all; N6c (report) Spearman(date coef, dAPI).
Statistics: median, share positive, and a year-block bootstrap 90% interval (resample years, keep transfers among resampled years)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(os.path.join(HERE, 'CONFIRM_FR_rows.csv')); C = pd.read_csv(os.path.join(HERE, 'CONFIRM_FR_coef.csv'))
YEARS = sorted(set(R.src) | set(R.tgt)); rng = np.random.default_rng(20261106)


def pair(a, b):                                          # NaN rows (N5 degenerate local archive, prereg §9 22:40) are skipped
    dd = (R[a] - R[b]); d = dd.dropna().values; boot = []
    for _ in range(2000):
        ys = set(rng.choice(YEARS, len(YEARS), replace=True).tolist()); m = R.src.isin(ys) & R.tgt.isin(ys) & dd.notna()
        if m.sum() >= 5: boot.append(float(np.median(dd[m])))
    return dict(n=int(len(d)), median=round(float(np.median(d)), 4), mean=round(float(np.mean(d)), 4), frac_pos=round(float((d > 0).mean()), 3),
                boot90=[round(float(np.percentile(boot, 5)), 4), round(float(np.percentile(boot, 95)), 4)])


o = dict(n_transfers=int(len(R)), years=YEARS, n5_valid=int(R.lw_local3.notna().sum()),
         median_ba={c: round(float(R[c].median()), 4) for c in R.columns if c not in ('src', 'tgt', 'n_slots', 'n_local_complete')})
n1, n2, n3, n6 = pair('lw_all', 'em_cov01'), pair('lw_all', 'hpj_cov01'), pair('hgb_lw', 'hgb_em'), pair('j2', 'hpj_cov01')
o['criteria'] = {
    'N1 lw_all vs em_cov01 (median>=+0.01 & >=2/3 pos)': dict(res=n1, pass_=n1['median'] >= 0.01 and n1['frac_pos'] >= 2 / 3),
    'N2 lw_all vs hpj_cov01 (median>=-0.003)': dict(res=n2, pass_=n2['median'] >= -0.003),
    'N3 hgb_lw vs hgb_em (median>=+0.01 & >=2/3 pos)': dict(res=n3, pass_=n3['median'] >= 0.01 and n3['frac_pos'] >= 2 / 3),
    'N6 j2 vs hpj_cov01': dict(res=n6, grade='improvement' if (n6['median'] > 0 and n6['frac_pos'] >= 0.55) else ('non-inferior' if n6['median'] >= -0.003 else 'fail'))}
o['report'] = {'N4 lw_all vs hpj per transfer': R.assign(d=R.lw_all - R.hpj_cov01)[['src', 'tgt', 'd']].round(4).to_dict('records'),
               'N5 lw_local3 vs lw_all': pair('lw_local3', 'lw_all'), 'N5 hpj_local3 vs hpj_cov01': pair('hpj_local3', 'hpj_cov01'),
               'N5 lw_local3 vs hpj_local3': pair('lw_local3', 'hpj_local3'),
               'secondary lw_T88 vs lw_all': pair('lw_T88', 'lw_all'), 'secondary lw_T88 vs em_cov01': pair('lw_T88', 'em_cov01'),
               'N6b j2 vs lw_all': pair('j2', 'lw_all'), 'hgb_lw vs hgb_hpj': pair('hgb_lw', 'hgb_hpj'), 'hgb_em vs hgb_raw': pair('hgb_em', 'hgb_raw'),
               'lw_all vs g_raw': pair('lw_all', 'g_raw'), 'em_cov01 vs g_raw': pair('em_cov01', 'g_raw'),
               'N6c spearman(date coef, dAPI)': dict(rho=round(float(spearmanr(C.coef, C.api_t - C.api_s)[0]), 3), n=int(len(C)))}
mb = o['median_ba']; o['report']['oracle_gap_recovery'] = {v: round((mb[v] - mb['g_raw']) / (mb['g_oracle'] - mb['g_raw']), 3) for v in ('em_cov01', 'hpj_cov01', 'lw_all')}
json.dump(o, open(os.path.join(HERE, 'CONFIRM_FR_SUMMARY.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in o.items() if k != 'report'}, indent=1)); print(json.dumps({k: v for k, v in o['report'].items() if not k.startswith('N4')}, indent=1))
