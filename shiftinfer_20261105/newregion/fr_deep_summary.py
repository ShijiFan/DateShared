"""FR deep baselines summary (report only; prereg §9 2026-11-07): same comparisons and severity split as NL S3a/P11d -> FR_DEEP_SUMMARY.json"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv(os.path.join(HERE, 'FR_DEEP_rows.csv')).merge(pd.read_csv(os.path.join(HERE, 'CONFIRM_FR_rows.csv'))[['src', 'tgt', 'lw_all', 'hgb_lw', 'em_cov01']], on=['src', 'tgt'])
def pair(d, a, b):
    z = (d[a] - d[b]).dropna(); return dict(n=int(len(z)), median=round(float(z.median()), 4), frac_pos=round(float((z > 0).mean()), 3))
o = dict(n=int(len(D)), median_ba={c: round(float(D[c].median()), 4) for c in D.columns if c not in ('src', 'tgt')})
o['pairs'] = {f'{a} - {b}': pair(D, a, b) for a, b in [('lw_all', 'dl_dann'), ('lw_all', 'dl_adabn'), ('lw_all', 'dl_raw'), ('tcn_lw', 'tcn_em'),
              ('tcn_lw', 'dl_raw'), ('tcn_lw', 'dl_dann'), ('dann_lw', 'dl_dann'), ('dann_lw', 'lw_all'), ('hgb_lw', 'dl_dann')]}
q = pd.qcut(D['dl_raw'], 3, labels=['severe', 'mid', 'mild'])
o['by_severity'] = {lv: {k: pair(D[q == lv], a, b) for k, (a, b) in {'dann_lw - dl_dann': ('dann_lw', 'dl_dann'), 'lw_all - dl_dann': ('lw_all', 'dl_dann')}.items()} for lv in ('severe', 'mid', 'mild')}
json.dump(o, open(os.path.join(HERE, 'FR_DEEP_SUMMARY.json'), 'w'), indent=1); print(json.dumps(o, indent=1))
