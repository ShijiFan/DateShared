"""Report-only comparison of baselines (S3a deep, S3b recent, P11 HGB) with the frozen method on identical transfers -> BASELINES.json
Includes scenario split by shift severity (tercile of hgb_raw drop relative to within-year is not available for all; use em_cov01 gain proxy:
severity = hpj-free gap = (P9 em_cov01 - P9 g_raw) is not stored; we use dl_raw BA terciles as severity proxy)."""
import os, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE); G = ['A37', 'A88', 'B88']; KEY = ['group', 'src', 'tgt']
DIRS = {'testB88_priorA': ['B88'], 'testA_priorB': ['A37', 'A88']}
def rd(pat):
    fs = [os.path.join(UP, pat.format(g)) for g in G if os.path.exists(os.path.join(UP, pat.format(g)))]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
def pair(df, a, b):
    d = (df[a] - df[b]).dropna()
    return dict(n=int(len(d)), median=round(float(d.median()), 4), frac_pos=round(float((d > 0).mean()), 3)) if len(d) else None
T = rd('p9_basis_{}.csv')[KEY + ['em_cov01', 'hpj_cov01']].merge(rd('p10_main_{}.csv')[KEY + ['k8_all|hybridD_lw']].rename(columns={'k8_all|hybridD_lw': 'gauss_lw'}), on=KEY)
for pat, cols in (('s3a_deep_{}.csv', ['dl_raw', 'dl_adabn', 'dl_dann']), ('s3b_recent_{}.csv', ['sfuda', 'raincoat']), ('p11_c2_{}.csv', ['hgb_raw', 'hgb_em', 'hgb_hpj', 'hgb_lw', 'hgb_oracle'])):
    x = rd(pat)
    if x is not None: T = T.merge(x[KEY + [c for c in cols if c in x]], on=KEY, how='left')
meth = [c for c in ['em_cov01', 'hpj_cov01', 'gauss_lw', 'hgb_raw', 'hgb_em', 'hgb_hpj', 'hgb_lw', 'hgb_oracle', 'dl_raw', 'dl_adabn', 'dl_dann', 'sfuda', 'raincoat'] if c in T]
out = {'n': {c: int(T[c].notna().sum()) for c in meth}}
for dn, gs in DIRS.items():
    d = T[T.group.isin(gs)]; o = {'median_ba': {c: round(float(d[c].median()), 4) for c in meth}}
    best_ours = [c for c in ('gauss_lw', 'hgb_lw') if c in d]
    o['ours_vs'] = {f'{a} - {b}': pair(d, a, b) for a in best_ours for b in meth if b not in best_ours}
    if 'dl_raw' in d:   # scenario: shift severity terciles by deep source-only BA (low BA = severe shift)
        q = pd.qcut(d['dl_raw'], 3, labels=['severe', 'mid', 'mild']); o['by_severity'] = {}
        for lv in ('severe', 'mid', 'mild'):
            dd = d[q == lv]; o['by_severity'][lv] = {f'{a} - {b}': pair(dd, a, b) for a in best_ours for b in ('dl_dann', 'dl_adabn', 'sfuda', 'raincoat', 'hpj_cov01') if b in dd}
    out[dn] = o
json.dump(out, open(os.path.join(HERE, 'BASELINES.json'), 'w'), indent=1)
for dn in DIRS: print(dn, json.dumps(out[dn]['median_ba'])); print(json.dumps(out[dn]['ours_vs'])); print(json.dumps(out[dn].get('by_severity', {})))
