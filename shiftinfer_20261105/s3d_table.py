"""Main comparison table with deep / recent-Q1 baselines (user 2026-11-10: deep comparison uses the TempCNN plug-in version of the
proposed method, one version for every column; frozen-method results stay in tab_nl / tab_fr).
Columns: NL site B (B88, 28 transfers), NL site A (A37 + A88, 56), France (56); NL = the S3c/S3d todo[0::2] transfers.
Per cell: median BA, paired median difference ours(TempCNN) - method, share of transfers where ours is better; plus mean BA."""
import os, pandas as pd, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
R = lambda f: pd.read_csv(os.path.join(H, f))
COLS = {'dl_raw': 'No adaptation (TempCNN)', 'dl_adabn': 'AdaBN', 'dl_dann': 'DANN', 'sfuda': 'SFUDA (RSE 2024)',
        'srs_A': 'RAINCOAT-SRS (ISPRS 2025), rule A', 'srs_D': 'RAINCOAT-SRS, rule D (target-test selection)',
        'tcn_lw': 'Proposed (TempCNN)', 'dann_lw': 'Proposed + DANN'}


def nl(groups):
    out = []
    for g in groups:
        s3d = R(f's3d_{g}.csv').drop_duplicates(['src', 'tgt'])[['src', 'tgt', 'srs_A', 'srs_D']]
        m = (s3d.merge(R(f's3a_deep_{g}.csv')[['src', 'tgt', 'dl_raw', 'dl_adabn', 'dl_dann']], on=['src', 'tgt'])
                .merge(R(f's3j_deep_{g}.csv').drop_duplicates(['src', 'tgt'])[['src', 'tgt', 'tcn_lw_x', 'dann_lw_x']]     # S3j 2026-11-12: archive
                       .rename(columns={'tcn_lw_x': 'tcn_lw', 'dann_lw_x': 'dann_lw'}), on=['src', 'tgt'])              # without the transfer's seasons
                .merge(R(f's3c_official_{g}.csv')[['src', 'tgt', 'sfuda_tuned']].rename(columns={'sfuda_tuned': 'sfuda'}), on=['src', 'tgt']))
        out.append(m.assign(group=g))
    return pd.concat(out)


def fr():
    s3d = R('s3d_FR.csv').drop_duplicates(['src', 'tgt'])[['src', 'tgt', 'srs_A', 'srs_D', 'sfuda']]
    return s3d.merge(R(os.path.join('newregion', 'FR_DEEP_rows.csv'))[['src', 'tgt', 'dl_raw', 'dl_adabn', 'dl_dann', 'tcn_lw', 'dann_lw']], on=['src', 'tgt'])


T = {'NL site B': nl(['B88']), 'NL site A': nl(['A37', 'A88']), 'France': fr()}
rows = []
for k, lab in COLS.items():
    r = {'method': lab}
    for c, d in T.items():
        dd = d['tcn_lw'] - d[k]
        r[f'{c} n'] = len(d); r[f'{c} median'] = d[k].median(); r[f'{c} mean'] = d[k].mean()
        r[f'{c} d_med'] = np.nan if k == 'tcn_lw' else dd.median(); r[f'{c} win'] = np.nan if k == 'tcn_lw' else (dd > 0).mean()
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(os.path.join(H, 's3d_main_table.csv'), index=False)
pd.set_option('display.width', 250)
for c in T:
    print(f'== {c} (n={len(T[c])})')
    print(S[['method', f'{c} median', f'{c} mean', f'{c} d_med', f'{c} win']].round(3).to_string(index=False))
