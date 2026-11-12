"""Score B-full candidates (PROTOCOL_B_FULL_ZH.md §2): paired BA differences per test direction (testA = A37+A88 with
archive B; testB = B88 with archive A), median and share positive; selection score S = min over directions of the median
difference vs hpj_cov01. Reproduction check of the P9 replicas against p9_basis_<grp>.csv.
Usage: python score.py <tag> [search|confirm] -> prints table, appends results.jsonl"""
import os, sys, glob, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(os.path.dirname(HERE))
REP = {'rep_hybridD': 'hybridD_r*', 'rep_hybridD_mw': 'hybridD_r*_mw', 'hpj_cov01': 'hpj_cov01', 'em_cov01': 'em_cov01'}


def load(tag, which):
    fs = [f for f in glob.glob(os.path.join(HERE, 'runs', f'ba_{tag}_{which}_*.csv'))]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def repro(df):
    out = {}
    for g, d in df.groupby('group'):
        f = os.path.join(UP, f'p9_basis_{g}.csv')
        if not os.path.exists(f): continue
        p = pd.read_csv(f)
        for mine, theirs in REP.items():
            a = d[d.cand == mine].merge(p[['src', 'tgt', theirs]], on=['src', 'tgt'])
            if len(a): out[f'{g}:{mine}'] = (len(a), float(np.abs(a.ba - a[theirs]).max()))
    return out


def add_p9(df):
    """attach P9's hybridD_r* (same transfers) as candidate 'p9_hybridD' when not computed here."""
    if 'rep_hybridD' in set(df.cand): return df
    ex = []
    for g in df.group.unique():
        p = pd.read_csv(os.path.join(UP, f'p9_basis_{g}.csv'))
        a = df[(df.group == g) & (df.cand == 'hpj_cov01')][['group', 'src', 'tgt']].merge(p[['src', 'tgt', 'hybridD_r*']], on=['src', 'tgt'])
        ex.append(a.rename(columns={'hybridD_r*': 'ba'}).assign(cand='p9_hybridD'))
    return pd.concat([df] + ex, ignore_index=True)


def grade(W, c, ref='hpj_cov01'):
    """PROTOCOL §4 / P9 H1 grade, per direction."""
    st = [((d[c] - d[ref]).median(), ((d[c] - d[ref]) > 0).mean()) for _, d in W.groupby('dir')]
    if all(m >= 0.01 and p >= 2 / 3 for m, p in st): return '更好'
    if all(m >= -0.003 for m, _ in st): return '不劣'
    return '更差'


def table(df, refs=('hpj_cov01', 'em_cov01', 'rep_hybridD', 'p9_hybridD')):
    df = add_p9(df)
    W = df.pivot_table(index=['group', 'src', 'tgt'], columns='cand', values='ba').dropna()
    W['dir'] = np.where(W.index.get_level_values(0).str[0] == 'A', 'testA', 'testB'); rows = []
    for c in [c for c in W.columns if c != 'dir']:
        r = dict(cand=c, n=len(W), meanBA=round(W[c].mean(), 4))
        for ref in [q for q in refs if q in W.columns]:
            for dr, d in W.groupby('dir'):
                x = d[c] - d[ref]; r[f'{dr}_vs_{ref}'] = f'{x.median():+.4f} ({(x > 0).mean():.2f})'
        r['S'] = round(min((d[c] - d['hpj_cov01']).median() for _, d in W.groupby('dir')), 4); r['grade_vs_hpj'] = grade(W, c)
        rows.append(r)
    return pd.DataFrame(rows).sort_values('S', ascending=False), W


if __name__ == '__main__':
    tag = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else 'search'
    df = load(tag, which); T, W = table(df); rp = repro(df)
    pd.set_option('display.width', 250); print(T.to_string(index=False)); print('repro (n, max|diff|):', rp)
    print(W.groupby('dir').size().to_dict())
    with open(os.path.join(HERE, 'results.jsonl'), 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(dict(tag=tag, split=which, t=time.strftime('%Y-%m-%d %H:%M'), n=len(W), table=T.to_dict('records'), repro=rp), ensure_ascii=False) + '\n')
