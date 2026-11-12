"""Per site/year/track: products needed by plan_pairs_B.csv vs aligned (product_index.jsonl)."""
import os, csv, json
ROOT = r'D:\sar_data\multiyear'; HERE = os.path.dirname(os.path.abspath(__file__))
plan = list(csv.DictReader(open(os.path.join(HERE, 'plan_pairs_B.csv'), encoding='utf-8')))
for s in 'AB':
    p = os.path.join(ROOT, s, 'product_index.jsonl')
    have = {(r['year'], r['track'], r['key']) for r in map(json.loads, open(p))} if os.path.exists(p) else set()
    for (y, t) in sorted({(int(r['year']), int(r['track'])) for r in plan if r['site'] == s}):
        P = [r for r in plan if r['site'] == s and int(r['year']) == y and int(r['track']) == t]
        need = {r['date1'] for r in P} | {r['date2'] for r in P} | {f"{r['date1']}_{r['date2']}" for r in P}
        got = {k for k in need if (y, t, k) in have}
        print(s, y, t, f'{len(got)}/{len(need)}', 'missing:', ' '.join(sorted(need - got))[:150])
