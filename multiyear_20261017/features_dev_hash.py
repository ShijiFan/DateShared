"""Hash every array of features_{A,B}.npz (per key) so a rebuild with 2023 can be checked to leave all dev-year arrays bit-identical.
Usage: python features_dev_hash.py <out.json>"""
import sys, json, hashlib, numpy as np
out = {}
for s in 'AB':
    z = np.load(rf'D:\sar_data\multiyear\features_{s}.npz', allow_pickle=True)
    for k in z.files: out[f'{s}:{k}'] = hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()[:16]
json.dump(out, open(sys.argv[1], 'w'), indent=0); print(len(out), 'arrays hashed')
