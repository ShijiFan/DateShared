"""KNMI daily precipitation 2016-12-01..2025-10-31 for stations 283 (Site A) and 280 (Site B), identical request and P definition
to SAR/shift_20261017/p2d_rain.py (RH<0 -> 0.025 mm). Public, no account. -> knmi_daily_my.csv (+ sha256 printed)"""
import os, hashlib, requests, numpy as np, pandas as pd
G = os.path.dirname(os.path.abspath(__file__))
r = requests.post('https://www.daggegevens.knmi.nl/klimatologie/daggegevens',
                  data={'start': '20161201', 'end': '20251031', 'vars': 'RH:EV24', 'stns': '283:280', 'fmt': 'json'}, timeout=300)
K = pd.DataFrame(r.json()); K['date'] = pd.to_datetime(K.date).dt.tz_localize(None)
K['P'] = np.where(K.RH < 0, 0.025, K.RH / 10.0); p = os.path.join(G, 'knmi_daily_my.csv'); K.to_csv(p, index=False)
old = pd.read_csv(os.path.join(G, r'..\shift_20261017\knmi_daily.csv'), parse_dates=['date'])
m = old.merge(K[['station_code', 'date', 'P']], on=['station_code', 'date'], suffixes=('_old', '_new'))
print(len(K), K.date.min(), K.date.max(), K.groupby('station_code').RH.apply(lambda s: int(s.isna().sum())).to_dict(),
      'overlap', len(m), 'max |diff|', float((m.P_old - m.P_new).abs().max()), hashlib.sha256(open(p, 'rb').read()).hexdigest())
