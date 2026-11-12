"""ClimID-UDA inputs: ERA5 reanalysis daily (Open-Meteo best_match = ERA5-Land/ERA5 seamless; era5_land alone lacks precip/radiation/pressure aggregates) (Open-Meteo archive API, public, no account) for the 6 ClimID variables (I11) on a 3x3 point grid
inside each AOI (A, B from build_labels_my.py; FR from DATA_NEWREGION_ZH.md). -> era5land_<site>.csv (+ sha256 printed)"""
import os, hashlib, json, time, urllib.request, urllib.parse, urllib.error
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
AOI = {'A': (6.00, 51.98, 6.54, 52.38), 'B': (6.55, 52.75, 7.05, 53.10), 'FR': (0.55, 49.55, 1.10, 49.90)}
V = ['temperature_2m_mean', 'temperature_2m_max', 'dew_point_2m_mean', 'precipitation_sum', 'shortwave_radiation_sum', 'surface_pressure_mean']
for s, (x0, y0, x1, y1) in AOI.items():
    lon = np.linspace(x0, x1, 5)[1:4]; lat = np.linspace(y0, y1, 5)[1:4]; P = [(la, lo) for la in lat for lo in lon]
    q = dict(latitude=','.join(f'{p[0]:.4f}' for p in P), longitude=','.join(f'{p[1]:.4f}' for p in P), start_date='2016-12-01',
             end_date='2025-10-31', daily=','.join(V), timezone='UTC')
    req = urllib.request.Request('https://archive-api.open-meteo.com/v1/archive?' + urllib.parse.urlencode(q), headers={'User-Agent': 'research-script'})
    for att in range(8):
        try: R = json.load(urllib.request.urlopen(req, timeout=600)); break
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(90 * (att + 1))
    R = R if isinstance(R, list) else [R]; rows = []
    for i, r in enumerate(R):
        d = pd.DataFrame(r['daily']); d['cell'] = i; d['lat'] = r['latitude']; d['lon'] = r['longitude']; rows.append(d)
    D = pd.concat(rows); p = os.path.join(H, f'era5land_{s}.csv'); D.to_csv(p, index=False)
    print(s, D.shape, D['time'].min(), D['time'].max(), 'NaN', int(D[V].isna().sum().sum()), hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]); time.sleep(60)
