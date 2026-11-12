"""N6 weather input for FR-A (rule fixed in N6_ADDENDUM_ZH.md before any FR transfer is run): the Météo-France station
nearest to the AOI centre (lon 0.825, lat 49.725) whose daily RR is >= 99% complete over 2016-11-01..2024-12-31.
Reads only the precipitation file; writes fr_rr_daily.csv (date, RR mm) and fr_station.json."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, 'weather_fr')
SRC = os.path.join(W, 'Q_76_previous-1950-2024_RR-T-Vent.csv.gz'); LON0, LAT0 = 0.825, 49.725
D0, D1 = pd.Timestamp('2016-11-01'), pd.Timestamp('2024-12-31'); NDAY = (D1 - D0).days + 1

d = pd.read_csv(SRC, sep=';', usecols=['NUM_POSTE', 'NOM_USUEL', 'LAT', 'LON', 'AAAAMMJJ', 'RR'], dtype={'NUM_POSTE': str})
d['date'] = pd.to_datetime(d.AAAAMMJJ.astype(str), format='%Y%m%d'); d = d[(d.date >= D0) & (d.date <= D1)]
st = d.groupby(['NUM_POSTE', 'NOM_USUEL']).agg(LAT=('LAT', 'first'), LON=('LON', 'first'), n=('RR', lambda s: s.notna().sum())).reset_index()
st['km'] = 6371 * np.hypot(np.radians(st.LON - LON0) * np.cos(np.radians(LAT0)), np.radians(st.LAT - LAT0)); st['complete'] = st.n / NDAY
st = st.sort_values('km'); print(st.head(12).to_string(index=False))
ok = st[st.complete >= 0.99]; s = ok.iloc[0]
r = d[d.NUM_POSTE == s.NUM_POSTE].set_index('date').RR.clip(lower=0).reindex(pd.date_range(D0, D1))
r.rename_axis('date').to_csv(os.path.join(HERE, 'fr_rr_daily.csv'))
info = dict(station=s.NUM_POSTE, name=s.NOM_USUEL, lat=float(s.LAT), lon=float(s.LON), km_from_centre=round(float(s.km), 1),
            completeness=round(float(s.complete), 4), missing_days=int(r.isna().sum()), source=os.path.basename(SRC),
            candidates=st.head(12)[['NUM_POSTE', 'NOM_USUEL', 'km', 'complete']].round(3).to_dict('records'))
json.dump(info, open(os.path.join(HERE, 'fr_station.json'), 'w'), indent=1, ensure_ascii=False); print(json.dumps({k: v for k, v in info.items() if k != 'candidates'}, ensure_ascii=False))
