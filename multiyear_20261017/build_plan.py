"""Multi-year extension plan (user approved ordering; THIS SCRIPT DOES NOT SUBMIT). For 2017-2023, tracks 37/88:
- ASF search S1A IW SLC (1SDV) March-October; per date keep scenes, check whether one scene footprint CONTAINS each site bbox.
- Select 9 disjoint S1A-S1A 12-day pairs by DOY matching to the 2025 M=9 slots (same rule site2 used for 2024: nearest date_1 DOY,
  non-overlapping, within 14 days), separately for the frame covering Site A and the frame covering Site B (one frame if it covers both).
- Payload parameters copied from the A1 SUBMISSION_PLAN (RTC_GAMMA 20 m gamma0 power; INSAR_GAMMA 10x2, wrapped phase, filter 0.6).
Outputs: plan_pairs.csv, SUBMISSION_PLAN_MULTIYEAR.json (payloads, n_jobs, credits), coverage.csv"""
import os, json, hashlib, time
import pandas as pd, requests
from shapely.geometry import shape, box
OUT = os.path.dirname(os.path.abspath(__file__))
SITES = {'A': box(6.00, 51.98, 6.54, 52.38), 'B': box(6.55, 52.75, 7.05, 53.10)}
SLOTS = {37: ['20250308', '20250401', '20250425', '20250519', '20250612', '20250706', '20250823', '20250916', '20251010'],
         88: ['20250311', '20250404', '20250428', '20250522', '20250615', '20250709', '20250802', '20250826', '20251013']}
COST = {'RTC_GAMMA': 15, 'INSAR_GAMMA': 15}
def search(year, track):
    wkt = 'POLYGON((6.0 51.98,7.05 51.98,7.05 53.10,6.0 53.10,6.0 51.98))'
    r = requests.get('https://api.daac.asf.alaska.edu/services/search/param', params={'platform': 'Sentinel-1A', 'processingLevel': 'SLC', 'beamMode': 'IW',
                     'polarization': 'VV+VH', 'relativeOrbit': track, 'intersectsWith': wkt, 'start': f'{year}-02-20T00:00:00Z', 'end': f'{year}-11-10T23:59:59Z',
                     'output': 'geojson', 'maxResults': 1000}, timeout=180).json()['features']
    rows = []
    for f in r:
        g = shape(f['geometry']); p = f['properties']
        rows.append(dict(scene=p['sceneName'], date=p['startTime'][:10], cov_A=g.contains(SITES['A']), cov_B=g.contains(SITES['B'])))
    return pd.DataFrame(rows)
cov_rows, pairs = [], []
for year in range(2017, 2024):
    for trk in (37, 88):
        S = search(year, trk); S['date'] = pd.to_datetime(S.date)
        for site in ('A', 'B'):
            col = f'cov_{site}'; s_ok = S[S[col]].sort_values('date').drop_duplicates('date')
            dates = list(s_ok.date)
            cand = [(d1, d2) for d1 in dates for d2 in dates if (d2 - d1).days == 12]
            used, chosen = set(), []
            for slot in SLOTS[trk]:
                doy = pd.Timestamp(slot).dayofyear
                opts = sorted([c for c in cand if abs(c[0].dayofyear - doy) <= 14 and c[0] not in used and c[1] not in used],
                              key=lambda c: abs(c[0].dayofyear - doy))
                if not opts: chosen.append(None); continue
                c = opts[0]; chosen.append(c); used |= {c[0], c[1]}
            for i, c in enumerate(chosen):
                if c is None: pairs.append(dict(year=year, track=trk, site=site, slot=i, date1=None, date2=None, scene1=None, scene2=None)); continue
                sc1 = s_ok[s_ok.date == c[0]].scene.iloc[0]; sc2 = s_ok[s_ok.date == c[1]].scene.iloc[0]
                pairs.append(dict(year=year, track=trk, site=site, slot=i, date1=c[0].date(), date2=c[1].date(), scene1=sc1, scene2=sc2))
        cov_rows.append(dict(year=year, track=trk, n_dates=S.date.nunique(), dates_cov_A=int(S[S.cov_A].date.nunique()), dates_cov_B=int(S[S.cov_B].date.nunique()),
                             dates_one_scene_covers_both=int(S[S.cov_A & S.cov_B].date.nunique())))
        print(year, trk, cov_rows[-1], flush=True)
P = pd.DataFrame(pairs); P.to_csv(os.path.join(OUT, 'plan_pairs.csv'), index=False); pd.DataFrame(cov_rows).to_csv(os.path.join(OUT, 'coverage.csv'), index=False)
ok = P.dropna(subset=['scene1'])
scenes = sorted(set(ok.scene1) | set(ok.scene2)); ifg = sorted({(a, b) for a, b in zip(ok.scene1, ok.scene2)})
def name(prefix, *g): return f'{prefix}_' + hashlib.sha1('|'.join(g).encode()).hexdigest()[:12]
payloads = [dict(job_type='RTC_GAMMA', name=name('RTC_GAMMA', s), job_parameters=dict(granules=[s], dem_matching=False, include_dem=False, include_inc_map=False,
            include_rgb=False, include_scattering_area=False, radiometry='gamma0', resolution=20, scale='power', speckle_filter=False, dem_name='copernicus')) for s in scenes]
payloads += [dict(job_type='INSAR_GAMMA', name=name('INSAR_GAMMA', a, b), job_parameters=dict(granules=[a, b], include_look_vectors=False, include_los_displacement=False,
             include_inc_map=True, looks='10x2', include_dem=False, include_wrapped_phase=True, apply_water_mask=True, include_displacement_maps=False,
             phase_filter_parameter=0.6)) for a, b in ifg]
credits = sum(COST[p['job_type']] for p in payloads)
plan = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), n_jobs=len(payloads), n_rtc=len(scenes), n_insar=len(ifg), credits=credits,
            missing_slots=int(P.scene1.isna().sum()), payloads=payloads, status='PLAN_ONLY_NOT_SUBMITTED')
json.dump(plan, open(os.path.join(OUT, 'SUBMISSION_PLAN_MULTIYEAR.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in plan.items() if k != 'payloads'}, indent=1))
print(P.groupby(['year', 'track', 'site']).scene1.apply(lambda s: s.notna().sum()).unstack())
