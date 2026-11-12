"""New-region HyP3 PLAN (THIS SCRIPT NEVER SUBMITS ANYTHING). Same rules and payloads as SAR/multiyear_20261017/build_plan.py:
- ASF search S1A IW SLC 1SDV, 20 Feb - 10 Nov, per relative orbit; keep dates whose scene footprint CONTAINS the AOI box;
- 9 disjoint 12-day S1A-S1A pairs per year by DOY matching (first date within 14 days) to the NL slot calendar of the chosen
  reference track (FREEZE_METHOD_ZH.md §5: default T88 calendar);
- payloads: RTC_GAMMA 20 m gamma0 power + INSAR_GAMMA 10x2 (wrapped phase, filter 0.6); 15 credits per job (as in NL plan; re-check).
Usage: python plan_region.py <name> <lon_min> <lat_min> <lon_max> <lat_max> <years comma> [orbit or 'auto'] [ref_track 88|37]
Outputs (newregion/<name>/): orbits.csv, plan_pairs.csv, SUBMISSION_PLAN.json (status PLAN_ONLY_NOT_SUBMITTED)."""
import os, sys, json, hashlib, time
import pandas as pd, requests
from shapely.geometry import shape, box
SLOTS = {37: ['20250308', '20250401', '20250425', '20250519', '20250612', '20250706', '20250823', '20250916', '20251010'],
         88: ['20250311', '20250404', '20250428', '20250522', '20250615', '20250709', '20250802', '20250826', '20251013']}
COST = {'RTC_GAMMA': 15, 'INSAR_GAMMA': 15}
API = 'https://api.daac.asf.alaska.edu/services/search/param'


def search(aoi, year, orbit=None):
    lo0, la0, lo1, la1 = aoi.bounds
    wkt = f'POLYGON(({lo0} {la0},{lo1} {la0},{lo1} {la1},{lo0} {la1},{lo0} {la0}))'
    p = dict(platform='Sentinel-1A', processingLevel='SLC', beamMode='IW', polarization='VV+VH', intersectsWith=wkt,
             start=f'{year}-02-20T00:00:00Z', end=f'{year}-11-10T23:59:59Z', output='geojson', maxResults=2000)
    if orbit: p['relativeOrbit'] = orbit
    rows = []
    for f in requests.get(API, params=p, timeout=300).json()['features']:
        g = shape(f['geometry']); q = f['properties']
        rows.append(dict(scene=q['sceneName'], date=q['startTime'][:10], orbit=int(q['pathNumber']), direction=q.get('flightDirection'), covers=g.contains(aoi)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    name = sys.argv[1]; aoi = box(*map(float, sys.argv[2:6])); years = [int(y) for y in sys.argv[6].split(',')]
    orbit = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != 'auto' else None; ref = int(sys.argv[8]) if len(sys.argv) > 8 else 88
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), name); os.makedirs(OUT, exist_ok=True)
    if orbit is None:     # pick the orbit with the most full-cover dates in the first year
        S = search(aoi, years[0]); cov = S[S.covers].groupby(['orbit', 'direction']).date.nunique().reset_index(name='full_cover_dates')
        cov.to_csv(os.path.join(OUT, 'orbits.csv'), index=False); print(cov.sort_values('full_cover_dates', ascending=False).to_string(index=False))
        orbit = int(cov.sort_values('full_cover_dates', ascending=False).iloc[0].orbit)
    pairs = []
    for year in years:
        S = search(aoi, year, orbit); S['date'] = pd.to_datetime(S.date); ok = S[S.covers].sort_values('date').drop_duplicates('date'); dates = list(ok.date)
        cand = [(a, b) for a in dates for b in dates if (b - a).days == 12]; used = set()
        for i, slot in enumerate(SLOTS[ref]):
            doy = pd.Timestamp(slot).dayofyear
            opts = sorted([c for c in cand if abs(c[0].dayofyear - doy) <= 14 and c[0] not in used and c[1] not in used], key=lambda c: abs(c[0].dayofyear - doy))
            if not opts: pairs.append(dict(year=year, orbit=orbit, slot=i, date1=None, date2=None, scene1=None, scene2=None)); continue
            c = opts[0]; used |= {c[0], c[1]}
            pairs.append(dict(year=year, orbit=orbit, slot=i, date1=c[0].date(), date2=c[1].date(),
                              scene1=ok[ok.date == c[0]].scene.iloc[0], scene2=ok[ok.date == c[1]].scene.iloc[0]))
        print(year, 'full-cover dates', len(dates), 'slots filled', sum(1 for p in pairs if p['year'] == year and p['scene1']), flush=True)
    P = pd.DataFrame(pairs); P.to_csv(os.path.join(OUT, 'plan_pairs.csv'), index=False); okp = P.dropna(subset=['scene1'])
    scenes = sorted(set(okp.scene1) | set(okp.scene2)); ifg = sorted({(a, b) for a, b in zip(okp.scene1, okp.scene2)})
    nm = lambda pre, *g: f'{pre}_' + hashlib.sha1('|'.join(g).encode()).hexdigest()[:12]
    pl = [dict(job_type='RTC_GAMMA', name=nm('RTC_GAMMA', s), job_parameters=dict(granules=[s], dem_matching=False, include_dem=False, include_inc_map=False,
          include_rgb=False, include_scattering_area=False, radiometry='gamma0', resolution=20, scale='power', speckle_filter=False, dem_name='copernicus')) for s in scenes]
    pl += [dict(job_type='INSAR_GAMMA', name=nm('INSAR_GAMMA', a, b), job_parameters=dict(granules=[a, b], include_look_vectors=False, include_los_displacement=False,
           include_inc_map=True, looks='10x2', include_dem=False, include_wrapped_phase=True, apply_water_mask=True, include_displacement_maps=False,
           phase_filter_parameter=0.6)) for a, b in ifg]
    plan = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), aoi=list(aoi.bounds), orbit=orbit, ref_calendar=ref, years=years,
                n_jobs=len(pl), n_rtc=len(scenes), n_insar=len(ifg), credits=sum(COST[p['job_type']] for p in pl),
                missing_slots=int(P.scene1.isna().sum()), payloads=pl, status='PLAN_ONLY_NOT_SUBMITTED')
    json.dump(plan, open(os.path.join(OUT, 'SUBMISSION_PLAN.json'), 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in plan.items() if k != 'payloads'}, indent=1, default=str))
