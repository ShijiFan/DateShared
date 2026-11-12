"""Plan B (user chose 2026-10-17): T88 both sites + T37 Site A, 2017-2023. Builds SUBMISSION_PLAN_B.json from plan_pairs.csv
(payload parameters identical to A1). Does not submit."""
import os, json, hashlib, time
import pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
P = pd.read_csv(os.path.join(OUT, 'plan_pairs.csv')).dropna(subset=['scene1'])
B = P[(P.track == 88) | ((P.track == 37) & (P.site == 'A'))]
B.to_csv(os.path.join(OUT, 'plan_pairs_B.csv'), index=False)
scenes = sorted(set(B.scene1) | set(B.scene2)); ifg = sorted({(a, b) for a, b in zip(B.scene1, B.scene2)})
def name(prefix, *g): return f'{prefix}_' + hashlib.sha1('|'.join(g).encode()).hexdigest()[:12]
payloads = [dict(job_type='RTC_GAMMA', name=name('RTC_GAMMA', s), job_parameters=dict(granules=[s], dem_matching=False, include_dem=False, include_inc_map=False,
            include_rgb=False, include_scattering_area=False, radiometry='gamma0', resolution=20, scale='power', speckle_filter=False, dem_name='copernicus')) for s in scenes]
payloads += [dict(job_type='INSAR_GAMMA', name=name('INSAR_GAMMA', a, b), job_parameters=dict(granules=[a, b], include_look_vectors=False, include_los_displacement=False,
             include_inc_map=True, looks='10x2', include_dem=False, include_wrapped_phase=True, apply_water_mask=True, include_displacement_maps=False,
             phase_filter_parameter=0.6)) for a, b in ifg]
plan = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), plan='B: T88 both sites + T37 site A, 2017-2023',
            n_jobs=len(payloads), n_rtc=len(scenes), n_insar=len(ifg), credits=15 * len(payloads),
            pairs_per_year_track=B.groupby(['year', 'track']).apply(lambda g: len({(a, b) for a, b in zip(g.scene1, g.scene2)})).to_dict(),
            payloads=payloads, status='PLAN_B_READY')
plan['pairs_per_year_track'] = {f'{k[0]}_T{k[1]}': v for k, v in plan['pairs_per_year_track'].items()}
json.dump(plan, open(os.path.join(OUT, 'SUBMISSION_PLAN_B.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in plan.items() if k != 'payloads'}, indent=1))
