"""Plan C = T37 Site B, 2017-2023 (user 2026-10-18: "补"). Credits now 1970 < 2640, so this run submits only the years given on the
command line (default 2017-2021, 1875 credits); 2022-2023 (765 credits) after the monthly credit reset:  python submit_plan_C.py 2022 2023
Payloads identical to plan B / A1 (built by the same code as make_plan_B.py). Scenes/pairs already ordered in plan B are not re-ordered.
Writes plan_pairs_C.csv, SUBMISSION_PLAN_C_<years>.json, SUBMITTED_JOBS_C.csv (appended), SUBMISSION_RECEIPT_C_<years>.json."""
import os, sys, csv, json, time, hashlib
import pandas as pd, hyp3_sdk
OUT = os.path.dirname(os.path.abspath(__file__))
YEARS = [int(y) for y in sys.argv[1:]] or [2017, 2018, 2019, 2020, 2021]
P = pd.read_csv(os.path.join(OUT, 'plan_pairs.csv')).dropna(subset=['scene1']); B = pd.read_csv(os.path.join(OUT, 'plan_pairs_B.csv'))
C = P[(P.track == 37) & (P.site == 'B')]; C.to_csv(os.path.join(OUT, 'plan_pairs_C.csv'), index=False)
C = C[C.year.isin(YEARS)]
done_sc = set(B.scene1) | set(B.scene2); done_if = {(a, b) for a, b in zip(B.scene1, B.scene2)}
scenes = sorted((set(C.scene1) | set(C.scene2)) - done_sc); ifg = sorted({(a, b) for a, b in zip(C.scene1, C.scene2)} - done_if)
def name(prefix, *g): return f'{prefix}_' + hashlib.sha1('|'.join(g).encode()).hexdigest()[:12]
payloads = [dict(job_type='RTC_GAMMA', name=name('RTC_GAMMA', s), job_parameters=dict(granules=[s], dem_matching=False, include_dem=False, include_inc_map=False,
            include_rgb=False, include_scattering_area=False, radiometry='gamma0', resolution=20, scale='power', speckle_filter=False, dem_name='copernicus')) for s in scenes]
payloads += [dict(job_type='INSAR_GAMMA', name=name('INSAR_GAMMA', a, b), job_parameters=dict(granules=[a, b], include_look_vectors=False, include_los_displacement=False,
             include_inc_map=True, looks='10x2', include_dem=False, include_wrapped_phase=True, apply_water_mask=True, include_displacement_maps=False,
             phase_filter_parameter=0.6)) for a, b in ifg]
cost = 15 * len(payloads); tag = '_'.join(map(str, YEARS))
json.dump(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), plan=f'C: T37 site B {YEARS}', n_rtc=len(scenes), n_insar=len(ifg), credits=cost,
               payloads=payloads), open(os.path.join(OUT, f'SUBMISSION_PLAN_C_{tag}.json'), 'w'), indent=1)
h = hyp3_sdk.HyP3(); c0 = h.check_credits(); print('years', YEARS, 'jobs', len(payloads), 'cost', cost, 'balance', c0, flush=True)
assert c0 >= cost, f'insufficient credits {c0} < {cost}'
recs = []
for i in range(0, len(payloads), 25):
    for j in h.submit_prepared_jobs(payloads[i:i + 25]):
        recs.append(dict(job_id=j.job_id, name=j.name, job_type=j.job_type, status_code=j.status_code, request_time=str(j.request_time), years=tag))
    print('submitted', len(recs), flush=True); time.sleep(1)
c1 = h.check_credits(); p = os.path.join(OUT, 'SUBMITTED_JOBS_C.csv'); new = not os.path.exists(p)
with open(p, 'a', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0]))
    if new: w.writeheader()
    w.writerows(recs)
json.dump(dict(submission_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), years=YEARS, n_jobs=len(recs), credits_before=c0, credits_after=c1,
               spent=c0 - c1, expected=cost), open(os.path.join(OUT, f'SUBMISSION_RECEIPT_C_{tag}.json'), 'w'), indent=1)
print('done', len(recs), 'jobs; spent', c0 - c1, '; balance', c1)
