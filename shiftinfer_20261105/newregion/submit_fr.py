"""Submit FR-A HyP3 jobs for the given years from FR-A_PaysDeCaux/SUBMISSION_PLAN.json (payloads identical to the NL plans).
User approval 2026-11-06 ("选法国FR-A"; e:\\组会.txt item 3: approve, smoke 1 year first, then the remaining 7 years).
Usage: python submit_fr.py 2023  -> FR-A_PaysDeCaux/SUBMITTED_JOBS.csv (appended), SUBMISSION_RECEIPT_<years>.json"""
import os, sys, csv, json, time
import pandas as pd, hyp3_sdk
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FR-A_PaysDeCaux')
YEARS = [int(y) for y in sys.argv[1:]]; assert YEARS
plan = json.load(open(os.path.join(D, 'SUBMISSION_PLAN.json'))); P = pd.read_csv(os.path.join(D, 'plan_pairs.csv')).dropna(subset=['scene1'])
P = P[P.year.isin(YEARS)]; scenes = set(P.scene1) | set(P.scene2); ifg = {(a, b) for a, b in zip(P.scene1, P.scene2)}
sub_p = os.path.join(D, 'SUBMITTED_JOBS.csv'); done = {r['name'] for r in csv.DictReader(open(sub_p, encoding='utf-8'))} if os.path.exists(sub_p) else set()
pl = [p for p in plan['payloads'] if p['name'] not in done and ((p['job_type'] == 'RTC_GAMMA' and p['job_parameters']['granules'][0] in scenes) or
      (p['job_type'] == 'INSAR_GAMMA' and tuple(p['job_parameters']['granules']) in ifg))]
cost = 15 * len(pl); h = hyp3_sdk.HyP3(); c0 = h.check_credits()
print('years', YEARS, 'jobs', len(pl), 'expected cost', cost, 'balance', c0, flush=True); assert c0 >= cost
recs = []
for i in range(0, len(pl), 25):
    for j in h.submit_prepared_jobs(pl[i:i + 25]):
        recs.append(dict(job_id=j.job_id, name=j.name, job_type=j.job_type, status_code=j.status_code, request_time=str(j.request_time), years='_'.join(map(str, YEARS))))
    time.sleep(1)
c1 = h.check_credits(); new = not os.path.exists(sub_p)
with open(sub_p, 'a', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0]))
    if new: w.writeheader()
    w.writerows(recs)
json.dump(dict(submission_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), years=YEARS, n_jobs=len(recs), credits_before=c0, credits_after=c1,
               spent=c0 - c1, expected=cost, hyp3_sdk=hyp3_sdk.__version__), open(os.path.join(D, f"SUBMISSION_RECEIPT_{'_'.join(map(str, YEARS))}.json"), 'w'), indent=1)
print('submitted', len(recs), 'spent', c0 - c1, 'balance', c1)
