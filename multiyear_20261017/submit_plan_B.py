"""Submit SUBMISSION_PLAN_B.json to HyP3. Authorized by user 2026-10-17 ("同意" to ordering; "方案 B").
Checks: exactly 372 payloads and 5580 credits; balance >= cost. Writes SUBMITTED_JOBS_B.csv and SUBMISSION_RECEIPT_B.json."""
import os, csv, json, time
import hyp3_sdk
OUT = os.path.dirname(os.path.abspath(__file__))
plan = json.load(open(os.path.join(OUT, 'SUBMISSION_PLAN_B.json')))
payloads = plan['payloads']
assert len(payloads) == 372 and plan['credits'] == 5580, (len(payloads), plan['credits'])
h = hyp3_sdk.HyP3(); c0 = h.check_credits(); print('balance before', c0, flush=True)
assert c0 >= plan['credits'], f'insufficient credits {c0}'
recs = []
for i in range(0, len(payloads), 25):
    batch = h.submit_prepared_jobs(payloads[i:i + 25])
    for j in batch:
        recs.append(dict(job_id=j.job_id, name=j.name, job_type=j.job_type, status_code=j.status_code, request_time=str(j.request_time)))
    print('submitted', len(recs), flush=True); time.sleep(1)
c1 = h.check_credits()
with open(os.path.join(OUT, 'SUBMITTED_JOBS_B.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
json.dump(dict(submission_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), hyp3_sdk=hyp3_sdk.__version__, n_jobs=len(recs),
               credits_before=c0, credits_after=c1, spent=c0 - c1, expected=plan['credits']), open(os.path.join(OUT, 'SUBMISSION_RECEIPT_B.json'), 'w'), indent=1)
print('done', len(recs), 'jobs; spent', c0 - c1, 'credits; balance', c1)
