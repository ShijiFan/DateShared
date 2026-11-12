"""Poll plan-B HyP3 jobs and download finished products to D:\\sar_data\\multiyear\\downloads (never submits).
Resumable; stops if free disk < 40 GB; writes JOB_STATUS_B.json and FAILED_JOBS_B.csv."""
import os, csv, json, shutil, time
import hyp3_sdk
OUT = os.path.dirname(os.path.abspath(__file__)); DL = r'D:\sar_data\multiyear\downloads'; os.makedirs(DL, exist_ok=True)
ids = [r['job_id'] for r in csv.DictReader(open(os.path.join(OUT, 'SUBMITTED_JOBS_B.csv'), encoding='utf-8'))]
h = hyp3_sdk.HyP3(); t_end = time.time() + 30 * 3600
while True:
    jobs = []
    for i in range(0, len(ids), 50):
        jobs += [h.get_job_by_id(x) for x in ids[i:i + 50]]
    counts = {}
    for j in jobs: counts[j.status_code] = counts.get(j.status_code, 0) + 1
    print(time.strftime('%H:%M:%S'), counts, flush=True)
    for j in jobs:
        if j.status_code != 'SUCCEEDED': continue
        for fd in j.files or []:
            tgt = os.path.join(DL, fd['filename'])
            if os.path.exists(tgt) and os.path.getsize(tgt) == fd.get('size', os.path.getsize(tgt)): continue
            free = shutil.disk_usage(DL).free / 1e9
            if free < 40: print('STOP: disk', free); raise SystemExit(1)
            try: j.download_files(DL)
            except Exception as e: print('download error', j.job_id, repr(e), flush=True)
    idx = [dict(job_id=j.job_id, name=j.name, job_type=j.job_type, status=j.status_code, files=';'.join(f['filename'] for f in (j.files or [])),
                granules=';'.join((j.job_parameters or {}).get('granules', []))) for j in jobs]
    json.dump(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), counts=counts, jobs=idx), open(os.path.join(OUT, 'JOB_STATUS_B.json'), 'w'), indent=1)
    with open(os.path.join(OUT, 'FAILED_JOBS_B.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0])); w.writeheader(); w.writerows([r for r in idx if r['status'] == 'FAILED'])
    if not any(j.status_code in ('PENDING', 'RUNNING') for j in jobs): print('all finished'); break
    if time.time() > t_end: print('timeout'); break
    time.sleep(600)
