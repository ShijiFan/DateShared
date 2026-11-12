"""Parallel replacement for download_B.py (sequential download was ~2.3 min per 1 GB zip -> >9 h). Same job list, same target folder,
never submits. Each file goes to <name>.part and is renamed after the size matches; existing complete files are skipped.
Stops if free disk < 40 GB. Writes JOB_STATUS_B.json / FAILED_JOBS_B.csv like download_B.py."""
import os, csv, json, shutil, time
from concurrent.futures import ThreadPoolExecutor
import requests, hyp3_sdk
OUT = os.path.dirname(os.path.abspath(__file__)); DL = r'D:\sar_data\multiyear\downloads'; NW = 6
ids = [r['job_id'] for r in csv.DictReader(open(os.path.join(OUT, 'SUBMITTED_JOBS_B.csv'), encoding='utf-8'))]
h = hyp3_sdk.HyP3(); t_end = time.time() + 30 * 3600
def get(fd):
    tgt = os.path.join(DL, fd['filename'])
    if os.path.exists(tgt) and os.path.getsize(tgt) == fd['size']: return 'skip'
    dz = os.path.join(OUT, 'DELETED_ZIPS.csv')                  # aligned + deleted with user approval: never re-download
    if os.path.exists(dz) and fd['filename'] in {r['zip'] for r in csv.DictReader(open(dz, encoding='utf-8'))}: return 'skip'
    if shutil.disk_usage(DL).free / 1e9 < 40: return 'disk'
    part = tgt + '.part'
    for attempt in range(3):
        try:
            with requests.get(fd['url'], stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(part, 'wb') as f:
                    for c in r.iter_content(8 << 20): f.write(c)
            if os.path.getsize(part) == fd['size']: os.replace(part, tgt); return 'ok'
        except Exception as e:
            print('retry', fd['filename'], repr(e), flush=True)
    return 'fail'
while True:
    jobs = [h.get_job_by_id(x) for x in ids]
    counts = {}
    for j in jobs: counts[j.status_code] = counts.get(j.status_code, 0) + 1
    print(time.strftime('%H:%M:%S'), counts, flush=True)
    fds = [fd for j in jobs if j.status_code == 'SUCCEEDED' for fd in (j.files or [])]
    import re
    fds.sort(key=lambda fd: (re.search(r'_(\d{8})T', fd['filename']).group(1)[:4], fd['filename']))   # year by year -> year-tracks complete progressively
    with ThreadPoolExecutor(NW) as ex: res = list(ex.map(get, fds))
    print(time.strftime('%H:%M:%S'), {k: res.count(k) for k in set(res)}, flush=True)
    idx = [dict(job_id=j.job_id, name=j.name, job_type=j.job_type, status=j.status_code, files=';'.join(f['filename'] for f in (j.files or [])),
                granules=';'.join((j.job_parameters or {}).get('granules', []))) for j in jobs]
    json.dump(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), counts=counts, jobs=idx), open(os.path.join(OUT, 'JOB_STATUS_B.json'), 'w'), indent=1)
    with open(os.path.join(OUT, 'FAILED_JOBS_B.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0])); w.writeheader(); w.writerows([r for r in idx if r['status'] == 'FAILED'])
    if 'disk' in res: print('STOP: disk'); raise SystemExit(1)
    if not any(j.status_code in ('PENDING', 'RUNNING') for j in jobs) and 'fail' not in res: print('all finished'); break
    if time.time() > t_end: print('timeout'); break
    time.sleep(120 if 'ok' in res else 600)
