"""S3h amendment: finishes the FR deep poolx jobs after part 2/3 was split into 2/6 and 5/6 (2026-11-11 23:17, see s3h_supervise2.log).
Restarts a job that exits without DONE (at most 3 times, when >= 6 GB RAM is free), then runs s3h_summary.py -> s3h_summary2.log."""
import os, sys, time, subprocess, psutil
H = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
JOBS = {('0', '3'): 's3h_poolx_deep_FR_p0.log', ('1', '3'): 's3h_poolx_deep_FR_p1.log',
        ('2', '6'): 's3h_poolx_deep_FR_p2of6.log', ('5', '6'): 's3h_poolx_deep_FR_p5of6.log'}
restarts = {j: 0 for j in JOBS}
log = lambda m: open(os.path.join(H, 's3h_supervise2.log'), 'a', encoding='utf-8').write(time.strftime('%Y-%m-%d %H:%M:%S ') + m + '\n')


def done(j): f = os.path.join(H, JOBS[j]); return os.path.exists(f) and 'DONE' in open(f, encoding='utf-8', errors='ignore').read()[-400:]


def running(j):
    for p in psutil.process_iter(['cmdline']):
        try:
            c = p.info['cmdline'] or []
            if any(x.endswith('s3h_poolx.py') for x in c) and c[-4:] == ['deep', 'FR', j[0], j[1]]: return True
        except Exception: pass
    return False


log('s3h_finish start')
while True:
    pend = [j for j in JOBS if not done(j)]
    if not pend: break
    for j in pend:
        if running(j) or restarts[j] >= 3 or psutil.virtual_memory().available / 2 ** 30 < 6: continue
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        subprocess.Popen([PY, '-u', 's3h_poolx.py', 'deep', 'FR', j[0], j[1]], cwd=H, env=env, stdout=open(os.path.join(H, JOBS[j]), 'a', encoding='utf-8'),
                         stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        restarts[j] += 1; log(f'restart FR deep part {j[0]}/{j[1]} (#{restarts[j]})'); time.sleep(60)
    if all(restarts[j] >= 3 and not running(j) for j in pend): log('gave up: ' + str(pend)); break
    time.sleep(120)
log('FR deep poolx finished; running summary')
with open(os.path.join(H, 's3h_summary2.log'), 'w', encoding='utf-8') as f:
    subprocess.run([PY, '-u', 's3h_summary.py'], cwd=H, stdout=f, stderr=subprocess.STDOUT)
log('summary written'); print('FINISH_DONE')
