"""S3h supervisor: keeps every S3h job running until its log ends with DONE. Jobs are resumable (per-transfer CSV rows), so a job that
exits without DONE (e.g. out of memory) is restarted, at most MAXR times. A new or restarted job starts only when enough RAM is free.
When all jobs are done, runs s3h_summary.py. Log: s3h_supervise.log"""
import os, sys, time, subprocess, psutil
H = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable; MAXR = 8
JOBS = [('cpu', 'B88', 0, 2), ('cpu', 'B88', 1, 2), ('cpu', 'A37', 0, 1), ('cpu', 'A88', 0, 1)] + [('cpu', 'FR', i, 4) for i in range(4)] + \
       [('deep', 'B88', 0, 2), ('deep', 'B88', 1, 2), ('deep', 'A37', 0, 1), ('deep', 'A88', 0, 1)] + [('deep', 'FR', i, 3) for i in range(3)]
NEED_GB = {'cpu': 6.0, 'deep': 6.0}          # v2 (2026-11-11 01:40): all jobs were reaped at ~27 GB used; run at most 3 + 3
MAXRUN = {'cpu': 3, 'deep': 3}
JOBS = [j for j in JOBS if j[1] == 'FR'] + [j for j in JOBS if j[1] != 'FR']     # longest first


def name(j): k, g, p, n = j; return f"s3h_{k}_{g}" + ('' if n == 1 else f'_p{p}')
def args(j): k, g, p, n = j; return [f's3h_{k}.py', g] + ([] if n == 1 else [str(p), str(n)])
def done(j):
    f = os.path.join(H, name(j) + '.log')
    return os.path.exists(f) and 'DONE' in open(f, encoding='utf-8', errors='ignore').read()[-400:]


def running(j):
    a = args(j)
    for pr in psutil.process_iter(['cmdline']):
        try:
            c = pr.info['cmdline'] or []
            if any(x.endswith(a[0]) for x in c) and c[-len(a) + 1:] == a[1:]: return True
        except Exception: pass
    return False


def log(msg):
    with open(os.path.join(H, 's3h_supervise.log'), 'a', encoding='utf-8') as f: f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


restarts = {name(j): 0 for j in JOBS}; log('supervisor start')
while True:
    pending = [j for j in JOBS if not done(j)]
    if not pending: break
    for j in pending:
        if running(j) or restarts[name(j)] > MAXR: continue
        if sum(running(x) for x in JOBS if x[0] == j[0]) >= MAXRUN[j[0]]: continue
        free = psutil.virtual_memory().available / 2 ** 30
        if free < NEED_GB[j[0]]: continue
        nt = '4' if j[0] == 'cpu' else '1'
        env = dict(os.environ, OMP_NUM_THREADS=nt, MKL_NUM_THREADS=nt, OPENBLAS_NUM_THREADS=nt)
        lf = open(os.path.join(H, name(j) + '.log'), 'a', encoding='utf-8')
        subprocess.Popen([PY, '-u'] + args(j), cwd=H, env=env, stdout=lf, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        restarts[name(j)] += 1; log(f'start {name(j)} (#{restarts[name(j)]}), free {free:.1f} GB'); time.sleep(90)
    gave_up = [name(j) for j in pending if restarts[name(j)] > MAXR and not running(j)]
    if gave_up and len(gave_up) == len([j for j in JOBS if not done(j)]): log('gave up: ' + ', '.join(gave_up)); break
    time.sleep(120)
log('all jobs finished or given up; running summary')
with open(os.path.join(H, 's3h_summary.log'), 'w', encoding='utf-8') as f:
    subprocess.run([PY, '-u', 's3h_summary.py'], cwd=H, stdout=f, stderr=subprocess.STDOUT)
log('summary written'); print('SUPERVISOR_DONE')
