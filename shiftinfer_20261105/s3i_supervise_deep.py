"""S3i deep supervisor (amendment 2): keeps every s3i_deep.py job running until its log ends with DONE (resumable per transfer; restart at most MAXR times).
Concurrency: 6 GPU jobs (1 CPU thread each); a job starts only with >= NEED_GB free RAM.
When all are done, runs s3i_summary.py -> s3i_summary_deep.log. Log: s3i_supervise_deep.log"""
import os, sys, time, subprocess, psutil
H = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable; MAXR = 8; NEED_GB = 7.0
JOBS = [('FR', i, 4) for i in range(4)] + [('B88', i, 2) for i in range(2)] + [('A37', i, 2) for i in range(2)] + [('A88', i, 2) for i in range(2)]


def name(j): g, p, n = j; return f's3i_deep_{g}_p{p}'
def args(j): g, p, n = j; return ['s3i_deep.py', g, str(p), str(n)]
def done(j):
    f = os.path.join(H, name(j) + '.log')
    return os.path.exists(f) and 'DONE' in open(f, encoding='utf-8', errors='ignore').read()[-400:]


def procs():
    out = []
    for pr in psutil.process_iter(['cmdline']):
        try: out.append(pr.info['cmdline'] or [])
        except Exception: pass
    return out


def running(j, P): a = args(j); return any(any(x.endswith(a[0]) for x in c) and c[-3:] == a[1:] for c in P)
def log(msg):
    with open(os.path.join(H, 's3i_supervise_deep.log'), 'a', encoding='utf-8') as f: f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + msg + '\n')


restarts = {name(j): 0 for j in JOBS}; log('supervisor start')
while True:
    pending = [j for j in JOBS if not done(j)]
    if not pending: break
    for j in pending:
        P = procs()
        if running(j, P) or restarts[name(j)] > MAXR: continue
        gpu = False
        n_run = sum(any(x.endswith('s3i_deep.py') for x in c) for c in P)
        if n_run >= 4: continue   # v2 (07:25): reaped at 6 jobs; 4 jobs, 7 GB gate
        free = psutil.virtual_memory().available / 2 ** 30
        if free < NEED_GB: continue
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        lf = open(os.path.join(H, name(j) + '.log'), 'a', encoding='utf-8')
        subprocess.Popen([PY, '-u'] + args(j), cwd=H, env=env, stdout=lf, stderr=subprocess.STDOUT,
                         creationflags=0x00000008 | getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        restarts[name(j)] += 1; log(f'start {name(j)} (#{restarts[name(j)]}), free {free:.1f} GB, gpu jobs running {gpu}'); time.sleep(60)
    P = procs(); gave_up = [name(j) for j in pending if restarts[name(j)] > MAXR and not running(j, P)]
    if gave_up and len(gave_up) == len([j for j in JOBS if not done(j)]): log('gave up: ' + ', '.join(gave_up)); break
    time.sleep(120)
log('all jobs finished or given up; running summary')
with open(os.path.join(H, 's3i_summary_deep.log'), 'w', encoding='utf-8') as f:
    subprocess.run([PY, '-u', 's3i_summary.py'], cwd=H, stdout=f, stderr=subprocess.STDOUT)
log('summary written'); print('SUPERVISOR_DONE')
