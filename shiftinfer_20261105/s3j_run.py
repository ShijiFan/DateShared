"""S3j runner: keeps s3j_cpu/s3j_deep jobs going until each log ends with DONE; at most 2 cpu + 2 deep at once, start only with >= 6 GB free.
Restarts a job that exits without DONE (resumable), at most 6 times. Log: s3j_run.log"""
import os, sys, time, subprocess, psutil
H = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
JOBS = [(k, g) for k in ('cpu', 'deep') for g in ('B88', 'A37', 'A88')]; R = {j: 0 for j in JOBS}
log = lambda m: open(os.path.join(H, 's3j_run.log'), 'a', encoding='utf-8').write(time.strftime('%Y-%m-%d %H:%M:%S ') + m + '\n')
def done(j): f = os.path.join(H, f's3j_{j[0]}_{j[1]}.log'); return os.path.exists(f) and 'DONE' in open(f, encoding='utf-8', errors='ignore').read()[-300:]
def procs():
    out = []
    for p in psutil.process_iter(['cmdline']):
        try: out.append(p.info['cmdline'] or [])
        except Exception: pass
    return out
log('runner start')
while True:
    pend = [j for j in JOBS if not done(j)]
    if not pend: break
    P = procs()
    for j in pend:
        if any(any(x.endswith(f's3j_{j[0]}.py') for x in c) and c[-1] == j[1] for c in P) or R[j] >= 6: continue
        if sum(any(x.endswith(f's3j_{j[0]}.py') for x in c) for c in P) >= 2: continue
        free = psutil.virtual_memory().available / 2 ** 30
        if free < 6: continue
        nt = '4' if j[0] == 'cpu' else '1'; env = dict(os.environ, OMP_NUM_THREADS=nt, MKL_NUM_THREADS=nt, OPENBLAS_NUM_THREADS=nt)
        subprocess.Popen([PY, '-u', f's3j_{j[0]}.py', j[1]], cwd=H, env=env, stdout=open(os.path.join(H, f's3j_{j[0]}_{j[1]}.log'), 'a', encoding='utf-8'),
                         stderr=subprocess.STDOUT, creationflags=0x00000008 | 0x08000000)
        R[j] += 1; log(f'start s3j_{j[0]} {j[1]} (#{R[j]}), free {free:.1f} GB'); time.sleep(60); P = procs()
    if all(R[j] >= 6 for j in pend) and not any(any(x.endswith(('s3j_cpu.py', 's3j_deep.py')) for x in c) for c in procs()): log('gave up'); break
    time.sleep(90)
log('all done')
