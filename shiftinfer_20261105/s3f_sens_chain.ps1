param([string]$g)
$env:OMP_NUM_THREADS = '4'; Set-Location E:\research\SAR\shiftinfer_20261105; $PY = 'D:\anaconda\envs\yolo11marin\python.exe'
while (-not (Test-Path "s3f_pred_$g.csv.gz")) { Start-Sleep -Seconds 20 }
$cfg = @('b 0','b 0.5','b 1.5','b 2','lam 0','lam 0.05','lam 0.2','lam 0.3','nit 10','nit 20','nit 50')
foreach ($c in $cfg) { $a = $c.Split(' '); & $PY -u s3f_core.py sens $g $a[0] $a[1] *>> "s3f_sens_$g.log" }
"DONE $g" | Out-File "s3f_sens_$g.done"
