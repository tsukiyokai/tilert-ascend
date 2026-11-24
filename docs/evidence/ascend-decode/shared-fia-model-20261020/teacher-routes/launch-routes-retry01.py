import datetime,json,pathlib,re,subprocess,time
root=pathlib.Path('/home/shanshan/trt-asc-194/experiments/teacher-input-router-20261020')
checks=[]
for _ in range(2):
    raw=subprocess.check_output(['npu-smi','info'],text=True)
    lines=[]
    for line in raw.splitlines():
        parts=[v.strip() for v in line.split('|') if v.strip()]
        if len(parts)==3 and re.fullmatch(r'\d+\s+0',parts[0]):
            used,total=map(int,re.search(r'(\d+)\s*/\s*(\d+)\s*$',parts[2]).groups())
            lines.append({'aicore':int(parts[2].split()[0]),'free_mib':total-used})
    assert len(lines)==1
    checks.append(lines[0])
    time.sleep(1)
(root/'routes-retry01-preflight-die0.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'die':0,'samples':checks},indent=2))
assert all(c['aicore']==0 and c['free_mib']>=6000 for c in checks),checks
command='source /workspace/environment/activate.sh; set -e; ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_PROCESS_LOG_PATH=/workspace/experiments/teacher-input-router-20261020/routes-logs TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --kill-after=10 300 /opt/trt-asc/venv/bin/python3 /workspace/experiments/teacher-input-router-20261020/preload_routes.py --phase routes --die 0'
with (root/'routes-retry01.log').open('w') as log:
    result=subprocess.run(['docker','exec','trt-asc','bash','-lc',command],stdout=log,stderr=subprocess.STDOUT)
(root/'routes-retry01-exit.json').write_text(json.dumps({'returncode':result.returncode}))
print(json.dumps({'returncode':result.returncode,'preflight':checks}))
raise SystemExit(result.returncode)
