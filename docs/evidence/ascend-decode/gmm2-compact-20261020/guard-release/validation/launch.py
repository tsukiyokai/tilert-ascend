import datetime,json,pathlib,re,subprocess,time
root=pathlib.Path('/home/shanshan/trt-asc-194/experiments/gmm2-compact-20261020/guard-release-validation')
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
(root/'screen-01-preflight.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'die':0,'samples':checks},indent=2))
assert all(c['aicore']==0 and c['free_mib']>=6000 for c in checks),checks
command='timeout --kill-after=10 300 bash /workspace/experiments/gmm2-compact-20261020/guard-release-validation/run-validation.sh npu:0'
with (root/'screen-01.log').open('w') as log:
    result=subprocess.run(['docker','exec','trt-asc','bash','-lc',command],stdout=log,stderr=subprocess.STDOUT)
(root/'screen-01-exit.json').write_text(json.dumps({'returncode':result.returncode}))
print(json.dumps({'returncode':result.returncode,'preflight':checks}))
raise SystemExit(result.returncode)
