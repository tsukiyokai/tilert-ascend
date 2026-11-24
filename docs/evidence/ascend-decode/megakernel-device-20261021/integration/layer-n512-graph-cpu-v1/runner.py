import hashlib, io, json, shlex, subprocess, tarfile
from pathlib import Path
root=Path('/tmp/tilert-megakernel-validation-20261021')
p=Path('/tmp/tilert-layer-n512-graph-staging-v1')
assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()=='6988778ffa2a0b593ed8924d3386a9a2efb6ffa2a6a2957781f1dced56ce49a5'
for line in (p/'frozen-inputs.sha256').read_text().splitlines():
    expected,name=line.split('  ',1)
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
remote='/home/shanshan/tilert-megakernel-20261021/vendor-hook-layer-n512-graph-v1'
ssh=['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-194']
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
    for child in p.iterdir(): archive.add(child,arcname=child.name)
subprocess.run(ssh+['mkdir',remote],check=True)
subprocess.run(ssh+[shlex.join(['tar','-xzf','-','-C',remote])],input=buffer.getvalue(),check=True)
subprocess.run(ssh+[shlex.join(['bash','-c','cd '+shlex.quote(remote)+' && sha256sum --quiet -c frozen-inputs.sha256'])],check=True)
name='layer-n512-graph-cpu-v1'
local=root/name
local.mkdir()
command=json.loads((root/'local-graph-cpu-v3/command.json').read_text())
command=[x.replace('vendor-hook-local-graph-v3','vendor-hook-layer-n512-graph-v1').replace('local-graph-cpu-v3',name) for x in command]
(local/'command.json').write_text(json.dumps(command,indent=2)+'\n')
remote_out='/home/shanshan/tilert-megakernel-20261021/'+name
subprocess.run(ssh+['mkdir',remote_out],check=True)
with (local/'run.log').open('w') as log:
    run=subprocess.run(ssh+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
fetch=subprocess.run(ssh+['cat',remote_out+'/report.json'],capture_output=True)
if fetch.returncode==0:
    (local/'report.json').write_bytes(fetch.stdout)
print('Layer N512 graph CPU gate exit',run.returncode,flush=True)
raise SystemExit(run.returncode)
