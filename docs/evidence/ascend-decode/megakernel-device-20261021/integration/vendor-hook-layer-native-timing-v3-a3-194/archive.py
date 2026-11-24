import io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
root=Path('/tmp/tilert-megakernel-validation-20261021')
name=sys.argv[1]
assert name.startswith(('vendor-hook-local-chain-', 'vendor-hook-local-graph-', 'vendor-hook-topk-owner-', 'vendor-hook-layer-n512-', 'vendor-hook-layer-pair-', 'vendor-hook-layer-native-', 'vendor-hook-tp-chain-')) and all(c.isalnum() or c=='-' for c in name)
task='/home/shanshan/tilert-megakernel-20261021'
collect=('from pathlib import Path; import sys,tarfile; p=Path('+repr(task+'/logs')+'); names='+repr([name,name+'-driver',name+'-extra-info'])+'; t=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz"); [t.add(p/n,arcname=n) for n in names if (p/n).exists()]; t.close()')
raw=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','a3-194','python3','-c',shlex.quote(collect)])
dest=root/(name+'-raw')
dest.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 archive.extractall(dest,filter='data')
files=[p for p in dest.rglob('*') if p.is_file()]
print(json.dumps({'destination':str(dest),'files':len(files),'bytes':sum(p.stat().st_size for p in files)}))
