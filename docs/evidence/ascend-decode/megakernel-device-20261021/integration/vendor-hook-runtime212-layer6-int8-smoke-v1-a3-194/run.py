import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=('bf16','int8'))
parser.add_argument('phase',choices=('smoke','graph'))
args=parser.parse_args()
root=Path('/tmp/tilert-megakernel-validation-20261021')
spec=importlib.util.spec_from_file_location('idle_gate',root/'run-pending-single-die.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
gate.SSH[-1]='a3-194'
runtime='/home/shanshan/tilert-megakernel-20261021/vendor-hook-runtime212'
identities={
 'tests/glm52/whole_layer/vendor_hook_runtime212_run.sh':'8ff2b03d8ec6832d41f9a2712a9f650de88de7f6a92ce80b84b3b8fe34f15d2b',
 'tests/glm52/whole_layer/vendor_hook_runtime212_device.py':'1d579efa0c6da63486af190ae8d1f049d1809b91daba1296e8e8c0e5a125571d',
 'frozen-inputs.sha256':'7c9dd7d290e20be920a6a2208e1799cacef7d529b6ae6e0614ef8c17fa97b8c8',
 'staging-manifest.json':'a5e57f7c3ca4205e53d94b507f921936e307f4313e4d3b522bfa31f9b3737332',
 'native/indexed/whole_layer_indexed_attention.so':'e957650c7def45dff8b918b74c987a8ce44eb0b2cca816e7246cbbff8a52378e',
 'native/shared/whole_layer_full_attention.so':'64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c',
 'native/sfa/vendor_sfa_adapter.so':'bd8aff02f36dca1948b4eea1aa03805d6977b5ae8bf39573b51d73b68a61be3e',
}
for path,expected in identities.items():
 actual=subprocess.check_output(gate.SSH+['sha256sum',runtime+'/'+path],text=True).split()[0]
 assert actual==expected,(path,actual)
assert gate.vacant(),'Initial idle gate refused'
time.sleep(5)
assert gate.vacant(),'Second idle gate refused'
users=subprocess.run(gate.SSH+['fuser','/dev/davinci4'],capture_output=True,text=True)
assert users.returncode==1 and not users.stdout.strip() and not users.stderr.strip(),users
name='vendor-hook-runtime212-layer6-'+args.mode+'-'+args.phase+'-v1'
options=['--forward-widths','1','--forward-replays','2'] if args.phase=='smoke' else ['--forward-widths','1','2','12','32','--forward-replays','8','--graph']
command=['bash',runtime+'/tests/glm52/whole_layer/vendor_hook_runtime212_run.sh','4',name,'--mode',args.mode,*options]
(root/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
print('Starting real layer-6 attention',args.mode,args.phase,'under TorchNPU212 on physical die4',flush=True)
with (root/(name+'.launcher.log')).open('w') as output:
 result=subprocess.run(gate.SSH+[shlex.join(command)],stdout=output,stderr=subprocess.STDOUT)
print('Runtime212 attention exit',result.returncode,flush=True)
raise SystemExit(result.returncode)
