import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('bf16', 'int8'))
args = parser.parse_args()
root = Path('/tmp/tilert-megakernel-validation-20261021')
spec = importlib.util.spec_from_file_location('idle_gate', root / 'run-pending-single-die.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
remote = '/home/shanshan/tilert-megakernel-20261021'
runtime = remote + '/vendor-hook-runtime'
identities = {'tilert/models/glm_5_2_ascend/attention_native.py': '336eaff0de153d32439d4d2b0557522b313a24a7e5bef82b8dc11f1257511c01', 'vendor-hook-pinned/ascend-python-build/vllm_ascend/attention/indexer.py': '47a47b06cf144bda4a87e80ef632144eeb86a4547e17d09f69f011ef565baefa', 'vendor-hook-pinned/ascend-python-build/vllm_ascend/ops/triton/rope.py': '82422c46f93c0962673ad51bc77fdd5caef412f45f2ff8b20bbeb20a4407f64b', 'vendor-hook-pinned/ascend-python-build/vllm_ascend/ops/linear.py': '7654777d9252fc6a3cc2c95f92b67a42410a0de24a5912ef20e807cf64ab74e6', 'vendor-hook-pinned/vllm-a982/vllm/model_executor/layers/layernorm.py': '7be031e88921fb21661f3b9ee28b4ccb7746405e0ff32ed5895d75b43d3357ed', 'vendor-hook-pinned/vllm-a982/vllm/model_executor/custom_op.py': 'caaa422b914d74a58c3d7a1653af76e2834f41cc2d4fbe65d10d4ad50dbb4d38', 'tests/glm52/whole_layer/vendor_hook_run.sh': 'a18be18f28fbc2bab670052edb0f3ed80606f6e5f0f9aed6302e3e8a4ed7d495', 'tests/glm52/whole_layer/vendor_hook_device.py': '16d1b936af443debbf13fbe81580588b3de144527c877424fdda215d817e12ad', 'tests/glm52/whole_layer/vendor_hook_forward.py': '376411d2c57501ac1661fb81b0d6f7c2ec571c9d83d2e8b65ddcbb0a68ba4b58', 'tests/glm52/whole_layer/vendor_hook_index_diagnostic.py': '382aa5cdc23dacf3617bacd4de94c7ba37e273f894863842f5351d9b63bbfbe6', 'tests/glm52/whole_layer/vendor_hook_diagnostic.py': 'd797c12d06fa714f9aa4756115020b3ae822d76e8ba17497aeec3b350c5e76b3', 'tests/glm52/whole_layer/vendor_hook_graph.py': '17fa65a25911db777e10ed335400e772f4a336d94255f92ae419a26e3eee992c', 'logs/attention-wk-kshift-v1/indexed/whole_layer_indexed_attention.so': '20095457af00d9299fcff884d806b03400669adfe3304750ca9ae97983a40ca8', 'logs/attention-wk-kshift-v1/indexed/manifest.json': '064e3f6bad0e9f48b45b2ba5ad8c3f705a9cc3b6a2c433a596f76c026afe7221', 'logs/attention-o-combined-v1/shared/whole_layer_full_attention.so': 'dd66e68b9c8f46c4fdee987d7b7bb29a062e3dcae5481237c45bc21544d346b9'}
for path, expected in identities.items():
    actual = subprocess.check_output(gate.SSH + ['sha256sum', runtime + '/' + path], text=True).split()[0]
    assert actual == expected, (path, actual)
assert gate.vacant(), 'Initial die gate refused'
time.sleep(5)
assert gate.vacant(), 'Second die gate refused'
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
name = 'vendor-hook-wk-kshift-v14-' + args.mode + '-m32'
command = ['bash', runtime + '/tests/glm52/whole_layer/vendor_hook_run.sh', '4', name,
    '--mode', args.mode, '--forward-widths', '32', '--forward-replays', '8',
    '--index-diagnostic',
    '--indexed-library', '/workspace/logs/attention-wk-kshift-v1/indexed/whole_layer_indexed_attention.so',
    '--shared-library', '/workspace/logs/attention-o-combined-v1/shared/whole_layer_full_attention.so']
(root / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
print('Starting real M32 outer + index diagnostic', args.mode, 'on a3-194 die4', flush=True)
with (root / (name + '.launcher.log')).open('w') as output:
    result = subprocess.run(gate.SSH + [shlex.join(command)], stdout=output, stderr=subprocess.STDOUT)
print('WK outer gate exit', result.returncode, flush=True)
raise SystemExit(result.returncode)
