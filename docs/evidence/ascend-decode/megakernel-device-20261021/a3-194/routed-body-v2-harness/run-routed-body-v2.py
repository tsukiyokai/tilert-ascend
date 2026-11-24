import importlib.util
from pathlib import Path
import subprocess
import time

path = Path(__file__).with_name('run-pending-single-die.py')
spec = importlib.util.spec_from_file_location('idle_gate', path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
gate.SSH[-1] = 'a3-194'
expected = '6dcb8401cf722f9b3c3650d2432321ce6100d79aa98ff329d3a17a63a0f1d6c4'
actual = subprocess.check_output(gate.SSH + ['sha256sum', '/home/shanshan/tilert-megakernel-20261021/tests/glm52/whole_layer/routed_body/build-v2/routed_int4_torch.so'], text=True).split()[0]
assert actual == expected
assert gate.vacant()
time.sleep(5)
assert gate.vacant()
users = subprocess.run(gate.SSH + ['fuser', '/dev/davinci4'], capture_output=True, text=True)
assert users.returncode == 1 and not users.stdout.strip() and not users.stderr.strip(), users
subprocess.run(gate.SSH + ['docker', 'start', gate.CONTAINER], check=True)
base = 'python3 tests/glm52/routed_int4/check_pipeline_front.py --device npu:0 --library tests/glm52/whole_layer/routed_body/build-v2/routed_int4_torch.so --baseline-library tests/glm52/routed_int4/build-pipeline-front/routed_int4_torch.so --native-namespace tilert_pipeline_body --baseline-namespace tilert_pipeline_front --baseline-input bf16 --widths 1 2 12 32 --replays 8 --warmup 1 --no-timing'
gate.CHECKS = tuple((name, base + options) for name, options in [
    ('routed-body-v2-mixed-b24', ' --blocks 24 --route-pattern mixed --shared-scale-dtype float32'),
    ('routed-body-v2-extremes-b24', ' --blocks 24 --route-pattern extremes --shared-scale-dtype bfloat16'),
    ('routed-body-v2-extremes-b16', ' --blocks 16 --route-pattern extremes --shared-scale-dtype bfloat16'),
])
gate.main()
