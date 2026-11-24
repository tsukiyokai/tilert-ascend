"""Compare explicit overload retention with the same CPU-only probe."""

import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path('/tmp/tilert-graph-observer-traceability-20261028.py')
OUTPUT = Path('/tmp/tilert-graph-observer-traceability-default-20261028.json')
spec = importlib.util.spec_from_file_location('retention_probe', BASE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def explicit_default(x, path):
    return probe.torch.ops.my_ns.acl_save.default(x, path)


probe.acl_save = explicit_default
probe.torch.set_num_threads(1)
result = {
    'scope': 'Explicit .default CPU Dynamo/AOT retention; NPU/SK untested',
    'torch': probe.torch.__version__,
    'source_sha256': {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), BASE, probe.SOURCE, probe.CPP)
    },
    'cases': [probe.run(aot, hook) for aot in (False, True)
              for hook in (False, True)],
}
result['passed'] = all(case['passed'] for case in result['cases'])
OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'output': str(OUTPUT), 'passed': result['passed']}))
