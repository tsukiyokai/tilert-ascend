import hashlib
import json
from pathlib import Path
import sys
import unittest
import torch
import torch_npu
sys.path.insert(0, '/source')
import vendor_hook_index_diagnostic_test
suite = unittest.defaultTestLoader.loadTestsFromModule(vendor_hook_index_diagnostic_test)
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {'scope': 'CPU diagnostic contracts; no device nodes or NPU operations',
          'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
          'npu_initialized': torch.npu.is_initialized(),
          'tests': result.testsRun, 'pass': result.wasSuccessful(),
          'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in Path('/source').glob('*.py')}}
Path('/evidence/report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
raise SystemExit(not result.wasSuccessful())
