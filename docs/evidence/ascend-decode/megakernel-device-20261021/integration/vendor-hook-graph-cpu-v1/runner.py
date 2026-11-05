import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest

import torch
import torch_npu

sys.path.insert(0, "/source")
suite = unittest.defaultTestLoader.discover("/source", "vendor_hook_graph_test.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
report = {
    "scope": "CPU persistent-producer contracts only; no graph or NPU execution",
    "image": "sha256:9afbed90a0f92e228cc1ba13adf3cee2e89e34f1dd4cad1ed8eb5ce8c091b4af",
    "python": sys.version, "torch": torch.__version__,
    "torch_npu": torch_npu.__version__, "platform": platform.platform(),
    "tests": result.testsRun, "failures": len(result.failures),
    "errors": len(result.errors), "npu_initialized": torch.npu.is_initialized(),
    "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("/source").glob("*.py"))},
    "pass": result.wasSuccessful() and not torch.npu.is_initialized(),
}
Path("/evidence/report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
raise SystemExit(0 if report["pass"] else 1)
