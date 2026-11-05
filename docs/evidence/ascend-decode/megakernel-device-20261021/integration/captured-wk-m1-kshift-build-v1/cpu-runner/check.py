import ctypes
import json
import runpy
import sys
import unittest
import torch
import torch_npu

assert not torch.npu.is_initialized()
sys.path.insert(0, "/probe")
library = ctypes.CDLL("/probe/build/wk_kshift.so")
assert getattr(library, "wk_m1_kshift_launch")
sys.argv = ["kshift_check.py", "--capture", "/capture/decoder-graph-index-diagnostic.pt", "--capture-report", "/capture/decoder-graph-index-diagnostic.json", "--output", "/evidence", "--cpu-only"]
runpy.run_path("/probe/kshift_check.py", run_name="__main__")
suite = unittest.defaultTestLoader.discover("/probe", pattern="test_kshift.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
assert not torch.npu.is_initialized()
print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__, "npu_initialized": torch.npu.is_initialized(), "cpu_only": True, "cabi_symbol_loaded": "wk_m1_kshift_launch"}))
