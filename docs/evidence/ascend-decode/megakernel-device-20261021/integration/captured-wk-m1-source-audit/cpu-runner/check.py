import json
import runpy
import sys
import torch
import torch_npu

assert not torch.npu.is_initialized()
sys.path.insert(0, "/probe")
sys.argv = ["dispatch.py", "--capture", "/capture/decoder-graph-index-diagnostic.pt", "--capture-report", "/capture/decoder-graph-index-diagnostic.json", "--output", "/evidence", "--cpu-only"]
runpy.run_path("/probe/dispatch.py", run_name="__main__")
sys.argv = ["test_contract.py", "--capture-report", "/capture/decoder-graph-index-diagnostic.json"]
runpy.run_path("/probe/test_contract.py", run_name="__main__")
assert not torch.npu.is_initialized()
print(json.dumps({"torch": torch.__version__, "torch_npu": torch_npu.__version__, "npu_initialized": torch.npu.is_initialized(), "cpu_only": True}))
