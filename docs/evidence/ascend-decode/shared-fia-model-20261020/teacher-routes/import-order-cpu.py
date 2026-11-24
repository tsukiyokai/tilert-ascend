"""Check real operator import order without initializing NPU."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import torch
import torch_npu

before = torch_npu.npu._initialized
import vllm_ascend.ops
from vllm_ascend.device.device_op import DeviceOperator

after = torch_npu.npu._initialized
assert before is False and after is False
assert callable(DeviceOperator.moe_gating_top_k)
result = {"status": "pass", "preload": "import vllm_ascend.ops",
          "npu_initialized_before": before, "npu_initialized_after": after,
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(Path(__file__).parent / "import-order-cpu.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
