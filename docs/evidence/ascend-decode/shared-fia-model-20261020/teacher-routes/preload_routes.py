"""Preload Ascend operator registration before the frozen route collector."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch
import torch_npu
import vllm_ascend.ops
from vllm_ascend.device.device_op import DeviceOperator

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "collect.py"
EXPECTED = "b3be9f705e85121f4547ef9c526c16b94088cc2ce75b5ffc34643be52085a567"
assert hashlib.sha256(TARGET.read_bytes()).hexdigest() == EXPECTED
assert not torch.npu.is_initialized()
assert callable(DeviceOperator.moe_gating_top_k)
source = Path(sys.modules[DeviceOperator.__module__].__file__)
binding = {"collector_sha256": EXPECTED,
           "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "device_operator_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
           "preload": "import vllm_ascend.ops", "npu_initialized_after_preload": False}
(ROOT / "routes-prelude.json").write_text(json.dumps(binding, indent=2) + "\n")
sys.argv = [str(TARGET), *sys.argv[1:]]
runpy.run_path(str(TARGET), run_name="__main__")
