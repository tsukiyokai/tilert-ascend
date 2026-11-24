"""Import the official source pair without initializing the NPU runtime."""

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

os.environ["VLLM_PLUGINS"] = "ascend"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
import torch_npu


def refuse_npu_init(*args, **kwargs):
    raise RuntimeError("CPU import audit refuses NPU initialization")


torch.npu._lazy_init = refuse_npu_init
torch_npu._C._npu_init = refuse_npu_init
report = {
    "vllm_commit": "a98247ab4db686ee03c66d5feb3c761e52a2f8ab",
    "ascend_commit": "8237ee6d690d1399bc6eb48ef5977ac2bd470f3f",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "initialized_before": torch.npu.is_initialized(),
    "pythonpath": sys.path,
    "platform_plugins": [
        {"name": entry.name, "value": entry.value}
        for entry in importlib.metadata.entry_points(group="vllm.platform_plugins")
    ],
    "imports": [],
}
for name in (
    "vllm.config", "vllm.forward_context", "vllm_ascend.platform",
    "vllm_ascend.attention.sfa_v1", "vllm_ascend.ops.mla",
    "vllm_ascend.compilation.acl_graph",
    "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic",
):
    item = {"module": name}
    try:
        module = importlib.import_module(name)
        item["path"] = module.__file__
        item["sha256"] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        item["success"] = True
    except Exception:
        item["success"] = False
        item["error"] = traceback.format_exc()
    report["imports"].append(item)
    print(json.dumps(item), flush=True)

try:
    from vllm.platforms import current_platform
    report["platform_type"] = (
        f"{type(current_platform).__module__}.{type(current_platform).__name__}"
    )
except Exception:
    report["platform_error"] = traceback.format_exc()
report["initialized_after"] = torch.npu.is_initialized()
report["success"] = all(item["success"] for item in report["imports"])
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(not report["success"] or report["initialized_after"])
