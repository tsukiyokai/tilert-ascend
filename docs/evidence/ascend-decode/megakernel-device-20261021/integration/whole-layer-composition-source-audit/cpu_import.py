"""Read the official decoder import boundary without initializing an NPU."""

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


os.environ["VLLM_PLUGINS"] = "ascend"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch
import torch_npu


def refuse_npu_init(*args, **kwargs):
    raise RuntimeError("The decoder CPU import audit refuses NPU initialization")


torch.npu._lazy_init = refuse_npu_init
torch_npu._C._npu_init = refuse_npu_init
report = {
    "scope": "Actual official module imports; no constructor or device execution",
    "image": os.environ["AUDIT_IMAGE"],
    "vllm_commit": "a98247ab4db686ee03c66d5feb3c761e52a2f8ab",
    "ascend_commit": "8237ee6d690d1399bc6eb48ef5977ac2bd470f3f",
    "script_sha256": sha256(__file__),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_path": torch.__file__,
    "torch_npu": torch_npu.__version__,
    "torch_npu_path": torch_npu.__file__,
    "initialized_before": torch.npu.is_initialized(),
    "pythonpath": sys.path,
    "packages": {},
    "source_compile": [],
    "imports": [],
}
for name in ("transformers", "triton", "triton-ascend", "numpy", "vllm", "vllm-ascend"):
    try:
        report["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report["packages"][name] = None

vllm = Path("/workspace/vendor-hook-pinned/vllm-a982")
for relative in (
    "vllm/model_executor/model_loader/sharded_state_loader.py",
    "vllm/model_executor/models/deepseek_v2.py",
):
    path = vllm / relative
    item = {"path": str(path), "sha256": sha256(path)}
    try:
        compile(path.read_bytes(), str(path), "exec")
        item["success"] = True
    except Exception:
        item["success"] = False
        item["error"] = traceback.format_exc()
    report["source_compile"].append(item)

for name in (
    "vllm.config",
    "vllm.forward_context",
    "vllm_ascend.platform",
    "vllm_ascend.attention.sfa_v1",
    "vllm.model_executor.models.deepseek_v2",
    "vllm_ascend.ops.fused_moe.gate_linear",
    "vllm.model_executor.layers.fused_moe.runner.moe_runner",
    "vllm_ascend.compilation.acl_graph",
):
    item = {"module": name}
    try:
        module = importlib.import_module(name)
        item.update(path=module.__file__, sha256=sha256(module.__file__), success=True)
    except Exception:
        item.update(success=False, error=traceback.format_exc())
    report["imports"].append(item)
    print(json.dumps({"module": name, "success": item["success"]}), flush=True)

report["initialized_after"] = torch.npu.is_initialized()
report["success"] = all(item["success"] for item in report["imports"])
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
raise SystemExit(not report["success"] or report["initialized_after"])
