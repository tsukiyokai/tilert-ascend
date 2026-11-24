"""Check the pinned FLA wheel's real import without NPU initialization."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

import torch
import torch_npu


def reject_device_initialization(*args, **kwargs):
    raise RuntimeError("This dependency import probe does not initialize an NPU")


torch.npu._lazy_init = reject_device_initialization
torch_npu._C._npu_init = reject_device_initialization
report = {
    "python": sys.version, "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "npu_initialized_before": torch.npu.is_initialized(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
try:
    import fla_npu
    from fla_npu.ops import ascendc

    report["version"] = importlib.metadata.version("flash-linear-attention-npu-a3")
    report["files"] = {
        module.__name__: {
            "path": module.__file__,
            "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        }
        for module in (fla_npu, ascendc)
    }
    report["required_exports"] = {
        name: callable(getattr(ascendc, name))
        for name in ("causal_conv1d_fn", "causal_conv1d_update",
                     "recurrent_gated_delta_rule")
    }
    report["success"] = all(report["required_exports"].values())
except Exception:
    report["success"] = False
    report["error"] = traceback.format_exc()
report["npu_initialized_after"] = torch.npu.is_initialized()
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
raise SystemExit(0 if report["success"] else 1)
