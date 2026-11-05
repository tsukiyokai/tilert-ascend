"""Record the original context-manager function without initializing an NPU."""

import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
import torch_npu
from vllm.forward_context import set_forward_context


def identify(function):
    path = Path(inspect.getfile(function)).resolve()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "module": function.__module__,
        "name": function.__qualname__,
    }


result = {
    "scope": "CPU import and contextmanager source identity only",
    "python": sys.version,
    "torch": torch.__version__,
    "torch_npu": torch_npu.__version__,
    "wrapper": identify(set_forward_context),
    "original": identify(inspect.unwrap(set_forward_context)),
    "npu_initialized": torch.npu.is_initialized(),
    "image": "sha256:9afbed90a0f92e228cc1ba13adf3cee2e89e34f1dd4cad1ed8eb5ce8c091b4af",
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if result["npu_initialized"]:
    raise RuntimeError("The source identity check initialized an NPU")
