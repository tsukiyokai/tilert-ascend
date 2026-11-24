"""Load the actual FFN extension without initializing an NPU."""

import hashlib
import json
from pathlib import Path
import sys

import torch
import torch_npu

from decoder_ffn import LIBRARY_SHA


library = Path(__file__).with_name("ffn_active_token_join.so")
if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA:
    raise RuntimeError("The complete-FFN library changed")
torch.ops.load_library(str(library))
schemas = {name: str(getattr(torch.ops.tilert_ffn_active_token_join, name).default._schema)
           for name in ("prepare", "forward_out")}
if torch.npu.is_initialized():
    raise RuntimeError("CPU preflight initialized an NPU")
Path(sys.argv[1]).write_text(json.dumps({
    "pass": True, "npu_initialized": False, "library_sha256": LIBRARY_SHA,
    "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    "schemas": schemas,
}, indent=2) + "\n")
