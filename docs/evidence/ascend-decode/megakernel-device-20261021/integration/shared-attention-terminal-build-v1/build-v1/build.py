"""Compile the isolated shared-attention terminal completion under Torch NPU 2.12."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare import LAYER, PARENT_LIBRARY, digest, prepare


def build(parent, output, parent_library):
    if digest(parent_library) != PARENT_LIBRARY:
        raise ValueError("Unexpected runtime212 control library")
    report = prepare(parent, output)
    source = output / "source" / LAYER
    environment = dict(os.environ, ATTENTION_FULL="1")
    command = ["bash", str(source / "attention_build.sh"), str(output)]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "build.log").open("w") as log:
        subprocess.run(command, env=environment, check=True,
                       stdout=log, stderr=subprocess.STDOUT)
    original_name = output / "whole_layer_full_attention.so"
    library = output / "shared_attention_terminal.so"
    original_name.rename(library)
    import torch
    import torch_npu

    torch.ops.load_library(str(parent_library))
    torch.ops.load_library(str(library))
    schemas = {}
    for name in ("shared_full_attention_prepare", "shared_full_attention_out"):
        control = str(getattr(torch.ops.tilert, name).default._schema)
        candidate = str(getattr(torch.ops.tilert_attention_terminal, name).default._schema)
        if candidate.replace("tilert_attention_terminal::", "tilert::") != control:
            raise ValueError("Candidate schema changed")
        schemas[name] = candidate
    if torch.npu.is_initialized():
        raise RuntimeError("CPU build initialized the NPU")
    report.update({
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "npu_initialized": False, "parent_co_load": True, "schemas": schemas,
        "artifacts": {path.name: digest(path) for path in sorted(output.iterdir())
                      if path.suffix in (".o", ".so")},
    })
    for name in ("prepare.py", "build.py", "test_prepare.py"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    report["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("artifacts", "npu_initialized", "parent_co_load")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("parent_library", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve(), args.parent_library.resolve())
