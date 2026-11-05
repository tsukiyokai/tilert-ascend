"""Compile the isolated token-split candidate using its parent's tool commands."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import PARENT_LIBRARY, digest, prepare


def build(parent, output):
    if digest(parent / "ffn_active_token_join.so") != PARENT_LIBRARY:
        raise ValueError("Frozen parent library changed")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel = next(value["argv"] for value in commands if value["step"] == "kernel")
    recorded_parent = str(Path(kernel[kernel.index("-o") + 1]).parent)
    for command in commands:
        argv = [value.replace(recorded_parent, str(output)).replace(
            "ffn_active_token_join.so", "ffn_token_split.so")
            for value in command["argv"]]
        if any("ROUTED_ENABLE_PROFILE" in value or "ROUTED_GU_PAD16" in value
               for value in argv):
            raise ValueError("Token split requires the frozen unpadded NoTrace parent")
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"step": command["step"], "argv": argv}) + "\n")
        with (output / (command["step"] + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed during compilation: {name}")
    version = subprocess.run([kernel[0], "--version"],
                             check=True, capture_output=True, text=True)
    (output / "compiler.txt").write_text(version.stdout + version.stderr)
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / "ffn_active_token_join.so"))
    torch.ops.load_library(str(output / "ffn_token_split.so"))
    schemas = {}
    for name in ("prepare", "forward_out"):
        before = str(getattr(torch.ops.tilert_ffn_active_token_join, name).default._schema)
        after = str(getattr(torch.ops.tilert_ffn_token_split, name).default._schema)
        if after != before.replace("ffn_active_token_join", "ffn_token_split"):
            raise RuntimeError(f"Public schema changed: {name}")
        schemas[name] = after
    if torch.npu.is_initialized():
        raise RuntimeError("CPU build initialized NPU")
    report.update({
        "namespace": "tilert_ffn_token_split",
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "parent_co_load": True, "npu_initialized": False, "schemas": schemas,
        "artifacts": {path.name: digest(path) for path in output.iterdir()
                      if path.suffix in (".o", ".so")},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"namespace": report["namespace"], "library_sha256":
                      report["artifacts"]["ffn_token_split.so"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
