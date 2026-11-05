"""Build the combined candidate with the active parent's frozen toolchain."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import PARENT_LIBRARY, digest, prepare


def build(parent, output):
    if digest(parent / "ffn_active.so") != PARENT_LIBRARY:
        raise ValueError("The active parent library changed")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel = next(c for c in commands if c["step"] == "kernel")["argv"]
    original = str(Path(kernel[kernel.index("-o") + 1]).parent)
    for command in commands:
        argv = [arg.replace(original, str(output)).replace(
            "/ffn_active.so", "/ffn_active_descriptor.so")
            for arg in command["argv"]]
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"step": command["step"], "argv": argv}) + "\n")
        with (output / (command["step"] + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed during compilation: {name}")

    import torch
    import torch_npu

    torch.ops.load_library(str(parent / "ffn_active.so"))
    torch.ops.load_library(str(output / "ffn_active_descriptor.so"))
    if torch.npu.is_initialized():
        raise RuntimeError("CPU build initialized NPU")
    schemas = {}
    for name in ("prepare", "forward_out"):
        expected = str(getattr(torch.ops.tilert_ffn_active, name).default._schema)
        actual = str(getattr(
            torch.ops.tilert_ffn_active_descriptor, name).default._schema)
        if actual != expected.replace("ffn_active", "ffn_active_descriptor"):
            raise RuntimeError(f"Combined candidate schema changed: {name}")
        schemas[name] = actual
    report.update({
        "namespace": "tilert_ffn_active_descriptor",
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "npu_initialized": False, "parent_co_load": True,
        "schemas": schemas,
        "artifacts": {p.name: digest(p) for pattern in ("*.o", "*.so")
                      for p in output.glob(pattern)},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "library_sha256":
                      report["artifacts"]["ffn_active_descriptor.so"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
