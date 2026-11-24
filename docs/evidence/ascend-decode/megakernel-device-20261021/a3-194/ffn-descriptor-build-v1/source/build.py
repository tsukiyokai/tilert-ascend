"""Compile the descriptor candidate with the frozen parent's tool commands."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import PARENT_LIBRARY, digest, prepare


def build(parent, output):
    if digest(parent / "ffn_composed.so") != PARENT_LIBRARY:
        raise ValueError("Parent library differs from the validated build")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel_command = next(value for value in commands if value["step"] == "kernel")
    argv = kernel_command["argv"]
    recorded_parent = str(Path(argv[argv.index("-o") + 1]).parent)
    for command in commands:
        argv = [value.replace(recorded_parent, str(output)).replace(
            "ffn_composed.so", "ffn_descriptor.so") for value in command["argv"]]
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"step": command["step"], "argv": argv}) + "\n")
        with (output / (command["step"] + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    version = subprocess.run([kernel_command["argv"][0], "--version"],
                             check=True, capture_output=True, text=True)
    (output / "compiler.txt").write_text(version.stdout + version.stderr)
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / "ffn_composed.so"))
    torch.ops.load_library(str(output / "ffn_descriptor.so"))
    if torch.npu.is_initialized():
        raise RuntimeError("CPU compilation initialized NPU")
    schemas = {}
    for name in ("prepare", "forward_out"):
        original = str(getattr(torch.ops.tilert_ffn_composed, name).default._schema)
        actual = str(getattr(torch.ops.tilert_ffn_descriptor, name).default._schema)
        if actual != original.replace("ffn_composed", "ffn_descriptor"):
            raise RuntimeError(f"Descriptor schema changed: {name}")
        schemas[name] = actual
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen build source changed: {name}")
    report.update({
        "namespace": "tilert_ffn_descriptor", "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "npu_initialized": False,
        "parent_co_load": True, "schemas": schemas,
        "artifacts": {p.name: digest(p) for pattern in ("*.o", "*.so")
                      for p in output.glob(pattern)},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"namespace": report["namespace"],
                      "library_sha256": report["artifacts"]["ffn_descriptor.so"],
                      "statement_sha256": report["body_statement_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
