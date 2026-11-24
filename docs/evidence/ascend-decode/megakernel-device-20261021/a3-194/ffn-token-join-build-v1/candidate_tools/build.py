"""Compile the tail-join candidate using the frozen token-output tool commands."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import PARENT_LIBRARY, digest, prepare


def build(parent, output):
    if digest(parent / "ffn_token_output.so") != PARENT_LIBRARY:
        raise ValueError("Parent library differs from the validated token-output build")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel = next(value for value in commands if value["step"] == "kernel")
    original = kernel["argv"]
    recorded_parent = str(Path(original[original.index("-o") + 1]).parent)
    for command in commands:
        argv = [value.replace(recorded_parent, str(output)).replace(
            "ffn_token_output.so", "ffn_token_join.so") for value in command["argv"]]
        if any("ROUTED_ENABLE_PROFILE" in value for value in argv):
            raise ValueError("Tail join requires the uninstrumented parent command")
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"step": command["step"], "argv": argv}) + "\n")
        with (output / (command["step"] + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    version = subprocess.run([original[0], "--version"],
                             check=True, capture_output=True, text=True)
    (output / "compiler.txt").write_text(version.stdout + version.stderr)
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / "ffn_token_output.so"))
    torch.ops.load_library(str(output / "ffn_token_join.so"))
    if torch.npu.is_initialized():
        raise RuntimeError("CPU compilation initialized NPU")
    schemas = {}
    for name in ("prepare", "forward_out"):
        expected = str(getattr(torch.ops.tilert_ffn_token_output, name).default._schema)
        actual = str(getattr(torch.ops.tilert_ffn_token_join, name).default._schema)
        if actual != expected.replace("ffn_token_output", "ffn_token_join"):
            raise RuntimeError(f"Tail-join schema changed: {name}")
        schemas[name] = actual
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen build source changed: {name}")
    report.update({
        "namespace": "tilert_ffn_token_join", "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "npu_initialized": False,
        "parent_co_load": True, "schemas": schemas,
        "artifacts": {path.name: digest(path) for pattern in ("*.o", "*.so")
                      for path in output.glob(pattern)},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"namespace": report["namespace"],
                      "library_sha256": report["artifacts"]["ffn_token_join.so"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
