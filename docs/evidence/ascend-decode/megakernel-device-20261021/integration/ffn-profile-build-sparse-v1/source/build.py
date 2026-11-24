"""Build a phase-profile variant with the accepted parent's compiler commands."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import digest, prepare


def build(parent, output, sparse=False):
    report = prepare(parent, output, sparse)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    original = commands[0]["argv"][-1].removesuffix("/kernel.o")
    if not original.endswith("/ffn_composed/build-v2"):
        raise ValueError("Unexpected parent build prefix")
    command_path = output / "commands.jsonl"
    for command in commands:
        arguments = [arg.replace(original, str(output)) for arg in command["argv"]]
        arguments = [arg.replace("/ffn_composed.so", "/ffn_profile.so")
                     for arg in arguments]
        with command_path.open("a") as stream:
            stream.write(json.dumps({"step": command["step"], "argv": arguments}) + "\n")
        with (output / (command["step"] + ".log")).open("w") as log:
            subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed during build: {name}")

    import torch
    import torch_npu

    for path in (parent / "ffn_composed.so", output / "ffn_profile.so"):
        torch.ops.load_library(str(path))
    if torch.npu.is_initialized():
        raise RuntimeError("CPU profile build initialized NPU")
    report.update({
        "namespace": "tilert_ffn_profile", "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "npu_initialized": False,
        "scratch_tensors": {"bridge_and_trace": 13, "body": 20},
        "artifacts": {p.name: digest(p) for pattern in ("*.o", "*.so")
                      for p in output.glob(pattern)},
        "schemas": {name: str(getattr(torch.ops.tilert_ffn_profile, name).default._schema)
                    for name in ("prepare", "forward_out")},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"library_sha256": report["artifacts"]["ffn_profile.so"],
                      "npu_initialized": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve(), args.sparse)
