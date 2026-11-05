"""Build the dual quantizer with the frozen parent's compiler commands."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import NAMESPACE, OLD_NAMESPACE, digest, prepare


BUILD_STEPS = {"kernel", "routed-tiling", "shared-tiling", "bridge-tiling",
               "binding", "link"}


def build(parent, output):
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel = next(item["argv"] for item in commands if item["step"] == "kernel")
    original = str(Path(kernel[kernel.index("-o") + 1]).parent)
    steps = set()
    for command in commands:
        step = command["step"]
        if step not in BUILD_STEPS:
            continue
        if step in steps:
            raise ValueError(f"Duplicate parent build step: {step}")
        steps.add(step)
        argv = [arg.replace(original, str(output)).replace(
            "/" + OLD_NAMESPACE + ".so", "/" + NAMESPACE + ".so")
            for arg in command["argv"]]
        if any("ROUTED_ENABLE_PROFILE" in arg for arg in argv):
            raise ValueError("Use the uninstrumented parent")
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"step": step, "argv": argv}) + "\n")
        with (output / (step + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    if steps != BUILD_STEPS:
        raise RuntimeError("The parent build command list is incomplete")
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed during compilation: {name}")
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / (OLD_NAMESPACE + ".so")))
    torch.ops.load_library(str(output / (NAMESPACE + ".so")))
    schemas = {}
    for name in ("prepare", "forward_out"):
        old = str(getattr(getattr(torch.ops, "tilert_" + OLD_NAMESPACE),
                          name).default._schema)
        new = str(getattr(getattr(torch.ops, "tilert_" + NAMESPACE),
                          name).default._schema)
        if new != old.replace(OLD_NAMESPACE, NAMESPACE):
            raise RuntimeError(f"The public schema changed: {name}")
        schemas[name] = new
    if torch.npu.is_initialized():
        raise RuntimeError("The CPU build initialized NPU")
    report.update({
        "namespace": "tilert_" + NAMESPACE,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "parent_co_load": True, "npu_initialized": False, "schemas": schemas,
        "artifacts": {path.name: digest(path) for path in output.iterdir()
                      if path.suffix in (".so", ".o")},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": True, "library_sha256":
                     report["artifacts"][NAMESPACE + ".so"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
