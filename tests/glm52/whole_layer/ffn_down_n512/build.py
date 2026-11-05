"""Compile and inspect Down N512 without initializing an NPU."""

import argparse
import json
from pathlib import Path
import subprocess

from prepare import NAMESPACE, OLD_NAMESPACE, PARENT_LIBRARY, digest, prepare


def run_step(output, step, argv):
    with (output / "commands.jsonl").open("a") as log:
        log.write(json.dumps({"step": step, "argv": argv}) + "\n")
    with (output / (step + ".log")).open("w") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)


def inspect_tilings(parent, output, commands):
    original = next(item["argv"] for item in commands
                    if item["step"] == "routed-tiling")
    prefix = original[:original.index("-c")]
    root = str(Path(original[original.index("-o") + 1]).parent)
    linker = next(item["argv"] for item in commands if item["step"] == "link")
    libraries = []
    iterator = iter(linker)
    for argument in iterator:
        if argument == "-L":
            libraries.extend((argument, next(iterator)))
        elif argument in ("-ltiling_api", "-lplatform", "-lregister",
                          "-lunified_dlog", "-lc_sec"):
            libraries.append(argument)
    records = []
    for label, tree in (("parent", parent), ("candidate", output)):
        binary = output / (label + "-tiling-probe")
        includes = [arg.replace(root, str(tree)) for arg in prefix]
        argv = includes + [str(output / "candidate_tools/tiling_probe.cpp"),
                         str(tree / "routed-tiling.o"), "-o", str(binary)] + libraries
        run_step(output, label + "-tiling-probe-build", argv)
        result = subprocess.check_output([str(binary)], text=True)
        (output / (label + "-tilings.jsonl")).write_text(result)
        records.append([json.loads(line) for line in result.splitlines()])
    for old, new in zip(*records, strict=True):
        if old["projection"] == 0:
            if new != old:
                raise RuntimeError("The GU tiling changed")
        elif (new["baseM"], new["baseN"], new["baseK"]) != (64, 512, 128):
            raise RuntimeError("The compiled Down tiler rejected N512")
    return records[1]


def build(parent, output):
    if digest(parent / (OLD_NAMESPACE + ".so")) != PARENT_LIBRARY:
        raise ValueError("The frozen parent library changed")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel = next(item["argv"] for item in commands if item["step"] == "kernel")
    original = str(Path(kernel[kernel.index("-o") + 1]).parent)
    for command in commands:
        argv = [arg.replace(original, str(output)).replace(
            "/" + OLD_NAMESPACE + ".so", "/" + NAMESPACE + ".so")
            for arg in command["argv"]]
        if any("ROUTED_ENABLE_PROFILE" in arg for arg in argv):
            raise ValueError("Use the uninstrumented parent")
        run_step(output, command["step"], argv)
    compiled_tilings = inspect_tilings(parent, output, commands)
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
        "compiled_tilings": compiled_tilings,
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
