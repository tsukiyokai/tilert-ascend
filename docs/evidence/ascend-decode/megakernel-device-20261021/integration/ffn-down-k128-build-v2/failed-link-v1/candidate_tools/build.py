"""Build the K128 host plan and unchanged FFN consumer without a device."""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from prepare import NAMESPACE, OLD_NAMESPACE, PARENT_LIBRARY, digest, prepare


BUILD_STEPS = ("kernel", "bridge-tiling", "routed-tiling", "shared-tiling",
               "binding", "link")


def run_step(output, step, argv):
    with (output / "commands.jsonl").open("a") as log:
        log.write(json.dumps({"step": step, "argv": argv}) + "\n")
    with (output / (step + ".log")).open("w") as log:
        subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)


def inspect_tilings(parent, output, commands):
    command = next(item["argv"] for item in commands
                   if item["step"] == "routed-tiling")
    prefix = command[:command.index("-c")]
    original = str(Path(command[command.index("-o") + 1]).parent)
    linker = next(item["argv"] for item in commands if item["step"] == "link")
    libraries = []
    iterator = iter(linker)
    for argument in iterator:
        if argument == "-L":
            libraries.extend((argument, next(iterator)))
        elif argument in ("-ltiling_api", "-lplatform", "-lregister",
                          "-lunified_dlog", "-lc_sec"):
            libraries.append(argument)
    expected = json.loads(
        (output / "candidate_tools/expected-tilings.json").read_text())
    records = {}
    for label, tree in (("parent", parent), ("candidate", output)):
        binary = output / (label + "-tiling-probe")
        argv = [arg.replace(original, str(tree)) for arg in prefix]
        argv += [str(output / "candidate_tools/tiling_probe.cpp"),
                 str(tree / "routed-tiling.o")]
        if label == "candidate":
            argv.append(str(output / "static-down-plan.o"))
        argv += ["-o", str(binary)] + libraries
        run_step(output, label + "-tiling-probe-build", argv)
        result_path = output / (label + "-tilings.jsonl")
        run_step(output, label + "-tiling-probe-run",
                 [str(binary), str(result_path)])
        records[label] = [json.loads(line) for line in
                          result_path.read_text().splitlines()]
        if len(records[label]) != 4:
            raise RuntimeError(f"Expected four {label} tiling records")
        for entry in records[label]:
            payload = bytes.fromhex(entry["raw_hex"])
            entry["raw_sha256"] = hashlib.sha256(payload).hexdigest()
            if entry["status"] != 0 or len(payload) != 200:
                raise RuntimeError(f"Invalid {label} tiling result")
            if entry["projection"] == 1:
                wanted = expected[label][str(entry["layout"])]
                if (entry["raw_hex"] != wanted["raw_hex"] or
                        entry["raw_sha256"] != wanted["raw_sha256"]):
                    raise RuntimeError(f"The complete {label} Down tiling changed")
    for old, new in zip(records["parent"], records["candidate"], strict=True):
        if old["projection"] == 0 and new != old:
            raise RuntimeError("The complete GU tiling changed")
    (output / "compiled-tilings.json").write_text(
        json.dumps(records, indent=2) + "\n")
    return records


def inspect_plan_object(output):
    path = output / "static-down-plan.o"
    payload = path.read_bytes()
    if payload[:6] != b"\x7fELF\x02\x01":
        raise RuntimeError("The planner object must be little-endian ELF64")
    kind, machine = struct.unpack_from("<HH", payload, 16)
    if (kind, machine) != (1, 183):
        raise RuntimeError("The planner must be a relocatable AArch64 host object")
    for flag, label in (("-h", "header"), ("-sW", "symbols"), ("-SW", "sections")):
        run_step(output, "plan-object-" + label, ["readelf", flag, str(path)])
    return {"elf_class": 64, "elf_type": kind, "elf_machine": machine,
            "sha256": digest(path)}


def sdk_identity(compiler):
    root = Path(compiler).parent.parent / "aarch64-linux/asc"
    directories = (root / "impl/adv_api/tiling/matmul",)
    files = [path for directory in directories for path in directory.glob("*.h")]
    files.extend(root / name for name in (
        "include/adv_api/matmul/constant_tiling.h",
        "include/adv_api/matmul/tiling.h",
        "impl/adv_api/detail/matmul/feature_trait/matmul_feature_trait.h",
        "impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_base.h",
        "impl/adv_api/detail/matmul/resource/l0_buffer/tbuf_pool_l0_common.h",
        "impl/adv_api/detail/matmul/scheduler/base/scheduler_norm.h",
        "impl/adv_api/detail/matmul/scheduler/iterator/k_loop/k_loop_norm.h",
        "impl/adv_api/detail/matmul/param/matmul_shape_info.h"))
    return {str(path.relative_to(root)): digest(path) for path in sorted(files)}


def build(parent, output):
    if digest(parent / (OLD_NAMESPACE + ".so")) != PARENT_LIBRARY:
        raise ValueError("The frozen bbd parent library changed")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    kernel = next(item["argv"] for item in commands if item["step"] == "kernel")
    original = str(Path(kernel[kernel.index("-o") + 1]).parent)

    def relocated(argv):
        return [arg.replace(original, str(output)).replace(
            "/" + OLD_NAMESPACE + ".so", "/" + NAMESPACE + ".so") for arg in argv]

    plan_command = relocated(kernel)
    plan_command[plan_command.index("-c") + 1] = str(output / "source/static_down_plan.cpp")
    plan_command[plan_command.index("-o") + 1] = str(output / "static-down-plan.o")
    report["sdk_source_sha256"] = sdk_identity(plan_command[0])
    run_step(output, "static-down-plan", plan_command)
    report["plan_host_object"] = inspect_plan_object(output)
    for step in BUILD_STEPS:
        command, = [entry for entry in commands if entry["step"] == step]
        argv = relocated(command["argv"])
        if any("ROUTED_ENABLE_PROFILE" in arg for arg in argv):
            raise ValueError("Use the uninstrumented parent")
        if step == "link":
            argv.insert(argv.index("-o"), str(output / "static-down-plan.o"))
        run_step(output, step, argv)
    compiled_tilings = inspect_tilings(parent, output, commands)
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Source changed during compilation: {name}")

    import torch
    import torch_npu

    if torch.npu.is_initialized():
        raise RuntimeError("NPU was initialized before CPU co-load")
    torch.ops.load_library(str(parent / (OLD_NAMESPACE + ".so")))
    torch.ops.load_library(str(output / (NAMESPACE + ".so")))
    library = ctypes.CDLL(str(output / (NAMESPACE + ".so")))
    plans = {}
    for layout in ("nd", "nz"):
        symbol = NAMESPACE + "_plan_" + layout
        plans[layout] = list((ctypes.c_int32 * 6).in_dll(library, symbol))
        if plans[layout] != [128, 2, 6, 1, 1, 1]:
            raise RuntimeError("The linked constant API plan changed")
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
        "parent_co_load": True, "npu_initialized": False,
        "schemas": schemas, "compiled_tilings": compiled_tilings,
        "linked_planner_pods": plans,
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
