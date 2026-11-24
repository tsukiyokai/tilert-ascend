"""Freeze a two-device eager fixture above the immutable local-chain parents."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import textwrap


PARENT_SHA = "2ebf8e0b80a708de354376ab7f182f5853986e879c767773b4016b9ebc901543"
TP_SHA = "c2b72e469c3ee80a5fe7d9df4a3cceedc0e65cca95e6f426aea762c213392462"
TP_HOST_SHA = "c3caf00eaf135c963c5ef5e990ae309531e9802aa03574edbbdc0fd310534493"
SHMEM = {
    "libshmem.so": "cf9b853bb7612ce1e44853b1d16851016c4a2c279bdc9bb32da0640bc411f532",
    "libshmem_utils.so": "743ea9661dbc6707fba5030967a22c59ebfdd0daed4884fdff39da12ae008ac0",
    "aclshmem_bootstrap_config_store.so": "ef9160f3706a7eac573ff0a83160ae2d38579550dd923f45b330f309a33d6541",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one source anchor: {old!r}")
    return source.replace(old, new)


def function_source(source, name, owner=None):
    tree = ast.parse(source)
    nodes = tree.body
    if owner:
        nodes = next(node.body for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    return textwrap.dedent("".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno]))


def phase_source(source):
    original = function_source(source, "separate", "LayerCase")
    candidate = once(original, "def separate(self):",
                     "def separate(self, attention_sum, output, reduce):")
    candidate = once(candidate, "    self.ffn.native.forward_out(\n        self.partial,",
                     "    reduce(self.partial, attention_sum, 'ar1')\n"
                     "    self.ffn.native.forward_out(\n        attention_sum,")
    candidate = once(candidate, "    self.parent_validated = True",
                     "    reduce(self.bridge[10], output, 'ar2')\n"
                     "    self.parent_validated = True")
    return original, candidate


def constructor_source(source):
    names = ("tensor_descriptor", "make_configuration", "run_rank_zero")
    functions = {name: function_source(source, name) for name in names}
    value = functions["run_rank_zero"]
    value = once(value, "def run_rank_zero(", "def run_device_rank(")
    value = once(value, "torch.npu.set_device(0)", "torch.npu.set_device(args.device_rank)")
    value = once(value, '"logical_device": 0', '"logical_device": args.device_rank')
    value = once(value, "torch.npu.get_device_name(0)", "torch.npu.get_device_name(args.device_rank)")
    value = once(value, "torch.npu.device_count() != 1", "torch.npu.device_count() != 2")
    value = once(value, 'raise RuntimeError("Expose exactly one NPU")',
                 'raise RuntimeError("Expose exactly two NPUs")\n'
                 '    stage("physical_device_mapping", lambda: device_mapping(torch, args))')
    value = once(value, "with rank_zero_mc2(torch, state, stage, command_sent) as mc2:",
                 "with device_groups_context(torch, state, stage, args.device_rank, args) as mc2:")
    value = once(value, "initialize_parameters(torch, layer), {\"seed\": 41021}",
                 "initialize_rank_parameters(torch, layer, args.device_rank, args.oracle_cpu), "
                 "{\"seed\": 41021, \"rank_shard\": args.device_rank}")
    value = once(value, "from decoder_storage_local_chain import run_decoder_forward",
                 "from decoder_storage_tp_chain import run_decoder_forward")
    value = once(value, 'stage("actual_local_component_chain",', 'stage("actual_two_rank_component_chain",')
    value = once(value, 'args.output / "local-chain.json"', 'args.output / "tp-chain.json"')
    value = once(value, '"local_component_chain_executed": True,', '"two_rank_component_chain_executed": True,')
    value = once(value,
                 '"scope": "Actual weights and metadata; separate native components versus one local layer launch; both TP reductions identity",',
                 '"scope": "Two TP16-shaped shards; separate native components and HCCL2 versus in-kernel SHMEM2",')
    value = once(value,
                 '"mc2_scope": "actual HCCL rank0 singleton used by the original W4A8 constructor"',
                 '"mc2_scope": "actual per-device singleton used only by the original W4A8 constructor"')
    header = '''"""Original constructor and postprocessing with explicit two-device setup."""

import ctypes
import hashlib
import importlib
import json
from pathlib import Path

from tp_device import device_mapping
from tp_groups import device_groups_context
from tp_parameters import initialize as initialize_rank_parameters

'''
    return header + functions["tensor_descriptor"] + "\n\n" + functions["make_configuration"] + "\n\n" + value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "tp-library", "tp-host", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deploy-name", default="vendor-hook-tp-chain-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9._-]+", args.deploy_name):
        raise ValueError("Use a simple deployment directory name")
    for path, expected in ((args.parent / "manifest.json", PARENT_SHA),
                           (args.tp_library, TP_SHA), (args.tp_host, TP_HOST_SHA)):
        if digest(path) != expected:
            raise RuntimeError(f"A required parent changed: {path}")
    hashes = {}
    for line in (args.parent / "frozen-inputs.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if digest(args.parent / name) != expected:
            raise RuntimeError(f"A parent source changed: {name}")
        hashes[name] = expected
    args.output.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        if name == "manifest.json":
            continue
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.parent / name, target)
    directory = Path(__file__).resolve().parent
    for source, target in (("chain.py", "tp_chain.py"), ("groups.py", "tp_groups.py"),
                           ("parameters.py", "tp_parameters.py"), ("device.py", "tp_device.py"),
                           ("stage.py", "stage_tp.py"), ("test_contract.py", "test_tp_contract.py"),
                           ("cpu_check.py", "tp_cpu_check.py")):
        shutil.copy2(directory / source, args.output / target)
    shutil.copy2(directory / "parameter_geometry.json", args.output / "parameter_geometry.json")
    shutil.copy2(args.tp_library, args.output / "layer_tp_chain.so")
    shutil.copy2(args.tp_host, args.output / "tp_host.py")
    (args.output / "shmem-libraries.json").write_text(json.dumps(SHMEM, indent=2) + "\n")
    original, phases = phase_source((args.parent / "local_chain.py").read_text())
    (args.output / "tp_parent.py").write_text("from local_chain import BLOCKS\n\n\n" + phases + "\n")
    (args.output / "parent-phases.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), phases.splitlines(keepends=True),
        fromfile="LayerCase.separate", tofile="tp_parent.separate")))
    constructor = constructor_source((args.parent / "device_local_chain.py").read_text())
    (args.output / "tp_constructor.py").write_text(constructor)
    parent_constructor = (args.parent / "device_local_chain.py").read_text()
    (args.output / "constructor.patch").write_text("".join(difflib.unified_diff(
        parent_constructor.splitlines(keepends=True), constructor.splitlines(keepends=True),
        fromfile="device_local_chain.py", tofile="tp_constructor.py")))
    storage = once((args.parent / "decoder_storage_local_chain.py").read_text(),
                   "from local_chain import run_local_chain as original_forward",
                   "from tp_chain import run_tp_chain as original_forward")
    storage = once(storage, "before the local component chain.", "before the two-rank local component chain.")
    storage = once(storage, "before the eager local component chain", "before the two-rank eager component chain")
    (args.output / "decoder_storage_tp_chain.py").write_text(storage)
    launch = (args.parent / "device_local_chain_run.sh").read_text()
    launch = once(launch, "if [[ $# -lt 2 ]]; then", "if [[ $# -lt 3 ]]; then")
    launch = launch.replace("PHYSICAL_DEVICE REPORT_NAME", "PHYSICAL_FIRST PHYSICAL_SECOND REPORT_NAME")
    launch = once(launch, "physical_device=$1\nreport_name=$2\nshift 2", "physical_first=$1\nphysical_second=$2\nreport_name=$3\nshift 3")
    launch = once(launch, "! $physical_device =~ ^[0-9]+$", "! $physical_first =~ ^[0-9]+$ || ! $physical_second =~ ^[0-9]+$")
    launch = launch.replace("vendor-hook-local-chain-v3", args.deploy_name)
    launch = once(launch, '--device "/dev/davinci$physical_device"',
                  '--device "/dev/davinci$physical_first" --device "/dev/davinci$physical_second"')
    launch = once(launch, '-v "$runtime_root:/workspace:ro"',
                  '-v "$task_root/communication-deps/shmem-build/lib:/shmem:ro" \\\n    -v "$runtime_root:/workspace:ro"')
    launch = once(launch, "        PYTHONPATH=", '        LD_LIBRARY_PATH="/shmem:${LD_LIBRARY_PATH:-}" \\\n        PYTHONPATH=')
    launch = once(launch, "python3 /decoder/device_local_chain.py", "python3 /decoder/tp_device.py")
    launch = once(launch, '--output "/evidence/$1" --mode int8 --timeout 600',
                  '--output "/evidence/$1" --mode int8 --timeout 900 \\\n            --physical-devices "$2" "$3"')
    launch = once(launch, '"${@:2}"', '"${@:4}"')
    launch = once(launch, "' decoder212 \"$report_name\" \"$@\"",
                  "' decoder212 \"$report_name\" \"$physical_first\" \"$physical_second\" \"$@\"")
    (args.output / "device_tp_run.sh").write_text(launch)
    parent = json.loads((args.parent / "manifest.json").read_text())
    files = {str(path.relative_to(args.output)): digest(path)
             for path in args.output.rglob("*") if path.is_file()}
    report = {key: value for key, value in parent.items()
              if key.endswith("_sha256") and key != "parent_manifest_sha256"}
    report.update(parent_manifest_sha256=PARENT_SHA, tp_library_sha256=TP_SHA,
                  tp_host_sha256=TP_HOST_SHA, deployment_directory=args.deploy_name,
                  scope="Eager two-rank component chain; TP16 constructor geometry only",
                  graph_capture=False, device_execution=False, shmem_libraries=SHMEM,
                  files_sha256=files)
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    files["manifest.json"] = digest(manifest)
    (args.output / "frozen-inputs.sha256").write_text("".join(
        f"{sha}  {name}\n" for name, sha in sorted(files.items())))
    print(json.dumps({"directory": str(args.output), "files": len(files),
                      "manifest_sha256": digest(manifest)}, indent=2))


if __name__ == "__main__":
    main()
