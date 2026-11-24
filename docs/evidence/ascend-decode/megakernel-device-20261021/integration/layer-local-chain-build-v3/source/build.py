"""Compile the frozen local layer chain with the parent's CANN toolchain."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, label, command):
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"step": label, "argv": command}) + "\n")
    with (output / (label + ".log")).open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)


def build(output, cann, vendor):
    manifest = json.loads((output / "manifest.json").read_text())
    for relative, expected in manifest["sources"].items():
        if digest(output / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    provenance = json.loads((output / "attention-provenance.json").read_text())
    for relative, expected in provenance["vendor_headers"].items():
        if digest(vendor / relative) != expected:
            raise ValueError(f"The attention vendor source changed: {relative}")
    commands = [json.loads(line) for line in
                (output / "ffn-commands.jsonl").read_text().splitlines()]
    parent = next(item["argv"] for item in commands if item["step"] == "kernel")
    parent_root = str(Path(parent[parent.index("-o") + 1]).parent)
    command = [value.replace(parent_root, str(output / "ffn")).replace(
        "/opt/ascend/cann-9.2.0", str(cann)) for value in parent]
    command[command.index("-c") + 1] = str(output / "source/kernel.cpp")
    command[command.index("-o") + 1] = str(output / "kernel.o")
    command.extend(["-I", str(vendor)])
    definitions = [value for value in command if value.startswith("-D")]
    run(output, "host-contract-build", ["c++", "-std=c++17", "-O2", "-Wall",
        "-Wextra", "-Werror", *definitions,
        str(output / "source/contract_test.cpp"), "-o", str(output / "contract-test")])
    run(output, "host-contract", [str(output / "contract-test")])
    run(output, "kernel", command)
    library = output / "layer_local_chain.so"
    run(output, "link", ["c++", "-shared", "-Wl,--no-undefined",
        str(output / "kernel.o"), "-o", str(library),
        "-L", str(cann / "aarch64-linux/lib64"), "-L", str(cann / "lib64"),
        "-lascendcl", "-lruntime", "-lunified_dlog", "-lc_sec", "-lregister",
        "-lprofapi", "-ltiling_api", "-lplatform"])
    run(output, "exports", ["nm", "-D", "--defined-only", str(library)])
    symbols = (output / "exports.log").read_text()
    for symbol in ("layer_local_indexed_launch", "layer_local_shared_launch"):
        if not any(line.split()[-1] == symbol for line in symbols.splitlines()):
            raise RuntimeError(f"Missing C ABI export: {symbol}")
    for relative, expected in manifest["sources"].items():
        if digest(output / relative) != expected:
            raise RuntimeError(f"Source changed during compilation: {relative}")
    report = {
        "parent_manifest_sha256": digest(output / "manifest.json"),
        "library_sha256": digest(library),
        "kernel_sha256": digest(output / "kernel.o"),
        "host_contract": json.loads((output / "host-contract.log").read_text()),
        "compiler_fp_contract": "off, matching the complete FFN parent",
        "device_execution": False,
        "artifacts": {path.name: digest(path) for path in output.iterdir()
                      if path.is_file() and path.suffix in (".o", ".so", ".log", ".jsonl")},
    }
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cann", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.cann.resolve(), args.vendor.resolve())
