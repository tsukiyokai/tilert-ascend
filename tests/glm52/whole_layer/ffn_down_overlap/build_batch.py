"""Compile the four-credit consumer child with verified parent tool commands."""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess

from prepare_batch import PARENT_LIBRARY, digest, prepare


def build(parent, output):
    if digest(parent / "ffn_down_overlap.so") != PARENT_LIBRARY:
        raise ValueError("Down-overlap parent library changed")
    report = prepare(parent, output)
    commands = [json.loads(line) for line in
                (parent / "commands.jsonl").read_text().splitlines()]
    original = next(value["argv"] for value in commands if value["step"] == "kernel")
    recorded_parent = str(Path(original[original.index("-o") + 1]).parent)
    sdk_root = Path(original[0]).parents[1] / "aarch64-linux/asc"
    audit = json.loads((output / "source/source-audit.json").read_text())
    report["sdk_headers"] = {}
    for entry in audit["sdk_sources"]:
        actual = digest(sdk_root / entry["path"])
        if actual != entry["sha256"]:
            raise RuntimeError(f"Audited SDK source changed: {entry['path']}")
        report["sdk_headers"][entry["path"]] = actual
    for command in commands:
        argv = [value.replace(recorded_parent, str(output)).replace(
            "ffn_down_overlap.so", "ffn_down_batch.so") for value in command["argv"]]
        if any("ROUTED_ENABLE_PROFILE" in value or "ROUTED_FIXPIPE_HALF" in value
               for value in argv):
            raise ValueError("Down batch requires the uninstrumented INT32 parent")
        with (output / "commands.jsonl").open("a") as log:
            log.write(json.dumps({"step": command["step"], "argv": argv}) + "\n")
        with (output / (command["step"] + ".log")).open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
    version = subprocess.run([original[0], "--version"],
                             check=True, capture_output=True, text=True)
    (output / "compiler.txt").write_text(version.stdout + version.stderr)
    import torch
    import torch_npu

    torch.ops.load_library(str(parent / "ffn_down_overlap.so"))
    torch.ops.load_library(str(output / "ffn_down_batch.so"))
    if torch.npu.is_initialized():
        raise RuntimeError("CPU compilation initialized NPU")
    schemas = {}
    for name in ("prepare", "forward_out"):
        expected = str(getattr(torch.ops.tilert_ffn_down_overlap, name).default._schema)
        actual = str(getattr(torch.ops.tilert_ffn_down_batch, name).default._schema)
        if actual != expected.replace("ffn_down_overlap", "ffn_down_batch"):
            raise RuntimeError(f"Down-batch schema changed: {name}")
        schemas[name] = actual
    native = ctypes.CDLL(str(output / "ffn_down_batch.so"))
    tiling = native.ffn_down_batch_body_build_routed_tiling
    tiling.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_int64]
    tiling.restype = ctypes.c_int
    tiling_checks = []
    for layout in (0, 1):
        for columns, inner in ((256, 6144), (6144, 128)):
            buffer = ctypes.create_string_buffer(1024)
            status = tiling(64, columns, inner, columns, buffer, layout)
            if status != 0:
                raise RuntimeError(f"Host tiling guard rejected {layout=}, {columns=}")
            tiling_checks.append({"layout": layout, "shape": [64, columns, inner],
                                  "status": status})
    for name, expected in report["sources"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen build source changed: {name}")
    report.update({
        "namespace": "tilert_ffn_down_batch", "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "npu_initialized": False,
        "parent_co_load": True, "schemas": schemas,
        "host_tiling_checks": tiling_checks,
        "artifacts": {path.name: digest(path) for pattern in ("*.o", "*.so")
                      for path in output.glob(pattern)},
        "device_validation": "not run",
    })
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"namespace": report["namespace"],
                      "library_sha256": report["artifacts"]["ffn_down_batch.so"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.parent.resolve(), args.output.resolve())
