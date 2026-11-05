"""Inventory baseline prerequisites without importing Torch or reading weights."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys


PACKAGES = ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers",
            "safetensors", "triton-ascend", "triton", "numpy", "memfabric-hybrid")
PREFIX = "offline/prefix8k-allocation-20261017/capture/case-00"


def stat_path(path):
    result = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        result["bytes"] = path.stat().st_size
    return result


def metadata(paths):
    result = {}
    normalized = {name.replace("_", "-").lower() for name in PACKAGES}
    for distribution in importlib.metadata.distributions(path=paths):
        name = distribution.metadata.get("Name", "").replace("_", "-").lower()
        if name in normalized:
            result.setdefault(name, []).append({
                "version": distribution.version,
                "directory": str(distribution.locate_file("")),
            })
    return result


def prefix_inventory(roots):
    rows = []
    for root in roots:
        case = root / PREFIX
        manifest = case / "manifest.json"
        record = stat_path(manifest)
        if manifest.is_file():
            if manifest.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Unexpectedly large prefix metadata file")
            raw = manifest.read_bytes()
            value = json.loads(raw)
            runtime = value.get("runtime", {})
            record.update({
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": value.get("status"), "seq_len": value.get("seq_len"),
                "world_size": runtime.get("world_size"),
                "layout_version": runtime.get("layout", {}).get("version"),
                "checkpoint_sha256": runtime.get("checkpoint_sha256"),
                "payload_stats": [stat_path(case / kind / f"rank-{rank}.pt")
                                  for kind in ("main-prefix", "mtp-prefix")
                                  for rank in range(16)],
                "payload_hashes_read": False,
            })
        rows.append(record)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--project-root", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    result = {
        "hostname": platform.node(), "machine": platform.machine(),
        "python": sys.version, "executable": sys.executable,
        "cpu_affinity": [cpu], "scope": "Package metadata and bounded project paths",
        "base_packages": metadata(sys.path),
        "tools": {name: shutil.which(name) for name in
                  ("python3", "torchrun", "g++", "bash", "npu-smi")},
        "prefixes": prefix_inventory(args.project_root),
        "project_paths": [stat_path(root / name) for root in args.project_root
                          for name in ("environment/activate.sh", "models/GLM-5.2-w4a8c8")],
        "weight_payload_read": False, "model_started": False,
        "mapped_davinci_devices": sorted(str(p) for p in Path("/dev").glob("davinci*")),
    }
    if args.runtime:
        paths = [args.runtime / "vendor-hook-deps/transformers519",
                 args.runtime / "vendor-hook-deps/python",
                 args.runtime / "vendor-hook-pinned/vllm-a982-py311-compatible",
                 args.runtime / "vendor-hook-pinned/vendor-metadata"]
        result["overlay_paths"] = [stat_path(path) for path in paths]
        result["overlay_packages"] = metadata([str(path) for path in paths])
        names = (
            "vendor-hook-pinned/vllm-a982-py311-compatible/vllm/engine/arg_utils.py",
            "vendor-hook-pinned/vllm-a982-py311-compatible/vllm/model_executor/model_loader/default_loader.py",
            "vendor-hook-pinned/ascend-python-build/vllm_ascend/worker/worker.py",
            "tilert/models/glm_5_2_ascend/cann_model.py",
            "tests/glm52/bench_cann_mtp.py",
        )
        result["runtime_paths"] = [stat_path(args.runtime / name) for name in names]
        pin_file = args.runtime / "frozen-inputs.sha256"
        if pin_file.is_file():
            result["runtime_pin_manifest"] = {
                **stat_path(pin_file),
                "sha256": hashlib.sha256(pin_file.read_bytes()).hexdigest(),
                "entries": len(pin_file.read_text().splitlines()),
                "listed_payloads_rehashed": False,
            }
    if args.decoder:
        result["decoder_paths"] = [stat_path(args.decoder / name) for name in (
            "python/vllm_ascend/worker/worker.py",
            "python/vllm_ascend/_C_ascend.so",
            "ffn_host_adapter.py", "device_full_graph.py")]
    result["accelerator_modules_imported"] = sorted(
        name for name in sys.modules if name == "torch" or name.startswith("torch_npu"))
    if result["accelerator_modules_imported"]:
        raise RuntimeError("The metadata inventory must not import the accelerator runtime")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
