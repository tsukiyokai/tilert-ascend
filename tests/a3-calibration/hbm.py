"""Compile or measure one A3 die's sustained GM-read bandwidth.

The primary numerator is the input size, read once per kernel launch. Every
element contributes to a checksum; the 32-byte output per AIV is excluded.
Four disjoint inputs rotate between launches. NPU execution requires --run.
"""

import argparse
import configparser
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time


MIB = 1024 * 1024
FLOAT_BYTES = 4
WARMUP = 100
SAMPLES = 1000
VALIDATE_EVERY = 100
PATTERN_PERIOD = 17
PATTERN_CENTER = 8
FLOAT_EXACT_INTEGER = 2**24


@dataclass(frozen=True)
class ReadShape:
    workers: int
    tiles_per_worker: int
    tile_elements: int
    buffers: int

    @property
    def elements(self) -> int:
        return self.workers * self.tiles_per_worker * self.tile_elements

    @property
    def read_bytes(self) -> int:
        return self.elements * FLOAT_BYTES

    @property
    def write_bytes(self) -> int:
        return self.workers * 32


def main() -> None:
    args = parse_arguments()
    platform_info = read_platform(args.platform_config)
    shape = make_shape(args, platform_info)
    expected, absolute_sums = expected_checksums(shape)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = make_metadata(args, platform_info, shape)
    write_json(args.output / "manifest.json", metadata)
    write_json(args.output / "expected-checksums.json", {
        "pattern": "(global_tile_id % 17) - 8 + buffer_slot",
        "expected_by_slot_and_worker": expected,
        "absolute_sums_by_slot_and_worker": absolute_sums,
        "exact_integer_limit": FLOAT_EXACT_INTEGER,
    })
    kernel = compile_kernel(args.output, shape)
    if args.run:
        run_measurement(args, shape, expected, kernel)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buffer-mib", type=int, default=512)
    parser.add_argument("--buffers", type=int, default=4)
    parser.add_argument("--tile-elements", type=int, default=8192)
    parser.add_argument("--workers", type=int,
                        help="Default: vector_core_cnt from the platform INI.")
    parser.add_argument("--device", type=int,
                        default=int(os.environ.get("LOCAL_RANK", "0")))
    parser.add_argument("--source-commit",
                        help="Source commit when running a copy without .git.")
    parser.add_argument(
        "--context-json", type=Path,
        help="Optional clock, power, temperature and load record.",
    )
    cann_home = Path(os.environ.get("ASCEND_HOME_PATH",
                                    "/opt/ascend/cann-9.2.0"))
    parser.add_argument(
        "--platform-config", type=Path,
        default=(cann_home / "aarch64-linux/data/platform_config"
                 / "Ascend910_9392.ini"),
    )
    return parser.parse_args()


def read_platform(path: Path) -> dict:
    parser = configparser.ConfigParser()
    with path.open(encoding="utf-8") as source:
        parser.read_file(source)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "soc": parser.get("version", "SoC_version"),
        "aiv_count": parser.getint("SoCInfo", "vector_core_cnt"),
        "aic_count": parser.getint("SoCInfo", "cube_core_cnt"),
        "l2_bytes": parser.getint("SoCInfo", "l2_size"),
        "ub_bytes": parser.getint("AICoreSpec", "ub_size"),
        "memory_bytes": parser.getint("SoCInfo", "memory_size"),
        "source_kind": "installed CANN platform INI; no device query",
    }


def make_shape(args: argparse.Namespace, info: dict) -> ReadShape:
    workers = info["aiv_count"] if args.workers is None else args.workers
    tile = args.tile_elements
    if workers <= 0 or workers % 2 or workers > info["aiv_count"]:
        raise ValueError(
            "workers must be even and within the platform AIV count"
        )
    if tile < 64 or tile & (tile - 1):
        raise ValueError("tile-elements must be a power of two, at least 64")
    if 3 * tile * FLOAT_BYTES + 4096 > info["ub_bytes"]:
        raise ValueError(
            "two input tiles plus accumulator exceed the UB budget"
        )
    if args.buffer_mib <= 0 or args.buffers < 2:
        raise ValueError(
            "buffer-mib must be positive; buffers must be at least 2"
        )
    group_bytes = workers * tile * FLOAT_BYTES * 2
    pairs = math.ceil(args.buffer_mib * MIB / group_bytes)
    shape = ReadShape(workers, pairs * 2, tile, args.buffers)
    if shape.read_bytes < 2 * info["l2_bytes"]:
        raise ValueError("each input buffer must be at least 2x the L2 size")
    if shape.read_bytes * shape.buffers < 8 * info["l2_bytes"]:
        raise ValueError("the input ring must be at least 8x the L2 size")
    return shape


def expected_checksums(shape: ReadShape) -> tuple[list, list]:
    expected = []
    absolute_sums = []
    for slot in range(shape.buffers):
        markers = [i - PATTERN_CENTER + slot for i in range(PATTERN_PERIOD)]
        slot_expected = []
        slot_absolute = []
        for worker in range(shape.workers):
            start = worker * shape.tiles_per_worker
            cycles, tail = divmod(shape.tiles_per_worker, PATTERN_PERIOD)
            remaining = [markers[(start + i) % PATTERN_PERIOD]
                         for i in range(tail)]
            total = cycles * sum(markers) + sum(remaining)
            absolute = (cycles * sum(map(abs, markers))
                        + sum(map(abs, remaining)))
            absolute *= shape.tile_elements
            if absolute >= FLOAT_EXACT_INTEGER:
                raise ValueError(
                    "checksum absolute sum must be <2**24; reduce buffer size "
                    "or slot count, or increase workers"
                )
            slot_expected.append(total * shape.tile_elements)
            slot_absolute.append(absolute)
        expected.append(slot_expected)
        absolute_sums.append(slot_absolute)
    return expected, absolute_sums


def make_metadata(
    args: argparse.Namespace, info: dict, shape: ReadShape
) -> dict:
    root = Path(__file__).resolve().parents[2]
    git_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    commit = args.source_commit
    if commit is None and git_result.returncode == 0:
        commit = git_result.stdout.strip()
    environment_keys = (
        "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES", "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "ACL_OP_INIT_MODE", "TILELANG_CACHE_DIR",
        "RANK", "LOCAL_RANK", "WORLD_SIZE",
    )
    context = None
    if args.context_json is not None:
        context = json.loads(args.context_json.read_text(encoding="utf-8"))
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "repository_commit": commit,
        "repository_commit_source": ("--source-commit" if args.source_commit
                                     else "git" if commit else "unavailable"),
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("hbm.py", "_hbm_tilelang.py")
        },
        "mode": "run" if args.run else "compile-only",
        "device": args.device,
        "rank": int(os.environ.get("RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "environment": {key: os.environ.get(key) for key in environment_keys},
        "platform": info,
        "shape": asdict(shape),
        "launch_block_groups": shape.workers // 2,
        "aiv_workers_per_group": 2,
        "requested_buffer_bytes": args.buffer_mib * MIB,
        "read_bytes_per_launch": shape.read_bytes,
        "checksum_write_bytes_per_launch": shape.write_bytes,
        "input_ring_bytes": shape.read_bytes * shape.buffers,
        "warmup": WARMUP,
        "samples": SAMPLES,
        "validate_every": VALIDATE_EVERY,
        "primary_metric": "read_bytes / NPU_event_seconds, decimal GB/s",
        "timing_scope": "one launch, event record to event record",
        "wall_scope": "start event record through end event synchronize",
        "outlier_policy": "retain all samples",
        "graph_capture": False,
        "profiler": False,
        "clock_policy": "unchanged by this program",
        "environmental_context": context,
    }


def compile_kernel(output: Path, shape: ReadShape):
    import torch
    import torch_npu

    before = torch_npu.npu.is_initialized()
    if before:
        raise RuntimeError("compilation requires an uninitialized NPU")
    import tilelang
    from _hbm_tilelang import compile_read_kernel

    kernel = compile_read_kernel(shape.workers, shape.tiles_per_worker,
                                 shape.tile_elements)
    after = torch_npu.npu.is_initialized()
    if after:
        raise RuntimeError("NPU initialization occurred during compilation")
    (output / "kernel.cpp").write_text(kernel.get_kernel_source(),
                                       encoding="utf-8")
    (output / "wrapped.cpp").write_text(kernel.adapter.wrapped_source,
                                        encoding="utf-8")
    (output / "kernel.tir").write_text(kernel.prim_func.script(),
                                       encoding="utf-8")
    library = output / "streaming-read.so"
    # Cache loads populate the CDLL handle but not LibraryGenerator.libpath.
    library_source = Path(kernel.adapter.lib._name)
    shutil.copy2(library_source, library)
    report = {
        "tilelang_version": tilelang.__version__,
        "tilelang_source": tilelang.__file__,
        "torch_version": torch.__version__,
        "torch_npu_version": importlib.metadata.version("torch-npu"),
        "target": "ascendc",
        "platform": "A3",
        "execution_backend": "cython",
        "npu_initialized_before": before,
        "npu_initialized_after": after,
        "kernel_launched": False,
        "tensor_allocated": False,
        "library": str(library.resolve()),
        "library_source": str(library_source),
        "library_sha256": file_sha256(library),
        "library_bytes": library.stat().st_size,
    }
    write_json(output / "compile-result.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return kernel


def run_measurement(args, shape: ReadShape, expected: list, kernel) -> None:
    import torch

    torch.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    inputs = make_inputs(shape, device)
    checksum = torch.empty((shape.workers, 8), dtype=torch.float32,
                           device=device)
    validations = []
    for slot, data in enumerate(inputs):
        kernel(data, checksum)
        torch.npu.synchronize()
        validations.append(validate_checksum(checksum, expected[slot], slot,
                                             "initial", slot))
    write_json(args.output / "validation.json", validations)
    for iteration in range(WARMUP):
        torch.npu.synchronize()
        kernel(inputs[iteration % shape.buffers], checksum)
        torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    samples = []
    raw_path = args.output / "raw.csv"
    with raw_path.open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=(
            "sample", "slot", "read_bytes", "checksum_write_bytes",
            "event_ms", "wall_ms", "read_gb_s", "wall_read_gb_s",
        ))
        writer.writeheader()
        for sample in range(SAMPLES):
            slot = (WARMUP + sample) % shape.buffers
            torch.npu.synchronize()
            wall_start = time.perf_counter_ns()
            start.record()
            kernel(inputs[slot], checksum)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
            event_ms = start.elapsed_time(end)
            if event_ms <= 0:
                raise RuntimeError(
                    f"nonpositive event duration at sample {sample}"
                )
            row = {
                "sample": sample,
                "slot": slot,
                "read_bytes": shape.read_bytes,
                "checksum_write_bytes": shape.write_bytes,
                "event_ms": event_ms,
                "wall_ms": wall_ms,
                "read_gb_s": shape.read_bytes / event_ms / 1e6,
                "wall_read_gb_s": shape.read_bytes / wall_ms / 1e6,
            }
            writer.writerow(row)
            out.flush()
            samples.append(row)
            if sample % VALIDATE_EVERY == 0 or sample == SAMPLES - 1:
                validations.append(validate_checksum(checksum, expected[slot],
                                                     slot, "sample", sample))
                write_json(args.output / "validation.json", validations)
                print(f"sample={sample} event_ms={event_ms:.6f} "
                      f"read_GB_s={row['read_gb_s']:.3f}", flush=True)
    metrics = ("event_ms", "wall_ms", "read_gb_s", "wall_read_gb_s")
    summary = {
        key: summarize([row[key] for row in samples]) for key in metrics
    }
    summary.update(samples=len(samples), checksum_validations=len(validations),
                   kernel_launched=True, all_samples_retained=True)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def make_inputs(shape: ReadShape, device) -> list:
    import torch

    tile_count = shape.workers * shape.tiles_per_worker
    tile_ids = torch.arange(tile_count, dtype=torch.int32, device=device)
    base = (tile_ids % PATTERN_PERIOD - PATTERN_CENTER).to(torch.float32)
    inputs = []
    for slot in range(shape.buffers):
        data = torch.empty((shape.workers, shape.tiles_per_worker,
                            shape.tile_elements), dtype=torch.float32,
                           device=device)
        markers = (base + slot)[:, None]
        data.view(tile_count, shape.tile_elements).copy_(markers)
        inputs.append(data)
    torch.npu.synchronize()
    return inputs


def validate_checksum(checksum, expected, slot: int, phase: str, index: int):
    observed = checksum.cpu().tolist()
    for worker, (row, reference) in enumerate(zip(observed, expected)):
        if row[0] != reference:
            raise RuntimeError(
                f"checksum mismatch: phase={phase}, index={index}, "
                f"slot={slot}, "
                f"worker={worker}, expected={reference}, observed={row}"
            )
    return {"phase": phase, "index": index, "slot": slot,
            "checksum_by_worker": [row[0] for row in observed], "passed": True}


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"min": min(values), "max": max(values),
            "mean": statistics.fmean(values), "std": statistics.pstdev(values),
            "p50": percentile(0.50), "p90": percentile(0.90),
            "p95": percentile(0.95), "p99": percentile(0.99)}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


if __name__ == "__main__":
    main()
