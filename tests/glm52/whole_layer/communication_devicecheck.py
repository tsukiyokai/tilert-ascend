#!/usr/bin/env python3
"""Two dependent TP reductions and a CUBE projection in one MIX launch."""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--endpoint", required=True,
                        help="SHMEM tcp://IP:port, separate from torchrun")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 12, 32])
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--no-timing", action="store_true",
                        help="run numerical and graph replay checks only")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class NativeProbe:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path.resolve()))
        signatures = {
            "initialize": [ctypes.c_int32, ctypes.c_int32,
                           ctypes.c_char_p, ctypes.c_uint64],
            "prepare": [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
                        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
            "execute": [ctypes.c_void_p] * 6,
            "destroy": [ctypes.c_void_p],
            "finalize": [],
        }
        for name, signature in signatures.items():
            function = getattr(self.library, "communication_probe_" + name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        self.library.communication_probe_last_error.restype = ctypes.c_char_p

    def call(self, name, *values):
        function = getattr(self.library, "communication_probe_" + name)
        if function(*values) != 0:
            error = self.library.communication_probe_last_error().decode()
            raise RuntimeError(error)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_input(torch, rows, rank, iteration, random_values):
    if random_values:
        generator = torch.Generator().manual_seed(19073 + 97 * rank + iteration)
        return (torch.randn(rows, 6144, generator=generator) / 8).bfloat16()
    values = torch.arange(rows * 6144).reshape(rows, 6144)
    return ((values + 7 * rank + 3 * iteration) % 31 - 15).bfloat16()


def make_weight(torch, rank):
    # Nontrivial exact dyadic coefficients exercise CUBE dependence while
    # keeping the tagged fixture insensitive to legal FP32 dot-product order.
    identity = torch.eye(16)
    return (identity * ((rank % 3 + 1) / 4)
            + identity.roll(shifts=1, dims=1) / 8).bfloat16()


def ordered_sum(torch, values):
    result = values[0].float().clone()
    for value in values[1:]:
        result.add_(value.float())
    return result.bfloat16()


def oracle(torch, rows, ranks, iteration, random_values):
    first = ordered_sum(torch, [
        make_input(torch, rows, rank, iteration, random_values)
        for rank in range(ranks)
    ])
    partials = [
        (first.float().reshape(-1, 16) @ make_weight(torch, rank).float())
        .reshape(rows, 6144).bfloat16()
        for rank in range(ranks)
    ]
    return first, ordered_sum(torch, partials)


def difference(torch, actual, expected):
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    actual_bytes = actual.contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.contiguous().reshape(-1).view(torch.uint8)
    delta = (actual.float() - expected.float()).abs()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "bitwise_equal": bool(torch.equal(actual_bytes, expected_bytes)),
        "bitwise_different_bytes": int((actual_bytes != expected_bytes).sum()),
        "mismatch_count": int((actual != expected).sum()),
        "max_absolute": float(delta.max()),
        "mean_absolute": float(delta.mean()),
    }


def check_tensor(torch_npu, value, shape):
    if (str(value.dtype) != "torch.bfloat16" or list(value.shape) != shape
            or not value.is_contiguous()
            or torch_npu.get_npu_format(value) != 2):
        raise ValueError(f"expected BF16 physical ND tensor with shape {shape}")


def identity(torch, torch_npu, arguments, rank, ranks):
    library = arguments.library.resolve()
    source_files = sorted(Path(__file__).parent.glob("communication*"))
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if "shmem" in path and path.startswith("/") and Path(path).is_file():
            mapped.add(path)
    versions = {}
    query_version = getattr(torch_npu.utils, "get_cann_version", None)
    if query_version is not None:
        for component in ("CANN", "RUNTIME", "COMPILER", "OPP", "OPP_KERNEL", "DRIVER"):
            try:
                versions[component] = query_version(module=component)
            except (RuntimeError, ValueError, TypeError, UnicodeError) as error:
                versions[component] = {"query_error": str(error)}
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "one MIX launch with two TP reductions and dependent CUBE projection",
        "rank": rank,
        "ranks": ranks,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": str(torch.npu.get_device_properties(torch.npu.current_device())),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cann": versions,
        "library": {"path": str(library), "sha256": digest(library)},
        "loaded_shmem": [{"path": path, "sha256": digest(Path(path))}
                         for path in sorted(mapped)],
        "source_sha256": {path.name: digest(path) for path in source_files
                          if path.is_file()},
        "build_identity": (library.parent / "identity.txt").read_text(),
        "warmup": arguments.warmup,
        "changed_input_replays": arguments.replays,
        "timed_iterations": 0 if arguments.no_timing else arguments.iterations,
        "blocks": arguments.blocks,
        "oracle": "rank-ascending FP32 sum, BF16; FP32 16x16 dot, BF16; same sum",
        "hccl_comparison": "installed torch.distributed HCCL BF16 SUM; no equality assumed",
    }


def check_shape(torch, torch_npu, dist, native, args, rows, rank, ranks):
    owner = ctypes.c_void_p()
    native.call("prepare", rows, rank, ranks, args.blocks, ctypes.byref(owner))
    device = torch.device("npu", torch.npu.current_device())
    partial = torch_npu.npu_format_cast(
        torch.empty((rows, 6144), dtype=torch.bfloat16, device=device), 2)
    weight = torch_npu.npu_format_cast(make_weight(torch, rank).to(device), 2)
    first = torch_npu.npu_format_cast(torch.empty_like(partial), 2)
    output = torch_npu.npu_format_cast(torch.empty_like(partial), 2)
    for value, shape in [(partial, [rows, 6144]), (weight, [16, 16]),
                         (first, [rows, 6144]), (output, [rows, 6144])]:
        check_tensor(torch_npu, value, shape)

    def execute():
        stream = torch.npu.current_stream().npu_stream
        native.call("execute", owner, stream, partial.data_ptr(),
                    weight.data_ptr(), first.data_ptr(), output.data_ptr())

    record = {"rows": rows, "checks": []}
    try:
        partial.copy_(make_input(torch, rows, rank, 0, False))
        torch.npu.synchronize()
        for _ in range(args.warmup):
            execute()
        torch.npu.synchronize()
        dist.barrier()
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            execute()
        torch.npu.synchronize()
        for iteration in range(args.replays):
            random_values = iteration % 2 == 1
            partial.copy_(make_input(torch, rows, rank, iteration, random_values))
            torch.npu.synchronize()
            # Alternate the delayed rank across replays. Device epochs remain
            # persistent; neither the host nor graph capture resets them.
            if rank == iteration % ranks:
                time.sleep(0.01)
            graph.replay()
            torch.npu.synchronize()
            expected_first, expected_output = oracle(
                torch, rows, ranks, iteration, random_values)
            first_difference = difference(torch, first, expected_first)
            output_difference = difference(torch, output, expected_output)
            hccl_first = partial.clone()
            dist.all_reduce(hccl_first)
            hccl_output = (hccl_first.float().reshape(-1, 16) @ weight.float())
            hccl_output = hccl_output.reshape(rows, 6144).bfloat16()
            dist.all_reduce(hccl_output)
            torch.npu.synchronize()
            record["checks"].append({
                "replay": iteration,
                "fixture": "random" if random_values else "tagged",
                "first_vs_ordered_oracle": first_difference,
                "output_vs_ordered_oracle": output_difference,
                "first_vs_hccl": difference(torch, first, hccl_first),
                "output_vs_hccl": difference(torch, output, hccl_output),
            })
        dist.barrier()
        elapsed = []
        for _ in range(0 if args.no_timing else args.iterations):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end) * 1000)
        record["timing_us"] = elapsed
        record["median_us"] = statistics.median(elapsed) if elapsed else None
        record["timing_scope"] = (
            "disabled" if args.no_timing
            else "single captured MIX launch; CUBE work included"
        )
        record["passed"] = all(
            check[name]["finite"] and check[name]["mismatch_count"] == 0
            for check in record["checks"]
            for name in ("first_vs_ordered_oracle", "output_vs_ordered_oracle")
        )
        # A live graph owns the pointer arguments even when it is not running.
        del graph
        torch.npu.synchronize()
        dist.barrier()
    finally:
        native.call("destroy", owner)
    return record


def main():
    args = arguments()
    if (any(rows < 1 or rows > 32 for rows in args.rows)
            or args.blocks < 1 or min(args.warmup, args.replays, args.iterations) < 1):
        raise ValueError("rows must be 1..32; blocks and sample counts must be positive")
    import torch
    import torch.distributed as dist
    import torch_npu

    rank = int(os.environ["RANK"])
    ranks = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    native = NativeProbe(args.library)
    native.call("initialize", rank, ranks, args.endpoint.encode(), 64 * 1024 * 1024)
    result = identity(torch, torch_npu, args, rank, ranks)
    result["shapes"] = []
    try:
        for rows in args.rows:
            result["shapes"].append(check_shape(
                torch, torch_npu, dist, native, args, rows, rank, ranks))
    finally:
        torch.npu.synchronize()
        dist.barrier()
        native.call("finalize")
        dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"communication-rank-{rank}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
    if not all(shape["passed"] for shape in result["shapes"]):
        raise RuntimeError(f"communication oracle mismatch; see {path}")
    print(path)


if __name__ == "__main__":
    main()
