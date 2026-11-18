"""Single-arm numerical gate for the real two-reduction MIX probe; no timing."""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_library(prepared, arm, frozen_driver=None):
    build = json.loads((prepared / "build-report.json").read_text())
    manifest = json.loads((prepared / "manifest.json").read_text())
    if digest(prepared / "manifest.json") != build["prepared_manifest_sha256"]:
        raise ValueError("Prepared source manifest differs from the build")
    for name, expected in manifest["sources"].items():
        if digest(prepared / name) != expected:
            raise ValueError(f"Frozen compiled source changed: {name}")
    library = Path(build["arms"][arm]["library"]["path"])
    if digest(library) != build["arms"][arm]["library"]["sha256"]:
        raise ValueError("Built private library identity changed")
    runtime = digest(Path(__file__))
    frozen = manifest["experiment_sources"]["device_check.py"]
    if runtime != frozen:
        if frozen_driver is None or digest(frozen_driver) != frozen:
            raise ValueError("Changed driver requires --frozen-driver matching the old manifest")
    elif frozen_driver is not None and digest(frozen_driver) != frozen:
        raise ValueError("Explicit frozen driver differs from the build manifest")
    return library, {
        "library_sha256": digest(library),
        "manifest_sha256": digest(prepared / "manifest.json"),
        "build_report_sha256": digest(prepared / "build-report.json"),
        "compiled_manifest_driver_sha256": frozen,
        "runtime_driver_sha256": runtime, "driver_changed": runtime != frozen,
        "validated_frozen_driver": None if frozen_driver is None else {
            "path": str(frozen_driver), "sha256": digest(frozen_driver)},
    }


class NativeProbe:
    def __init__(self, path, arm):
        self.prefix = f"trt_rsag_{arm}_probe_"
        self.library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        signatures = {
            "initialize": [ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p,
                           ctypes.c_uint64],
            "prepare": [ctypes.c_int64, ctypes.c_int32, ctypes.c_int32,
                        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
            "execute": [ctypes.c_void_p] * 6,
            "destroy": [ctypes.c_void_p], "finalize": [],
        }
        for name, signature in signatures.items():
            function = getattr(self.library, self.prefix + name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        self.error = getattr(self.library, self.prefix + "last_error")
        self.error.argtypes = []
        self.error.restype = ctypes.c_char_p

    def call(self, name, *values):
        if getattr(self.library, self.prefix + name)(*values) != 0:
            raise RuntimeError(self.error().decode())


def source(torch, rows, rank, iteration):
    if iteration % 2:
        generator = torch.Generator().manual_seed(19073 + 97 * rank + iteration)
        return (torch.randn(rows, 6144, generator=generator) / 8).bfloat16()
    values = torch.arange(rows * 6144).reshape(rows, 6144)
    return ((values + 7 * rank + 3 * iteration) % 31 - 15).bfloat16()


def weight(torch, rank):
    identity = torch.eye(16)
    return (identity * ((rank % 3 + 1) / 4)
            + identity.roll(shifts=1, dims=1) / 8).bfloat16()


def tree_sum(values):
    values = [value.float() for value in values]
    while len(values) > 1:
        split = 1
        while split * 2 < len(values):
            split *= 2
        values = [values[index] + values[index + split]
                  if index + split < len(values) else values[index]
                  for index in range(split)]
    return values[0].bfloat16()


def oracle(torch, rows, ranks, iteration):
    first = tree_sum([source(torch, rows, rank, iteration)
                      for rank in range(ranks)])
    partials = [(first.float().reshape(-1, 16) @ weight(torch, rank).float())
                .reshape(rows, 6144).bfloat16() for rank in range(ranks)]
    return first, tree_sum(partials)


def hccl_columns(ranks, family):
    if family == "witness":
        if ranks < 4:
            raise ValueError("The four-rank association witness requires >=4 ranks")
        columns = [("four_rank_order", [0x3F80, 0x3300, 0xBF80, 0x0000])]
    elif family == "finite":
        columns = [("exact_dyadic", [0x3F80, 0x3F00, 0xBE80, 0x3E00]),
                   ("large_finite_cancellation", [0x6F00, 0xEF00, 0x3F80, 0xBF00]),
                   ("small_normal", [0x0080, 0x0080, 0x8080, 0x0080])]
    elif family == "ties":
        columns = [("even_bf16_tie", [0x3F80, 0x3B80]),
                   ("odd_bf16_tie", [0x3F81, 0x3B80]),
                   ("negative_even_tie", [0xBF80, 0xBB80]),
                   ("negative_odd_tie", [0xBF81, 0xBB80])]
    elif family == "edges":
        columns = [("negative_zero", [0x8000] * ranks),
                   ("mixed_zero", [0x8000, 0]),
                   ("subnormal", [0x0001, 0x0001]),
                   ("subnormal_to_normal", [0x007F, 0x0001]),
                   ("finite_inputs_fp32_overflow", [0x7F7F] * ranks),
                   ("four_rank_nan_tree", [0x7F7F, 0xFF7F, 0x7F7F, 0xFF7F])]
    else:
        raise ValueError(f"Unknown HCCL fixture family: {family}")
    return [(name, (values + [0] * ranks)[:ranks]) for name, values in columns]


def hccl_source(torch, rows, rank, ranks, family, iteration):
    columns = hccl_columns(ranks, family)
    # Pad the fixture pattern to one payload line, then move its columns.
    values = [columns[index % len(columns)][1][rank] for index in range(256)]
    bits = torch.tensor(values, dtype=torch.uint16).roll(iteration)
    return bits.repeat(rows * 24).view(torch.bfloat16).reshape(rows, 6144)


def tensor_summary(torch, value):
    value = value.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    dtype = torch.uint16 if value.dtype == torch.bfloat16 else torch.int32
    mask = 0xFFFF if dtype == torch.uint16 else 0xFFFFFFFF
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "finite": int(torch.isfinite(value).sum()),
            "nan": int(torch.isnan(value).sum()),
            "positive_inf": int(torch.isposinf(value).sum()),
            "negative_inf": int(torch.isneginf(value).sum()),
            "first_32_bits": [int(item) & mask for item in
                              value.view(dtype).reshape(-1)[:32].tolist()]}


def compare(torch, actual, expected):
    actual = actual.detach().cpu().contiguous()
    expected = expected.contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Numerical comparison requires identical shape and dtype")
    raw = actual.view(torch.uint8).numpy().tobytes()
    reference = expected.view(torch.uint8).numpy().tobytes()
    return {"bitwise_equal": raw == reference,
            "different_bytes": sum(left != right for left, right in zip(raw, reference)),
            "actual_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_sha256": hashlib.sha256(reference).hexdigest(),
            "first_32_actual_bits": actual.view(torch.uint16).reshape(-1)[:32].tolist(),
            "first_32_expected_bits": expected.view(torch.uint16).reshape(-1)[:32].tolist()}


def consensus(dist, ranks, value):
    gathered = [None] * ranks
    dist.all_gather_object(gathered, value)
    if any(item != value for item in gathered):
        raise ValueError(f"Rank protocol configuration differs: {gathered}")


def check_rows(torch, torch_npu, dist, native, args, rows, rank, ranks):
    owner = ctypes.c_void_p()
    consensus(dist, ranks, {"rows": rows, "capacity": rows, "epoch": 0,
                            "blocks": args.blocks, "arm": args.arm})
    native.call("prepare", rows, rank, ranks, args.blocks, ctypes.byref(owner))
    # align() does not zero memory. Native prepare zeroes and synchronizes;
    # every rank must finish that operation before any rank may scatter.
    torch.npu.synchronize()
    dist.barrier()
    device = torch.device("npu", torch.npu.current_device())
    partial = torch_npu.npu_format_cast(
        torch.empty((rows, 6144), dtype=torch.bfloat16, device=device), 2)
    local_weight = torch_npu.npu_format_cast(weight(torch, rank).to(device), 2)
    storage = [torch.full((rows * 6144 + 512,), -37, dtype=torch.bfloat16,
                          device=device) for _ in range(2)]
    first, output = [value[256:-256].reshape(rows, 6144) for value in storage]
    for value in (partial, local_weight, first, output):
        if (value.data_ptr() % 512 or not value.is_contiguous()
                or torch_npu.get_npu_format(value) != 2):
            raise ValueError("Probe requires 512-byte aligned, contiguous BF16 ND tensors")
    graph = None

    def execute():
        native.call("execute", owner, torch.npu.current_stream().npu_stream,
                    partial.data_ptr(), local_weight.data_ptr(),
                    first.data_ptr(), output.data_ptr())

    def snapshot(iteration, mode, launches=1):
        expected_first, expected_output = oracle(torch, rows, ranks, iteration)
        guards = all(bool((value[:256] == -37).all())
                     and bool((value[-256:] == -37).all()) for value in storage)
        return {
            "iteration": iteration, "mode": mode, "launches": launches,
            "fixture": "tagged" if iteration % 2 == 0 else "random",
            "guard_regions_intact": guards,
            "input": compare(torch, partial, source(torch, rows, rank, iteration)),
            "weight": compare(torch, local_weight, weight(torch, rank)),
            "first": compare(torch, first, expected_first),
            "output": compare(torch, output, expected_output),
        }

    records = []
    try:
        partial.copy_(source(torch, rows, rank, 0))
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
            fixture = source(torch, rows, rank, iteration)
            partial.copy_(fixture)
            for value in storage:
                value.fill_(-37)
            torch.npu.synchronize()
            dist.barrier()
            if rank == iteration % ranks:
                time.sleep(0.01)
            if iteration < 2:
                execute()
            else:
                graph.replay()
            torch.npu.synchronize()
            record = snapshot(iteration, "eager" if iteration < 2 else "graph")
            record["delayed_rank"] = iteration % ranks
            records.append(record)
        # No inter-rank barrier or device synchronize between these replays.
        # The immutable input lets fast/slow ranks exercise parity reuse while
        # every launch still executes both reductions and the CUBE consumer.
        partial.copy_(source(torch, rows, rank, args.replays))
        for value in storage:
            value.fill_(-37)
        torch.npu.synchronize()
        dist.barrier()
        for replay in range(args.stress_replays):
            if rank == replay % ranks:
                time.sleep(0.001)
            graph.replay()
        torch.npu.synchronize()
        records.append(snapshot(args.replays, "graph-parity-stress", args.stress_replays))
    finally:
        if graph is not None:
            graph.reset()
        torch.npu.synchronize()
        dist.barrier()
        native.call("destroy", owner)
    return {"rows": rows, "capacity": rows, "checks": records,
            "pass": all(record["guard_regions_intact"]
                        and all(record[name]["bitwise_equal"]
                                for name in ("input", "weight", "first", "output"))
                        for record in records)}


def check_hccl_rows(torch, torch_npu, dist, hccl, native, args, rows, rank, ranks):
    owner = ctypes.c_void_p()
    native.call("prepare", rows, rank, ranks, args.blocks, ctypes.byref(owner))
    torch.npu.synchronize()
    dist.barrier()
    partial = torch_npu.npu_format_cast(
        torch.empty((rows, 6144), dtype=torch.bfloat16, device="npu"), 2)
    local_weight = torch_npu.npu_format_cast(weight(torch, rank).npu(), 2)
    first, output = torch.empty_like(partial), torch.empty_like(partial)
    graph = None

    def execute():
        native.call("execute", owner, torch.npu.current_stream().npu_stream,
                    partial.data_ptr(), local_weight.data_ptr(),
                    first.data_ptr(), output.data_ptr())

    records = []
    try:
        partial.copy_(hccl_source(torch, rows, rank, ranks, args.hccl_fixtures[0], 0))
        torch.npu.synchronize()
        dist.barrier()
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            execute()
        torch.npu.synchronize()
        for family in args.hccl_fixtures:
            columns = hccl_columns(ranks, family)
            for iteration in range(4):
                parts = [hccl_source(torch, rows, sender, ranks, family, iteration)
                         for sender in range(ranks)]
                expected_first = tree_sum(parts)
                serial = parts[0].float()
                for value in parts[1:]:
                    serial = serial + value.float()
                serial = serial.bfloat16()
                expected_output = tree_sum([
                    (expected_first.float().reshape(-1, 16) @ weight(torch, peer).float())
                    .reshape(rows, 6144).bfloat16() for peer in range(ranks)])
                partial.copy_(parts[rank])
                first.fill_(-29)
                output.fill_(31)
                torch.npu.synchronize()
                dist.barrier()
                execute() if iteration < 2 else graph.replay()
                torch.npu.synchronize()
                hccl_float = partial.float()
                dist.all_reduce(hccl_float, group=hccl)
                hccl_first = hccl_float.bfloat16()
                hccl_next = (hccl_first.float().reshape(-1, 16) @ local_weight.float())
                hccl_next = hccl_next.reshape(rows, 6144).bfloat16().float()
                dist.all_reduce(hccl_next, group=hccl)
                hccl_output = hccl_next.bfloat16()
                torch.npu.synchronize()
                records.append({
                    "family": family, "iteration": iteration,
                    "mode": "eager" if iteration < 2 else "graph",
                    "columns": [{"name": name, "rank_input_bits": bits}
                                for name, bits in columns],
                    "native_first_vs_tree": compare(torch, first, expected_first),
                    "native_output_vs_tree": compare(torch, output, expected_output),
                    "native_first_vs_hccl": compare(torch, first, hccl_first.cpu()),
                    "native_output_vs_hccl": compare(torch, output, hccl_output.cpu()),
                    "hccl_first_vs_tree": compare(torch, hccl_first, expected_first),
                    "hccl_first_vs_serial": compare(torch, hccl_first, serial),
                    "hccl_first_fp32": tensor_summary(torch, hccl_float),
                    "hccl_second_fp32": tensor_summary(torch, hccl_next),
                    "input": compare(torch, partial, parts[rank]),
                })
    finally:
        if graph is not None:
            graph.reset()
        torch.npu.synchronize()
        dist.barrier()
        native.call("destroy", owner)
    return {"rows": rows, "observational": True, "checks": records,
            "hccl_tree_equality_assumed": False,
            "scope": "finite fixtures do not prove all BF16 semantics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 12, 32])
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--stress-replays", type=int, default=8)
    parser.add_argument("--frozen-driver", type=Path)
    parser.add_argument("--hccl-compare", action="store_true")
    parser.add_argument("--hccl-fixtures", nargs="+", default=["witness", "finite", "ties", "edges"],
                        choices=("witness", "finite", "ties", "edges"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ranks, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    if (not 2 <= ranks <= 16 or not args.rows
            or any(not 1 <= rows <= 32 for rows in args.rows)
            or args.blocks < 1 or args.warmup < 1 or args.replays < 4
            or args.stress_replays < 4
            or 2 * (args.warmup + args.replays + args.stress_replays + 1) > 0xFFFFFFFE):
        raise ValueError("Invalid geometry, replay count or uint32 epoch lifetime")
    prepared = args.prepared.resolve()
    library, binding = bind_library(prepared, args.arm, args.frozen_driver)
    if args.hccl_compare:
        for family in args.hccl_fixtures:
            hccl_columns(ranks, family)
    import torch
    import torch.distributed as dist
    import torch_npu

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    consensus(dist, ranks, {"arm": args.arm, "library_sha256": binding["library_sha256"],
                            "rows": args.rows, "blocks": args.blocks,
                            "warmup": args.warmup, "replays": args.replays,
                            "stress_replays": args.stress_replays,
                            "hccl_fixtures": args.hccl_fixtures if args.hccl_compare else [],
                            "endpoint": args.endpoint, "ranks": ranks})
    native = NativeProbe(library, args.arm)
    hccl = dist.new_group(backend="hccl") if args.hccl_compare else None
    native.call("initialize", rank, ranks, args.endpoint.encode(), 64 * 1024 * 1024)
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        name = line.split()[-1]
        if "shmem" in name and name.startswith("/") and Path(name).is_file():
            mapped.add(name)
    report = {"rank": rank, "ranks": ranks, "arm": args.arm,
              "scope": "synthetic two reductions with adjacent CUBE consumer; no timing",
              "binding": binding,
              "driver_sha256": digest(Path(__file__)), "torch": torch.__version__,
              "torch_npu": torch_npu.__version__,
              "device": str(torch.npu.get_device_properties(torch.npu.current_device())),
              "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "loaded_shmem": [{"path": path, "sha256": digest(Path(path))}
                               for path in sorted(mapped)],
              "shapes": []}
    try:
        for rows in args.rows:
            report["shapes"].append(check_rows(
                torch, torch_npu, dist, native, args, rows, rank, ranks))
        if args.hccl_compare:
            report["hccl"] = {"arithmetic": "BF16 -> device FP32 HCCL SUM -> BF16",
                              "environment": {name: os.environ.get(name) for name in (
                                  "HCCL_ALGO", "HCCL_ALGO_CONFIG", "HCCL_DETERMINISTIC",
                                  "HCCL_OP_EXPANSION_MODE")},
                              "shapes": [check_hccl_rows(
                                  torch, torch_npu, dist, hccl, native, args, rows, rank, ranks)
                                  for rows in args.rows]}
        report["pass"] = all(shape["pass"] for shape in report["shapes"])
    finally:
        torch.npu.synchronize()
        dist.barrier()
        native.call("finalize")
        dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.arm}-rank-{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise RuntimeError("Probe numerical gate failed; inspect the rank report")


if __name__ == "__main__":
    main()
