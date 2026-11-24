"""Compare a private tree reducer, its serial parent and FP32 HCCL."""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path

from communication_devicecheck import NativeProbe, check_tensor, make_weight
from communication_order_devicecheck import (
    LIBRARY_SHA as PARENT_SHA,
    cpu_contract as order_contract,
    device_mapping,
    make_partial as order_partial,
)


# Each tuple contains BF16 bits for ranks 0, 1 and 2. Inputs stay finite;
# several sums deliberately overflow FP32 or exercise signed zeros/subnormals.
EDGE_COLUMNS = (
    ("cancellation", (0x3F80, 0x3300, 0xBF80)),
    ("reversed_cancellation", (0x3F80, 0xBF80, 0x3300)),
    ("integer_control", (0x3F80, 0x4000, 0x4040)),
    ("positive_zero", (0x0000, 0x0000, 0x0000)),
    ("negative_zero", (0x8000, 0x8000, 0x8000)),
    ("mixed_zero", (0x8000, 0x0000, 0x8000)),
    ("two_min_subnormals", (0x0001, 0x0001, 0x0000)),
    ("normal_cancellation_tiny", (0x0080, 0x8080, 0x0001)),
    ("subnormal_to_normal", (0x007F, 0x0001, 0x0000)),
    ("subnormal_cancellation", (0x8001, 0x0001, 0x8000)),
    ("positive_overflow", (0x7F7F, 0x7F7F, 0x0000)),
    ("large_cancellation", (0x7F7F, 0xFF7F, 0xFF7F)),
    ("serial_positive_overflow", (0x7F7F, 0x7F7F, 0xFF7F)),
    ("serial_negative_overflow", (0xFF7F, 0xFF7F, 0x7F7F)),
    ("tree_positive_overflow", (0x7F7F, 0xFF7F, 0x7F7F)),
    ("tree_negative_overflow", (0xFF7F, 0x7F7F, 0xFF7F)),
    ("three_min_subnormals", (0x0001, 0x0001, 0x0001)),
    ("three_negative_subnormals", (0x8001, 0x8001, 0x8001)),
    ("negative_tiny_after_cancellation", (0x0080, 0x8001, 0x8080)),
    ("positive_small_after_cancellation", (0x3F80, 0x3300, 0xBF80)),
    ("negative_small_after_cancellation", (0x3F80, 0xB300, 0xBF80)),
    ("largest_subnormal_after_cancellation", (0x0080, 0x007F, 0x8080)),
    ("power_of_two_overflow", (0x7F00, 0x7F00, 0xFF00)),
    ("even_bf16_tie", (0x3F80, 0x3B80, 0x0000)),
    ("odd_bf16_tie", (0x3F81, 0x3B80, 0x0000)),
    ("negative_even_bf16_tie", (0xBF80, 0xBB80, 0x8000)),
    ("negative_odd_bf16_tie", (0xBF81, 0xBB80, 0x0000)),
    ("minimum_normal", (0x0080, 0x0080, 0x8080)),
    ("three_ones", (0x3F80, 0x3F80, 0x3F80)),
    ("three_negative_ones", (0xBF80, 0xBF80, 0xBF80)),
    ("maximum_cancel_zero", (0x7F7F, 0x0000, 0xFF7F)),
    ("negative_maximum_cancel_zero", (0xFF7F, 0x0000, 0x7F7F)),
)


class TreeProbe:
    """Private C ABI with the same argument layout as NativeProbe."""

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
            function = getattr(self.library, "communication_tree_probe_" + name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        self.library.communication_tree_probe_last_error.restype = ctypes.c_char_p

    def call(self, name, *values):
        function = getattr(self.library, "communication_tree_probe_" + name)
        if function(*values) != 0:
            raise RuntimeError(
                self.library.communication_tree_probe_last_error().decode())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_partial(torch, rows, rank, iteration, family):
    if family == "order":
        return order_partial(torch, rows, rank, iteration)
    bits = torch.tensor([values[rank] for _, values in EDGE_COLUMNS],
                        dtype=torch.uint16).roll(iteration)
    return bits.repeat(rows * 192).view(torch.bfloat16).reshape(rows, 6144)


def oracles(torch, rows, iteration, family):
    parts = [make_partial(torch, rows, rank, iteration, family).float()
             for rank in range(3)]
    serial = ((parts[0] + parts[1]) + parts[2]).bfloat16()
    tree = ((parts[0] + parts[2]) + parts[1]).bfloat16()
    return serial, tree


def bits(torch, value, count=32):
    value = value.detach().cpu().contiguous().reshape(-1)[:count]
    dtype = torch.uint16 if value.dtype == torch.bfloat16 else torch.int32
    mask = 0xFFFF if dtype == torch.uint16 else 0xFFFFFFFF
    return [int(item) & mask for item in value.view(dtype).tolist()]


def compare(torch, actual, expected):
    actual = actual.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Comparison tensors must have identical shape and dtype")
    raw = actual.view(torch.uint8) != expected.view(torch.uint8)
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    delta = (actual.double()[finite] - expected.double()[finite]).abs()
    return {
        "bitwise_equal": not bool(raw.any()),
        "different_bytes": int(raw.sum()),
        "finite_elements": int(finite.sum()),
        "actual_nonfinite": int((~torch.isfinite(actual)).sum()),
        "expected_nonfinite": int((~torch.isfinite(expected)).sum()),
        "finite_max_absolute": float(delta.max()) if delta.numel() else None,
        "first_32_actual_bits": bits(torch, actual),
        "first_32_expected_bits": bits(torch, expected),
    }


def cpu_contract(torch):
    original = order_contract(torch)
    assert len(EDGE_COLUMNS) == 32
    serial, tree = oracles(torch, 1, 0, "edges")
    expected = {
        0: 0x3300, 1: 0, 2: 0x40C0, 3: 0, 4: 0x8000,
        5: 0, 6: 2, 7: 1, 8: 0x0080, 9: 0,
        10: 0x7F80, 11: 0xFF7F, 12: 0x7F7F, 13: 0xFF7F,
        14: 0x7F80, 15: 0xFF80, 16: 3, 17: 0x8003,
        18: 0x8001, 19: 0x3300, 20: 0xB300, 21: 0x007F,
        22: 0x7F00, 23: 0x3F80, 24: 0x3F82, 25: 0xBF80,
        26: 0xBF82, 27: 0x0080, 28: 0x4040, 29: 0xC040,
        30: 0, 31: 0,
    }
    assert bits(torch, tree) == [expected[i] for i in range(32)]
    for rank in range(3):
        assert bool(torch.isfinite(make_partial(torch, 1, rank, 0, "edges")).all())
    for rows in (1, 12):
        for iteration in range(6):
            left, right = oracles(torch, rows, iteration, "edges")
            for value, reference in ((left, serial), (right, tree)):
                expected_row = reference[0].reshape(-1, 32).roll(iteration, 1)
                expected_full = expected_row.reshape(1, 6144).repeat(rows, 1)
                assert compare(torch, value, expected_full)["bitwise_equal"]
    return {"pass": True, "order_contract": original,
            "edge_columns": [{"name": name, "rank_input_bf16_bits": values}
                             for name, values in EDGE_COLUMNS],
            "serial_bits": bits(torch, serial), "tree_bits": bits(torch, tree),
            "ranks": 3, "rows": [1, 12], "iterations_per_family": 6}


class ProbeRun:
    def __init__(self, torch, native, rows, rank, partial, weight):
        self.torch = torch
        self.native = native
        self.partial = partial
        self.weight = weight
        self.owner = ctypes.c_void_p()
        self.graph = None
        self.first = torch.empty_like(partial)
        self.output = torch.empty_like(partial)
        self.consumer = torch.empty_like(partial)
        native.call("prepare", rows, rank, 3, 24, ctypes.byref(self.owner))

    def execute(self):
        self.native.call(
            "execute", self.owner, self.torch.npu.current_stream().npu_stream,
            self.partial.data_ptr(), self.weight.data_ptr(),
            self.first.data_ptr(), self.output.data_ptr())
        self.consumer.copy_(self.first)

    def capture(self, dist):
        for _ in range(2):
            self.execute()
        self.torch.npu.synchronize()
        dist.barrier()
        stream = self.torch.npu.Stream()
        self.graph = self.torch.npu.NPUGraph()
        with self.torch.npu.graph(self.graph, stream=stream):
            self.execute()
        self.torch.npu.synchronize()

    def run(self, iteration):
        self.first.fill_(-29)
        self.output.fill_(31)
        self.consumer.fill_(-37)
        self.torch.npu.synchronize()
        if iteration < 2:
            self.execute()
        else:
            self.graph.replay()
        self.torch.npu.synchronize()
        return self.first.cpu(), self.consumer.cpu()

    def close(self):
        if self.graph is not None:
            self.graph.reset()
        self.torch.npu.synchronize()
        self.native.call("destroy", self.owner)

    def addresses(self):
        return {name: getattr(self, name).data_ptr() for name in
                ("partial", "weight", "first", "output", "consumer")}


def check_rows(torch, torch_npu, dist, hccl, parent, candidate, rows, rank):
    partial = torch_npu.npu_format_cast(
        make_partial(torch, rows, rank, 0, "order").npu(), 2)
    weight = torch_npu.npu_format_cast(make_weight(torch, rank).npu(), 2)
    check_tensor(torch_npu, partial, [rows, 6144])
    check_tensor(torch_npu, weight, [16, 16])
    arms = []
    records = []
    addresses = []
    try:
        for native in (parent, candidate):
            arm = ProbeRun(torch, native, rows, rank, partial, weight)
            arms.append(arm)
            for value in (arm.first, arm.output, arm.consumer):
                check_tensor(torch_npu, value, [rows, 6144])
            arm.capture(dist)
            addresses.append(arm.addresses())
        assert addresses[0]["partial"] == addresses[1]["partial"]
        assert addresses[0]["weight"] == addresses[1]["weight"]
        outputs = [item[name] for item in addresses
                   for name in ("first", "output", "consumer")]
        assert len(set(outputs)) == 6
        assert not set(outputs) & {partial.data_ptr(), weight.data_ptr()}
        for family in ("order", "edges"):
            for iteration in range(6):
                source = make_partial(torch, rows, rank, iteration, family)
                partial.copy_(source)
                torch.npu.synchronize()
                dist.barrier()
                parent_first, parent_consumer = arms[0].run(iteration)
                candidate_first, candidate_consumer = arms[1].run(iteration)
                serial, tree = oracles(torch, rows, iteration, family)
                hccl_float = partial.float()
                widened_bits = bits(torch, hccl_float)
                dist.all_reduce(hccl_float, group=hccl)
                hccl_bf16 = hccl_float.bfloat16()
                torch.npu.synchronize()
                records.append({
                    "family": family, "iteration": iteration,
                    "mode": "eager" if iteration < 2 else "graph",
                    "input_unchanged": compare(torch, partial, source),
                    "hccl_input_first_32_fp32_bits": widened_bits,
                    "hccl_sum_first_32_fp32_bits": bits(torch, hccl_float),
                    "parent_vs_serial": compare(torch, parent_first, serial),
                    "parent_consumer": compare(torch, parent_consumer, parent_first),
                    "candidate_vs_tree": compare(torch, candidate_first, tree),
                    "candidate_consumer": compare(
                        torch, candidate_consumer, candidate_first),
                    "hccl_vs_tree": compare(torch, hccl_bf16, tree),
                    "candidate_vs_hccl": compare(torch, candidate_first, hccl_bf16),
                    "parent_vs_hccl": compare(torch, parent_first, hccl_bf16),
                })
        dist.barrier()
    finally:
        for arm in reversed(arms):
            arm.close()
    normal_checks = ("input_unchanged", "parent_vs_serial", "parent_consumer",
                     "candidate_vs_tree", "candidate_consumer",
                     "hccl_vs_tree", "candidate_vs_hccl")
    return {
        "rows": rows, "checks": records,
        "parent_addresses": addresses[0], "candidate_addresses": addresses[1],
        "order_pass": all(record[name]["bitwise_equal"] for record in records
                          if record["family"] == "order" for name in normal_checks),
        "edge_hccl_equal": all(record["candidate_vs_hccl"]["bitwise_equal"]
                               for record in records if record["family"] == "edges"),
        "all_consumers_pass": all(record[name]["bitwise_equal"]
                                  for record in records for name in
                                  ("input_unchanged", "parent_consumer",
                                   "candidate_consumer")),
    }


def loaded_libraries():
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        name = line.split()[-1]
        if (name.startswith("/") and Path(name).is_file()
                and any(token in name for token in
                        ("/libshmem", "/aclshmem_bootstrap", "/libhccl"))):
            mapped.add(name)
    return {name: digest(Path(name)) for name in sorted(mapped)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--endpoint")
    parser.add_argument("--physical-devices", type=int, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    import torch

    contract = cpu_contract(torch)
    if args.cpu_only:
        args.output.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
        return
    if any(value is None for value in (args.parent, args.candidate,
                                      args.candidate_sha256, args.endpoint,
                                      args.physical_devices)):
        parser.error("Device mode requires both libraries, candidate hash and devices")
    if digest(args.parent) != PARENT_SHA:
        raise ValueError("The accepted serial parent changed")
    if digest(args.candidate) != args.candidate_sha256:
        raise ValueError("The frozen private candidate changed")
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 3:
        raise ValueError("This comparison requires exactly three ranks")
    import torch.distributed as dist
    import torch_npu

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    mapping = device_mapping(torch, args.physical_devices, rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    hccl = dist.new_group(backend="hccl",
                          timeout=datetime.timedelta(seconds=180))
    parent = NativeProbe(args.parent)
    candidate = TreeProbe(args.candidate)
    parent.call("initialize", rank, 3, args.endpoint.encode(), 64 * 1024 * 1024)
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rank": rank, "ranks": 3, "cpu_contract": contract, "shapes": [],
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device_mapping": mapping,
        "parent_sha256": PARENT_SHA, "candidate_sha256": args.candidate_sha256,
        "driver_sha256": digest(Path(__file__)),
        "hccl_arithmetic": "BF16.float -> HCCL FP32 SUM -> BF16",
        "compared_boundary": "First reduction and its immediate consumer",
        "execution": "Each probe executes two reductions and dependent CUBE",
        "shmem_lifetime": "Parent initializes once; both probes own distinct windows",
        "environment": {name: os.environ.get(name) for name in (
            "ASCEND_RT_VISIBLE_DEVICES", "HCCL_OP_EXPANSION_MODE",
            "HCCL_DETERMINISTIC", "HCCL_ALGO", "HCCL_ALGO_CONFIG",
            "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_PROCESS_LOG_PATH")},
    }
    try:
        for rows in (1, 12):
            report["shapes"].append(check_rows(
                torch, torch_npu, dist, hccl, parent, candidate, rows, rank))
        report["order_pass"] = all(shape["order_pass"] for shape in report["shapes"])
        report["edge_hccl_equal"] = all(shape["edge_hccl_equal"]
                                        for shape in report["shapes"])
        report["pass"] = report["order_pass"] and all(
            shape["all_consumers_pass"] for shape in report["shapes"])
        report["loaded_libraries"] = loaded_libraries()
    finally:
        torch.npu.synchronize()
        dist.barrier()
        parent.call("finalize")
        dist.destroy_process_group(hccl)
        dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"rank-{rank}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["pass"]:
        raise RuntimeError(f"Tree reduction order/consumer contract failed: {path}")
    print(path, flush=True)


if __name__ == "__main__":
    main()
