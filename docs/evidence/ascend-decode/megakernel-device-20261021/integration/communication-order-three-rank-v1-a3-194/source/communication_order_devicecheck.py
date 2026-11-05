"""Compare the three-rank ordered reduction with installed FP32 HCCL."""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path

from communication_devicecheck import (
    NativeProbe, check_tensor, difference, make_weight,
)


LIBRARY_SHA = "1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f"
DEVICE_HEADER_SHA = "cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7"


def make_partial(torch, rows, rank, iteration):
    epsilon = 2.0 ** -25
    columns = ((1.0, 1.0, 1.0),
               (epsilon, -1.0, 2.0),
               (-1.0, epsilon, 3.0))
    scale = 2.0 ** (iteration - 2)
    pattern = torch.tensor(columns[rank], dtype=torch.float32) * scale
    return pattern.repeat(rows * 2048).reshape(rows, 6144).bfloat16()


def oracles(torch, rows, iteration):
    values = [make_partial(torch, rows, rank, iteration).float()
              for rank in range(3)]
    serial = ((values[0] + values[1]) + values[2]).bfloat16()
    tree = ((values[0] + values[2]) + values[1]).bfloat16()
    return serial, tree


def cpu_contract(torch):
    for rows in (1, 12):
        for iteration in range(6):
            serial, tree = oracles(torch, rows, iteration)
            scale = 2.0 ** (iteration - 2)
            expected_serial = torch.tensor(
                [0.0, (2.0 ** -25) * scale, 6.0 * scale]).bfloat16()
            expected_tree = expected_serial[[1, 0, 2]]
            assert torch.equal(serial[0, :3], expected_serial)
            assert torch.equal(tree[0, :3], expected_tree)
            assert int((serial != tree).sum()) == rows * 4096
    return {"pass": True, "ranks": 3, "rows": [1, 12], "iterations": 6,
            "different_elements_per_row": 4096,
            "fixture_columns": ["1 + epsilon - 1", "1 - 1 + epsilon",
                                "exact integer control"],
            "epsilon": 2.0 ** -25}


def device_mapping(torch, expected, rank):
    runtime = ctypes.CDLL("libascendcl.so")
    convert = runtime.aclrtGetPhyDevIdByLogicDevId
    convert.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    convert.restype = ctypes.c_int
    access = runtime.aclrtDeviceCanAccessPeer
    access.argtypes = [ctypes.POINTER(ctypes.c_int32),
                       ctypes.c_int32, ctypes.c_int32]
    access.restype = ctypes.c_int
    physical = []
    for logical in range(3):
        value = ctypes.c_int32(-1)
        if convert(logical, ctypes.byref(value)) != 0:
            raise RuntimeError("Cannot resolve logical-to-physical devices")
        physical.append(value.value)
    if (torch.npu.device_count() != 3 or physical != expected
            or torch.npu.current_device() != rank):
        raise RuntimeError(f"Unexpected physical device mapping: {physical}")
    peers = {}
    for peer in range(3):
        if peer != rank:
            value = ctypes.c_int32()
            if access(ctypes.byref(value), rank, peer) != 0 or value.value != 1:
                raise RuntimeError(f"Rank {rank} cannot access peer {peer}")
            peers[str(peer)] = value.value
    return {"logical_to_physical": physical, "logical_rank": rank,
            "direct_peer_access": peers}


def check_rows(torch, torch_npu, dist, hccl, native, rows, rank):
    owner = ctypes.c_void_p()
    native.call("prepare", rows, rank, 3, 24, ctypes.byref(owner))
    partial = torch_npu.npu_format_cast(
        make_partial(torch, rows, rank, 0).npu(), 2)
    weight = torch_npu.npu_format_cast(make_weight(torch, rank).npu(), 2)
    first = torch.empty_like(partial)
    output = torch.empty_like(partial)
    consumer = torch.empty_like(partial)
    for value in (partial, first, output, consumer):
        check_tensor(torch_npu, value, [rows, 6144])
    check_tensor(torch_npu, weight, [16, 16])
    graph = None

    def execute():
        native.call("execute", owner, torch.npu.current_stream().npu_stream,
                    partial.data_ptr(), weight.data_ptr(), first.data_ptr(),
                    output.data_ptr())
        consumer.copy_(first)

    records = []
    try:
        for _ in range(2):
            execute()
        torch.npu.synchronize()
        dist.barrier()
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            execute()
        torch.npu.synchronize()
        for iteration in range(6):
            partial.copy_(make_partial(torch, rows, rank, iteration))
            first.fill_(-29)
            output.fill_(31)
            consumer.fill_(-37)
            torch.npu.synchronize()
            dist.barrier()
            if iteration < 2:
                execute()
            else:
                graph.replay()
            torch.npu.synchronize()
            serial, tree = oracles(torch, rows, iteration)
            first_cpu = first.cpu()
            consumer_cpu = consumer.cpu()
            hccl_float = partial.float()
            dist.all_reduce(hccl_float, group=hccl)
            hccl_bf16 = hccl_float.bfloat16()
            torch.npu.synchronize()
            records.append({
                "iteration": iteration,
                "mode": "eager" if iteration < 2 else "graph",
                "native_vs_serial": difference(torch, first_cpu, serial),
                "consumer_vs_native": difference(
                    torch, consumer_cpu, first_cpu),
                "hccl_vs_tree": difference(torch, hccl_bf16, tree),
                "native_vs_hccl": difference(torch, first_cpu, hccl_bf16),
                "first_three_native": first_cpu[0, :3].float().tolist(),
                "first_three_hccl": hccl_bf16.cpu()[0, :3].float().tolist(),
                "first_three_serial": serial[0, :3].float().tolist(),
                "first_three_tree": tree[0, :3].float().tolist(),
            })
        torch.npu.synchronize()
        dist.barrier()
    finally:
        if graph is not None:
            graph.reset()
        torch.npu.synchronize()
        native.call("destroy", owner)
    return {"rows": rows, "checks": records,
            "pass": all(record[name]["bitwise_equal"] for record in records
                        for name in ("native_vs_serial", "consumer_vs_native")),
            "hccl_matches_tree": all(record["hccl_vs_tree"]["bitwise_equal"]
                                     for record in records),
            "order_difference_observed": all(
                record["native_vs_hccl"]["mismatch_count"] == rows * 4096
                for record in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--physical-devices", type=int, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    import torch

    cpu = cpu_contract(torch)
    if args.cpu_only:
        args.output.write_text(json.dumps(cpu, indent=2) + "\n")
        return
    if (args.library is None or args.endpoint is None
            or args.physical_devices is None):
        parser.error("Device mode requires library, endpoint and physical devices")
    if hashlib.sha256(args.library.read_bytes()).hexdigest() != LIBRARY_SHA:
        raise ValueError("The accepted communication probe library changed")
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 3:
        raise ValueError("This diagnostic requires exactly three ranks")
    import torch.distributed as dist
    import torch_npu

    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    mapping = device_mapping(torch, args.physical_devices, rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    hccl = dist.new_group(backend="hccl",
                          timeout=datetime.timedelta(seconds=180))
    native = NativeProbe(args.library)
    native.call("initialize", rank, 3, args.endpoint.encode(), 64 * 1024 * 1024)
    report = {"rank": rank, "ranks": 3, "cpu_contract": cpu, "shapes": [],
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "library_sha256": LIBRARY_SHA,
              "device_mapping": mapping,
              "communication_device_header_sha256": DEVICE_HEADER_SHA,
              "hccl_arithmetic": "BF16.float -> HCCL FP32 SUM -> BF16",
              "compared_boundary": "First reduction only",
              "native_execution": (
                  "Two reductions and a dependent CUBE projection"),
              "environment": {name: os.environ.get(name) for name in (
                  "ASCEND_RT_VISIBLE_DEVICES", "HCCL_OP_EXPANSION_MODE",
                  "HCCL_DETERMINISTIC", "HCCL_ALGO", "HCCL_ALGO_CONFIG",
                  "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_PROCESS_LOG_PATH")}}
    try:
        for rows in (1, 12):
            report["shapes"].append(check_rows(
                torch, torch_npu, dist, hccl, native, rows, rank))
        report["pass"] = all(shape["pass"] for shape in report["shapes"])
        report["order_difference_observed"] = all(
            shape["order_difference_observed"] for shape in report["shapes"])
        mapped = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            name = line.split()[-1]
            if (name.startswith("/") and Path(name).is_file()
                    and any(token in name for token in (
                        "/libshmem", "/aclshmem_bootstrap", "/libhccl"))):
                mapped.add(name)
        report["loaded_libraries"] = {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in sorted(mapped)}
    finally:
        torch.npu.synchronize()
        dist.barrier()
        native.call("finalize")
        dist.destroy_process_group(hccl)
        dist.destroy_process_group()
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"rank-{rank}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["pass"]:
        raise RuntimeError(f"Native ordered-reduction contract failed: {path}")
    print(path, flush=True)


if __name__ == "__main__":
    main()
