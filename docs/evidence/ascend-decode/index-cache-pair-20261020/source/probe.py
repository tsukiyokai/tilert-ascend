"""Compare paired C8 cache writes using a byte view of FP16 scales."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import struct
import time
import traceback


CAPACITY = 9344
PAGE_SIZE = 128
HEAD_DIM = 128
WIDTHS = (1, 8, 32)
QUOTA = 256 * 2**20
SOURCE_ROOT = Path("/workspace/src/vllm-ascend-unified")
BUILTIN = Path(
    "/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe"
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def fixture(width, state):
    seed = width * 17 + (state == "B")
    generator = random.Random(seed)
    start = CAPACITY - width if state == "B" else PAGE_SIZE - width // 2
    slots = list(range(start, start + width))
    keys = generator.randbytes(width * HEAD_DIM)
    scales = b"".join(
        struct.pack("<e", 0.004 + generator.random() * 0.016)
        for _ in slots
    )
    return slots, keys, scales


def initial_cache():
    return bytearray(b"\x5a" * (CAPACITY * HEAD_DIM)), bytearray(
        struct.pack("<e", 0.125) * CAPACITY
    )


def reference_update(caches, payload):
    keys, scales = caches
    slots, key_input, scale_input = payload
    for row, slot in enumerate(slots):
        keys[slot * HEAD_DIM:(slot + 1) * HEAD_DIM] = key_input[
            row * HEAD_DIM:(row + 1) * HEAD_DIM
        ]
    for row, slot in enumerate(slots):
        scales[slot * 2:(slot + 1) * 2] = scale_input[row * 2:(row + 1) * 2]


def paired_address_update(caches, payload):
    keys, scales = caches
    slots, key_input, scale_input = payload
    for row, slot in enumerate(slots):
        block, offset = divmod(slot, PAGE_SIZE)
        key_offset = block * PAGE_SIZE * HEAD_DIM + offset * HEAD_DIM
        scale_offset = block * PAGE_SIZE * 2 + offset * 2
        keys[key_offset:key_offset + HEAD_DIM] = key_input[
            row * HEAD_DIM:(row + 1) * HEAD_DIM
        ]
        scales[scale_offset:scale_offset + 2] = scale_input[
            row * 2:(row + 1) * 2
        ]


def cpu_contract():
    results = []
    for width in WIDTHS:
        reference, paired = initial_cache(), initial_cache()
        for state in ("A", "B", "reset"):
            if state == "reset":
                reference, paired = initial_cache(), initial_cache()
            payload = fixture(width, "A" if state == "reset" else state)
            slots, keys, scales = payload
            assert len(set(slots)) == width
            assert all(0 <= slot < CAPACITY for slot in slots)
            reference_update(reference, payload)
            paired_address_update(paired, payload)
            assert reference == paired
            results.append({
                "width": width, "state": state, "slots": slots,
                "key_input_sha256": digest(keys),
                "scale_input_sha256": digest(scales),
                "cache_sha256": [digest(value) for value in reference],
                "full_cache_byte_equal": True,
            })
    return {
        "scope": "CPU address and byte contract; no NPU/API execution",
        "cases": results,
        "conditions": [
            "Unique nonnegative physical slots below9344; Norm layout",
            "K INT8[M,1,128], FP16 scale storage viewed as INT8[M,1,2]",
            "FP16 scale cache[...,1] viewed as contiguous INT8[...,2]",
            "No quantization, casting, or source/destination aliasing",
        ],
    }


def binary_identity(descriptor, inspect_binary):
    metadata = json.loads(descriptor.read_text())
    binary = descriptor.with_suffix(".o")
    if inspect_binary:
        assert digest(binary.read_bytes()) == metadata["sha256"]
    return {"path": str(binary), "sha256": metadata["sha256"]}


def source_receipt(inspect_binaries):
    kernel = BUILTIN / "impl/ops_transformer/ascendc/scatter_pa_kv_cache/op_kernel"
    paths = [
        Path(__file__).resolve(),
        SOURCE_ROOT / "vllm_ascend/attention/indexer.py",
        SOURCE_ROOT / "vllm_ascend/device/device_op.py",
        kernel / "scatter_pa_kv_cache.cpp",
        kernel / "scatter_pa_kv_cache_normal.h",
        Path("/opt/ascend/cann-9.2.0/include/aclnnop/aclnn_scatter_pa_kv_cache.h"),
        BUILTIN / "config/ascend910_93/aic-ascend910_93-ops-info-transformer.json",
    ]
    descriptor = (
        BUILTIN / "kernel/ascend910_93/ops_transformer/scatter_pa_kv_cache"
        / "ScatterPaKvCache_346b9fee4db79c17fefad5dfbb199fc6.json"
    )
    metadata = json.loads(descriptor.read_text())
    baseline = Path(
        "/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay"
        "/vendors/custom_transformer/op_impl/ai_core/tbe/kernel"
        "/ascend910_93/scatter_nd_update_sk"
    )
    baseline_descriptors = [
        baseline / "ScatterNdUpdateSk_38f6c0f42e6dd25f1ba132d7ecf7990f.json",
        baseline / "ScatterNdUpdateSk_0be86e3b56504af1b118b0d1a2a8e1bc.json",
    ]
    return {
        "files": [file_identity(path) for path in paths],
        "candidate_descriptor": file_identity(descriptor),
        "candidate_elf": binary_identity(descriptor, inspect_binaries),
        "baseline_descriptors": [file_identity(path)
                                 for path in baseline_descriptors],
        "baseline_elfs": [binary_identity(path, inspect_binaries)
                          for path in baseline_descriptors],
        "elf_hashes_verified_in_process": inspect_binaries,
        "descriptor_is_loaded_elf_proof": False,
        "candidate_signature": metadata["supportInfo"]["inputs"],
        "candidate_kernels": metadata["kernelList"],
        "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
        "process_log_path": os.environ.get("ASCEND_PROCESS_LOG_PATH", ""),
    }


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def tensor_layout(tensor):
    return {
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype), "npu_format": torch_npu.get_npu_format(tensor),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def memory_snapshot(device):
    free, total = torch.npu.mem_get_info(device)
    return {
        "free_bytes": free, "total_bytes": total,
        "allocated_bytes": torch.npu.memory_allocated(device),
        "reserved_bytes": torch.npu.memory_reserved(device),
        "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(device),
        "host_meminfo": Path("/proc/meminfo").read_text().splitlines()[:5],
    }


def cpu_tensor(data, dtype, shape):
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def scale_byte_views(scale_input, scale_cache, byte_dtype):
    assert scale_input.ndim == 2 and scale_input.shape[-1] == 1
    assert scale_cache.ndim == 4 and scale_cache.shape[-1] == 1
    assert scale_input.dtype == scale_cache.dtype
    assert scale_input.element_size() == 2
    assert scale_input.is_contiguous() and scale_cache.is_contiguous()
    value = scale_input.view(byte_dtype).view(-1, 1, 2)
    value_cache = scale_cache.view(byte_dtype)
    for original, byte_view in ((scale_input, value), (scale_cache, value_cache)):
        assert byte_view.element_size() == 1 and byte_view.is_contiguous()
        assert original.data_ptr() == byte_view.data_ptr()
        assert original.numel() * 2 == byte_view.numel()
    return value, value_cache


def run_width(width, device, pairs, calls, progress):
    base_keys, base_scales = initial_cache()
    base = (
        cpu_tensor(base_keys, torch.int8, (73, 128, 1, 128)).to(device),
        cpu_tensor(base_scales, torch.float16, (73, 128, 1, 1)).to(device),
    )
    caches = {arm: tuple(value.clone() for value in base)
              for arm in ("two_scatter", "paired_scatter")}
    keys = torch.empty((width, 128), dtype=torch.int8, device=device)
    scales = torch.empty((width, 1), dtype=torch.float16, device=device)
    slots = torch.empty(width, dtype=torch.int32, device=device)
    paired_value, paired_value_cache = scale_byte_views(
        scales, caches["paired_scatter"][1], torch.int8,
    )
    payloads = {}
    for state in ("A", "B"):
        positions, key_input, scale_input = fixture(width, state)
        payloads[state] = (
            torch.tensor(positions, dtype=torch.int32, device=device),
            cpu_tensor(key_input, torch.int8, (width, 128)).to(device),
            cpu_tensor(scale_input, torch.float16, (width, 1)).to(device),
        )

    def set_input(state):
        values = payloads["A" if state == "reset" else state]
        for target, source in zip((slots, keys, scales), values):
            target.copy_(source)

    def reset(arm):
        for target, source in zip(caches[arm], base):
            target.copy_(source)

    def update(arm):
        progress["arm"] = arm
        key_cache, scale_cache = caches[arm]
        if arm == "two_scatter":
            torch.ops._C_ascend.npu_scatter_nd_update_sk(
                key_cache.view(-1, 128), slots.view(-1, 1), keys,
            )
            torch.ops._C_ascend.npu_scatter_nd_update_sk(
                scale_cache.view(-1, 1), slots.view(-1, 1), scales,
            )
        else:
            torch_npu.npu_scatter_pa_kv_cache(
                key=keys.view(width, 1, 128),
                value=paired_value,
                key_cache=key_cache, value_cache=paired_value_cache,
                slot_mapping=slots, cache_mode="Norm",
            )

    def check(arm, expected):
        actual = [tensor_bytes(value) for value in caches[arm]]
        for name, actual_bytes, expected_bytes in zip(
                ("key", "scale"), actual, expected):
            if actual_bytes != expected_bytes:
                mismatch = next(i for i, (a, b) in enumerate(
                    zip(actual_bytes, expected_bytes)) if a != b)
                raise AssertionError((width, arm, name, mismatch))
        return [digest(value) for value in actual]

    results = {"width": width, "validation": [], "samples": [],
               "cache_layout": [tensor_layout(value) for value in base],
               "input_layout": [tensor_layout(value) for value in (keys, scales, slots)],
               "paired_value_layout": [tensor_layout(value) for value in
                                       (paired_value, paired_value_cache)],
               "scale_byte_views_share_original_storage": True}
    graphs = {}
    for phase in ("eager", "graph"):
        progress.update(width=width, phase=phase)
        for arm in caches:
            reset(arm)
        if phase == "graph":
            set_input("A")
            torch.npu.synchronize()
            for arm in caches:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    update(arm)
                graphs[arm] = graph
            for arm in caches:
                reset(arm)
        expected = initial_cache()
        for state in ("A", "B", "reset"):
            progress["state"] = state
            if state == "reset":
                expected = initial_cache()
                for arm in caches:
                    reset(arm)
            set_input(state)
            reference_update(expected, fixture(width, "A" if state == "reset" else state))
            for arm in caches:
                update(arm) if phase == "eager" else graphs[arm].replay()
            torch.npu.synchronize()
            hashes = {arm: check(arm, expected) for arm in caches}
            results["validation"].append({
                "phase": phase, "state": state, "full_cache_byte_equal": True,
                "actual_cache_sha256": hashes,
            })

    set_input("A")
    torch.npu.synchronize()
    timing_graphs = {}
    progress.update(phase="timing_capture", state="A")
    for arm in caches:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(calls):
                update(arm)
        timing_graphs[arm] = graph
    for arm in caches:
        timing_graphs[arm].replay()
    torch.npu.synchronize()
    progress.update(phase="timing", state="A")
    for pair in range(pairs):
        order = list(caches) if pair % 2 == 0 else list(reversed(caches))
        for arm in order:
            progress.update(arm=arm, pair=pair)
            reset(arm)
            torch.npu.synchronize()
            memory = memory_snapshot(device) if pair in (0, pairs - 1) else None
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            started = time.perf_counter_ns()
            timing_graphs[arm].replay()
            end.record()
            end.synchronize()
            wall_us = (time.perf_counter_ns() - started) / 1000
            results["samples"].append({
                "pair": pair, "order": order, "arm": arm,
                "calls": calls, "device_us_per_call": begin.elapsed_time(end) * 1000 / calls,
                "wall_us_per_call": wall_us / calls, "memory": memory,
            })
    expected = initial_cache()
    reference_update(expected, fixture(width, "A"))
    results["timing_output_sha256"] = {arm: check(arm, expected) for arm in caches}
    results["median_device_us"] = {
        arm: statistics.median(row["device_us_per_call"] for row in results["samples"]
                               if row["arm"] == arm)
        for arm in caches
    }
    results["paired_device_delta_us"] = [
        next(row["device_us_per_call"] for row in results["samples"]
             if row["pair"] == pair and row["arm"] == "two_scatter")
        - next(row["device_us_per_call"] for row in results["samples"]
               if row["pair"] == pair and row["arm"] == "paired_scatter")
        for pair in range(pairs)
    ]
    results["memory_final"] = memory_snapshot(device)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-npu", action="store_true")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--calls", type=int, default=10)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Synthetic paired C8 index-cache store comparison",
              "npu_launched": False, "model_tpot_measured": False,
              "quota_bytes": QUOTA, "quota_scope": "Torch allocator only",
              "timing_scope": (
                  "One replay of a graph containing calls_per_sample updates; "
                  "event and wall divided by calls_per_sample; identical "
                  "per-arm cache reset and synchronization excluded"
              ),
              "cpu_contract": cpu_contract(),
              "source": source_receipt(not arguments.run_npu),
              "pairs": arguments.pairs, "calls_per_sample": arguments.calls,
              "kernels_per_call": {"two_scatter": 2, "paired_scatter": 1},
              "live_tensor_bytes_by_width": {
                  width: CAPACITY * 130 * 3 + width * 402
                  for width in WIDTHS
              },
              "cache_store_invocations_by_width": (
                  2 * (7 + (arguments.pairs + 2) * arguments.calls)
                  if arguments.run_npu else 0
              ),
              "validation_graph_calls": 1,
              "timing_graph_calls": arguments.calls,
              "warm_timing_graph_replays_per_arm": 1,
              "progress": {},
              "widths": []}
    result = arguments.output / "result.json"
    try:
        if arguments.run_npu:
            global torch, torch_npu
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            import torch
            import torch_npu
            from vllm_ascend.utils import enable_custom_op

            assert 1 <= arguments.pairs <= 30 and 1 <= arguments.calls <= 10
            assert enable_custom_op()
            torch.set_num_threads(1)
            torch.npu.set_device(arguments.device)
            total = torch.npu.get_device_properties(arguments.device).total_memory
            torch.npu.set_per_process_memory_fraction(QUOTA / total)
            device = torch.device("npu", arguments.device)
            report["device"] = arguments.device
            report["schemas"] = {
                "baseline": str(torch.ops._C_ascend.npu_scatter_nd_update_sk.default._schema),
                "candidate": str(torch.ops.npu.npu_scatter_pa_kv_cache.default._schema),
            }
            report["memory_before"] = memory_snapshot(device)
            report["npu_launched"] = True
            for width in WIDTHS:
                report["widths"].append(run_width(
                    width, device, arguments.pairs, arguments.calls,
                    report["progress"],
                ))
                result.write_text(json.dumps(report, indent=2) + "\n")
                gc.collect()
                torch.npu.empty_cache()
            report["memory_after"] = memory_snapshot(device)
        report["passed"] = True
    except Exception:
        report["passed"] = False
        report["exception"] = traceback.format_exc()
        raise
    finally:
        result.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"result": str(result), "passed": report.get("passed"),
                          "npu_launched": report["npu_launched"]}))


if __name__ == "__main__":
    main()
