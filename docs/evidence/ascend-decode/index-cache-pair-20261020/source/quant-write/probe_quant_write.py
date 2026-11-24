"""Compare frozen indexer methods after real dynamic quantization."""

import argparse
import ast
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT.parent / "probe.py"
spec = importlib.util.spec_from_file_location("cache_pair_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
assert base.digest(BASE_PATH.read_bytes()) == (
    "18df66d94d4c0cdb3e6be684762001885bf6e713cbb9595279813f763d062c27"
)
ARMS = ("control", "candidate")
KEY_OFFSET = 256
KEY_BYTES = base.CAPACITY * 128
SCALE_OFFSET = KEY_OFFSET + KEY_BYTES + 64
SCALE_BYTES = base.CAPACITY * 2
STORAGE_BYTES = SCALE_OFFSET + SCALE_BYTES + 256


def backing_bytes():
    raw = bytearray(b"\xa5" * STORAGE_BYTES)
    keys, scales = base.initial_cache()
    raw[KEY_OFFSET:KEY_OFFSET + KEY_BYTES] = keys
    raw[SCALE_OFFSET:SCALE_OFFSET + SCALE_BYTES] = scales
    return raw


def byte_regions(raw):
    return (memoryview(raw)[KEY_OFFSET:KEY_OFFSET + KEY_BYTES],
            memoryview(raw)[SCALE_OFFSET:SCALE_OFFSET + SCALE_BYTES])


def input_fixture(width, state):
    slots, raw, _ = base.fixture(width, state)
    values = [(value if value < 128 else value - 256) / 16 for value in raw]
    bf16 = b"".join(struct.pack("<f", value)[2:] for value in values)
    return slots, bf16


def cpu_check():
    sources = {arm: ast.parse((ROOT / arm / "indexer.py").read_text())
               for arm in ARMS}
    methods = {}
    for arm, tree in sources.items():
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "AscendSFAIndexerBackend")
        methods[arm] = next(node for node in cls.body
                            if isinstance(node, ast.FunctionDef)
                            and node.name == "write_cache")
    old, new = methods["control"], methods["candidate"]
    assert len(new.body) == len(old.body) + 1
    assert ast.dump(ast.Module(body=new.body[:2] + new.body[3:], type_ignores=[])) == (
        ast.dump(ast.Module(body=old.body, type_ignores=[]))
    )
    branch = new.body[2]
    assert isinstance(branch, ast.If) and isinstance(branch.body[-1], ast.Return)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ("to", "clone", "contiguous", "type")
                   for node in ast.walk(branch))
    new.body = old.body
    assert ast.dump(sources["control"]) == ast.dump(sources["candidate"])
    cases = []
    for width in base.WIDTHS:
        reference, paired = backing_bytes(), backing_bytes()
        for state in ("A", "B", "reset"):
            if state == "reset":
                reference, paired = backing_bytes(), backing_bytes()
            payload = base.fixture(width, "A" if state == "reset" else state)
            base.reference_update(byte_regions(reference), payload)
            base.paired_address_update(byte_regions(paired), payload)
            assert reference == paired
            slots, inputs = input_fixture(width, "A" if state == "reset" else state)
            cases.append({"width": width, "state": state, "slots": slots,
                          "bf16_input_sha256": base.digest(inputs),
                          "full_backing_byte_equal": True})
    return {"only_change": "INT8 C8 early return in write_cache",
            "other_module_ast_unchanged": True, "cases": cases,
            "key_byte_offset": KEY_OFFSET, "scale_byte_offset": SCALE_OFFSET,
            "backing_bytes": STORAGE_BYTES, "npu_launched": False}


def load_method(arm):
    name = "vllm_ascend.attention.cache_pair_" + arm
    spec = importlib.util.spec_from_file_location(name, ROOT / arm / "indexer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.AscendSFAIndexerBackend.write_cache


def run_width(width, device, methods, pairs, calls, progress):
    original = base.cpu_tensor(backing_bytes(), torch.uint8, (STORAGE_BYTES,)).to(device)
    storage = {arm: original.clone() for arm in ARMS}
    holders = {}
    for arm, raw in storage.items():
        key = raw[KEY_OFFSET:KEY_OFFSET + KEY_BYTES].view(torch.int8).view(73, 128, 1, 128)
        scale = raw[SCALE_OFFSET:SCALE_OFFSET + SCALE_BYTES].view(torch.float16).view(73, 128, 1, 1)
        holders[arm] = SimpleNamespace(
            k_cache=SimpleNamespace(kv_cache=(key, scale)),
            enable_sparse_li_c8=True, enable_sparse_li_quant=True,
            c8_k_cache_dtype=torch.int8,
        )
    inputs = torch.empty((width, 128), dtype=torch.bfloat16, device=device)
    slots = torch.empty(width, dtype=torch.int32, device=device)
    payloads = {}
    for state in ("A", "B"):
        positions, raw = input_fixture(width, state)
        payloads[state] = (
            torch.tensor(positions, dtype=torch.int32, device=device),
            base.cpu_tensor(raw, torch.bfloat16, (width, 128)).to(device),
        )
    produced = {}

    def set_input(state):
        positions, values = payloads["A" if state == "reset" else state]
        slots.copy_(positions)
        inputs.copy_(values)

    def update(arm):
        progress["arm"] = arm
        key, scale = torch_npu.npu_dynamic_quant(inputs, dst_type=torch.int8)
        scale = scale.to(torch.float16).unsqueeze(-1)
        produced[arm] = (key, scale)
        methods[arm](holders[arm], key, scale, slots)

    def check(expected):
        hashes = {}
        for arm in ARMS:
            actual = base.tensor_bytes(storage[arm])
            assert actual == expected, (width, arm, next(
                index for index, (a, b) in enumerate(zip(actual, expected)) if a != b))
            hashes[arm] = base.digest(actual)
        return hashes

    def expect_update(expected, state):
        positions, _ = input_fixture(width, "A" if state == "reset" else state)
        key, scale = (base.tensor_bytes(value) for value in produced["control"])
        base.reference_update(byte_regions(expected), (positions, key, scale))

    result = {"width": width, "validation": [], "samples": [],
              "cache_layout": [base.tensor_layout(value) for value in
                               holders["control"].k_cache.kv_cache],
              "cache_storage_offsets": [value.storage_offset() for value in
                                        holders["control"].k_cache.kv_cache]}
    graphs = {}
    for phase in ("eager", "graph"):
        progress.update(width=width, phase=phase)
        for raw in storage.values():
            raw.copy_(original)
        if phase == "graph":
            set_input("A")
            torch.npu.synchronize()
            for arm in ARMS:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    update(arm)
                graphs[arm] = graph
            for raw in storage.values():
                raw.copy_(original)
        expected = backing_bytes()
        for state in ("A", "B", "reset"):
            progress["state"] = state
            if state == "reset":
                expected = backing_bytes()
                for raw in storage.values():
                    raw.copy_(original)
            set_input(state)
            torch.npu.synchronize()
            eager_wall = {}
            for arm in ARMS:
                started = time.perf_counter_ns()
                update(arm) if phase == "eager" else graphs[arm].replay()
                torch.npu.synchronize()
                eager_wall[arm] = (time.perf_counter_ns() - started) / 1000
            expect_update(expected, state)
            result["validation"].append({
                "phase": phase, "state": state,
                "full_backing_sha256": check(expected),
                "wall_us_including_host_method": eager_wall,
            })
    result["quant_output_layout"] = {
        arm: [base.tensor_layout(value) for value in produced[arm]] for arm in ARMS
    }
    set_input("A")
    torch.npu.synchronize()
    progress.update(phase="timing_capture", state="A")
    for arm in ARMS:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            for _ in range(calls):
                update(arm)
        graphs[arm] = graph
    for graph in graphs.values():
        graph.replay()
    torch.npu.synchronize()
    progress["phase"] = "timing"
    for pair in range(pairs):
        order = ARMS if pair % 2 == 0 else ARMS[::-1]
        for arm in order:
            progress.update(arm=arm, pair=pair)
            storage[arm].copy_(original)
            torch.npu.synchronize()
            memory = base.memory_snapshot(device) if pair in (0, pairs - 1) else None
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            begin.record()
            started = time.perf_counter_ns()
            graphs[arm].replay()
            end.record()
            end.synchronize()
            result["samples"].append({
                "pair": pair, "order": order, "arm": arm, "calls": calls,
                "device_us_per_call": begin.elapsed_time(end) * 1000 / calls,
                "wall_us_per_call": (time.perf_counter_ns() - started) / 1000 / calls,
                "memory": memory,
            })
    expected = backing_bytes()
    expect_update(expected, "A")
    result["timing_output_sha256"] = check(expected)
    result["median_device_us"] = {
        arm: statistics.median(row["device_us_per_call"] for row in result["samples"]
                               if row["arm"] == arm) for arm in ARMS
    }
    result["memory_final"] = base.memory_snapshot(device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-npu", action="store_true")
    parser.add_argument("--device", type=int, default=15)
    parser.add_argument("--pairs", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--calls", type=int, choices=range(1, 6), default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Synthetic post-Hadamard BF16 input -> dynamic_quant -> FP16 scale -> actual write_cache methods",
        "npu_launched": False, "cpu_check": cpu_check(),
        "files": [base.file_identity(path) for path in
                  [Path(__file__), ROOT / "index-cache-pair.patch",
                   ROOT / "control/indexer.py", ROOT / "candidate/indexer.py"]],
        "runtime_source": base.source_receipt(not args.run_npu),
        "quota_bytes": base.QUOTA, "quota_scope": "Torch allocator; runtime/driver excluded",
        "persistent_tensor_bytes_max": 3 * STORAGE_BYTES + 32 * 780,
        "pairs": args.pairs, "calls_per_sample": args.calls,
        "max_update_calls": 6 * (7 + (args.pairs + 2) * args.calls),
        "timing_scope": "Each graph contains calls_per_sample complete producer+method updates; one replay/sample. View creation executes inside eager/capture; metadata is replay-free. Eager validation wall includes host view creation. Cache reset excluded identically.",
        "progress": {}, "widths": [],
    }
    result_path = args.output / "result.json"
    try:
        if args.run_npu:
            global torch, torch_npu
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            import torch
            import torch_npu
            from vllm_ascend.utils import enable_custom_op

            assert enable_custom_op()
            base.torch, base.torch_npu = torch, torch_npu
            torch.set_num_threads(1)
            torch.npu.set_device(args.device)
            total = torch.npu.get_device_properties(args.device).total_memory
            torch.npu.set_per_process_memory_fraction(base.QUOTA / total)
            device = torch.device("npu", args.device)
            methods = {arm: load_method(arm) for arm in ARMS}
            report["device"] = args.device
            report["npu_launched"] = True
            report["memory_before"] = base.memory_snapshot(device)
            with torch.inference_mode():
                for width in base.WIDTHS:
                    report["widths"].append(run_width(
                        width, device, methods, args.pairs, args.calls, report["progress"],
                    ))
                    result_path.write_text(json.dumps(report, indent=2) + "\n")
                    gc.collect()
                    torch.npu.empty_cache()
        report["passed"] = True
    except Exception:
        report["passed"] = False
        report["exception"] = traceback.format_exc()
        raise
    finally:
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"result": str(result_path), "passed": report.get("passed"),
                          "npu_launched": report["npu_launched"]}))


if __name__ == "__main__":
    main()
