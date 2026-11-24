"""Compare token-wise SFA with an exact four-token block representation."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


BLOCK_SIZE = 4
PAGE_SIZE = 128
TOP_K = 2048


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_case(rows, context, seed):
    generator = torch.Generator().manual_seed(seed)
    length = context + rows
    pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32)[None]
    indices = torch.full((rows, 1, TOP_K), -1, dtype=torch.int32)
    for row in range(rows):
        threshold = context + row + 1
        full_blocks, tail = divmod(threshold, BLOCK_SIZE)
        count = min(full_blocks, TOP_K // BLOCK_SIZE - bool(tail))
        blocks = torch.randperm(full_blocks, generator=generator)[:count]
        positions = [BLOCK_SIZE * block + offset
                     for block in blocks.tolist()
                     for offset in range(BLOCK_SIZE)]
        positions.extend(range(full_blocks * BLOCK_SIZE, threshold))
        indices[row, 0, :len(positions)] = torch.tensor(positions)
    options = {"generator": generator, "dtype": torch.bfloat16}
    return {
        "query": torch.randn(rows, 4, 512, **options),
        "query_rope": torch.randn(rows, 4, 64, **options),
        "key": torch.randn(pages, PAGE_SIZE, 1, 512, **options),
        "key_rope": torch.randn(pages, PAGE_SIZE, 1, 64, **options),
        "sparse_indices": indices,
        "block_table": table,
        "actual_seq_lengths_query": torch.tensor([rows], dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor([length], dtype=torch.int32),
    }


def captured_case(saved):
    query, query_rope = saved["query_nope"], saved["query_rope"]
    rows, length = query.shape[0], saved["latent"].shape[0]
    pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32)[None]
    case = {"query": query, "query_rope": query_rope,
            "sparse_indices": saved["indices"], "block_table": table,
            "actual_seq_lengths_query": torch.tensor([rows],
                                                     dtype=torch.int32),
            "actual_seq_lengths_kv": torch.tensor([length], dtype=torch.int32)}
    for name, field, dimension in (("key", "latent", 512),
                                   ("key_rope", "rope", 64)):
        logical = torch.zeros(pages * PAGE_SIZE, dimension,
                              dtype=torch.bfloat16)
        logical[:length].copy_(saved[field])
        cache = torch.empty(pages, PAGE_SIZE, 1, dimension,
                            dtype=torch.bfloat16)
        cache[table[0].long()] = logical.view(pages, PAGE_SIZE, 1, dimension)
        case[name] = cache
    return case


def exact_blocks(case):
    indices = case["sparse_indices"]
    rows = case["query"].shape[0]
    if indices.ndim != 3 or tuple(indices.shape[:2]) != (rows, 1):
        raise ValueError("Expected sparse indices with shape [M,1,K]")
    if case["actual_seq_lengths_query"].tolist() != [rows]:
        raise ValueError("This experiment requires one TND sequence")
    length = int(case["actual_seq_lengths_kv"].item())
    capacity = (indices.shape[-1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Pipeline drain can read one entry after the final row's sparse array.
    storage = torch.full((rows * capacity + 1,), -1, dtype=torch.int32)
    packed = storage[:-1].view(rows, 1, capacity)
    report = []
    for row, values in enumerate(indices[:, 0].tolist()):
        threshold = length - rows + row + 1
        selected, invalid_seen = [], False
        for value in values:
            if value == -1:
                invalid_seen = True
            elif value < -1 or invalid_seen:
                raise ValueError(f"Row {row}: invalid indices must trail")
            elif value < threshold:
                selected.append(value)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError(f"Row {row}: empty or repeated selection")
        blocks = list(dict.fromkeys(value // BLOCK_SIZE for value in selected))
        expanded = [position for block in blocks
                    for position in range(block * BLOCK_SIZE,
                                          min((block + 1) * BLOCK_SIZE,
                                              threshold))]
        if expanded != selected:
            raise ValueError(f"Row {row}: C4 expansion changes selection")
        if len(blocks) > capacity:
            raise ValueError(f"Row {row}: packed selection exceeds capacity")
        packed[row, 0, :len(blocks)] = torch.tensor(blocks)
        partial = (threshold % BLOCK_SIZE != 0
                   and blocks[-1] == threshold // BLOCK_SIZE)
        report.append({"row": row, "causal_threshold": threshold,
                       "selected_positions": len(selected),
                       "selected_blocks": len(blocks),
                       "causal_partial_block": partial,
                       "ordered_expansion_equal": True})
    return packed, report


def difference(value, reference):
    delta = value.double() - reference.double()
    return {"bit_equal": torch.equal(value, reference),
            "finite": bool(torch.isfinite(value).all()),
            "different_elements": int((value != reference).sum()),
            "max_abs": delta.abs().max().item(),
            "nrmse": (delta.norm() / reference.double().norm()).item()}


def execute(case, packed, args):
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_device(args.device)
    total = torch_npu.npu.get_device_properties(args.device).total_memory
    torch_npu.npu.set_per_process_memory_fraction(
        args.memory_limit_mib * 2**20 / total)
    if not enable_custom_op():
        raise RuntimeError("The vLLM custom operator is unavailable")
    device = torch.device("npu", args.device)
    device_case = {key: value.to(device) for key, value in case.items()}
    capacity = packed.shape[-1]
    storage = torch.full((packed.numel() + 1,), -1, dtype=torch.int32,
                         device=device)
    device_blocks = storage[:-1].view(-1, 1, capacity)
    device_blocks.copy_(packed)
    torch_npu.npu.synchronize()

    def operation(block_size):
        arguments = dict(device_case)
        if block_size == BLOCK_SIZE:
            arguments["sparse_indices"] = device_blocks
        return torch.ops._C_ascend.npu_sparse_flash_attention(
            **arguments, value=device_case["key"], scale_value=1 / 16,
            sparse_block_size=block_size, layout_query="TND",
            layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
            return_softmax_lse=False,
        )[0]

    outputs, graphs = {}, {}
    sizes = (args.block_size,) if args.block_size else (1, BLOCK_SIZE)
    stream = torch_npu.npu.Stream()
    with torch.inference_mode(), torch_npu.npu.stream(stream):
        for block_size in sizes:
            graph = torch_npu.npu.NPUGraph()
            operation(block_size)
            stream.synchronize()
            with torch_npu.npu.graph(graph, stream=stream):
                output = operation(block_size)
            graphs[block_size] = (graph, output)
        for label, sign in (("a", 1), ("b", -1), ("reset_a", 1)):
            for name in ("query", "query_rope"):
                device_case[name].copy_(case[name].to(device) * sign)
            for block_size, (graph, output) in graphs.items():
                eager = operation(block_size)
                stream.synchronize()
                outputs[f"{label}_eager_{block_size}"] = eager.cpu().clone()
                output.fill_(float("nan"))
                graph.replay()
                stream.synchronize()
                outputs[f"{label}_graph_{block_size}"] = output.cpu().clone()
        for graph, _ in graphs.values():
            graph.reset()
    checks = {
        "graph_vs_eager": {
            f"{label}_{size}": difference(outputs[f"{label}_graph_{size}"],
                                          outputs[f"{label}_eager_{size}"])
            for label in ("a", "b", "reset_a") for size in sizes},
        "c4_vs_v1": {
            label: difference(outputs[f"{label}_eager_4"],
                              outputs[f"{label}_eager_1"])
            for label in ("a", "b", "reset_a")
            if len(sizes) == 2},
        "reset": {
            str(size): difference(outputs[f"reset_a_graph_{size}"],
                                  outputs[f"a_graph_{size}"])
            for size in sizes},
    }
    mappings = Path("/proc/self/maps").read_text().splitlines()
    libraries = sorted({line.split()[-1] for line in mappings
                        if "opmaster" in line or "optiling" in line})
    runtime = {"host_libraries": libraries,
               "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
               "peak_allocated_bytes": torch_npu.npu.max_memory_allocated(),
               "peak_reserved_bytes": torch_npu.npu.max_memory_reserved()}
    return outputs, checks, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--block-size", type=int, choices=(1, BLOCK_SIZE))
    parser.add_argument("--memory-limit-mib", type=int, default=256)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-only", action="store_true")
    mode.add_argument("--device", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.input:
        saved = torch.load(args.input, map_location="cpu", weights_only=True)
        case = captured_case(saved["actual"]) if "actual" in saved else saved
    else:
        case = synthetic_case(args.rows, args.context, args.seed)
    packed, rows = exact_blocks(case)
    fixture = args.output / "input.pt"
    torch.save(case, fixture)
    report = {"status": "cpu_selection_verified", "rows": rows,
              "input_kind": "provided" if args.input else "synthetic",
              "input_sha256": sha256(args.input or fixture),
              "fixture_sha256": sha256(fixture),
              "source_sha256": sha256(__file__),
              "sparse_shape_v1": list(case["sparse_indices"].shape),
              "sparse_shape_c4": list(packed.shape),
              "block_size": args.block_size,
              "input_tensor_bytes": sum(value.numel() * value.element_size()
                                        for value in case.values()),
              "c4_backing_bytes": (packed.numel() + 1) * 4,
              "torch_allocator_limit_bytes": args.memory_limit_mib * 2**20,
              "quota_scope": "Torch allocator; excludes NPU runtime/context",
              "trailing_storage_guard_int32": 1,
              "precision_verified": False, "performance_measured": False,
              "npu_launched": False,
              "npu_initialized": bool(hasattr(torch, "npu")
                                      and torch.npu.is_initialized())}
    if args.device is not None:
        outputs, checks, runtime = execute(case, packed, args)
        torch.save(outputs, args.output / "outputs.pt")
        equal = all(item["bit_equal"] and item["finite"]
                    for group in (checks["graph_vs_eager"], checks["reset"])
                    for item in group.values())
        report.update(status="device_reproducible" if equal else "device_diff",
                      npu_launched=True, npu_initialized=True, runtime=runtime,
                      comparisons=checks,
                      outputs_sha256=sha256(args.output / "outputs.pt"))
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["status"] == "device_diff")


if __name__ == "__main__":
    raise SystemExit(main())
