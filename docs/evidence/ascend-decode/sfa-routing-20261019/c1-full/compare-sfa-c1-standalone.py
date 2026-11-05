"""Replay unchanged arbitrary token indices through V1 and host-routed C1."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch


BLOCK_SIZE = 1
PAGE_SIZE = 128
TOP_K = 2048
ORIGINAL_HELPER_SHA256 = "395d9a62e69f12a18ee62b1574cfaf64e0491a1e726638829ea5bedf832d64b5"
ORIGINAL_WRAPPER_SHA256 = "ea49b79e589f690dc113799e4c9cc4ad0f86b575a356d034a5954187a6939f6f"


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
    if args.ordinary_indices:
        device_blocks = device_case["sparse_indices"]
    else:
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


def make_case():
    case = synthetic_case(8, 8191, 20261019)
    generator = torch.Generator().manual_seed(20261020)
    raw = case['sparse_indices']
    raw.fill_(-1)
    counts = (2048, 2047, 2046, 2045, 1793, 1537, 1025, 513)
    length = int(case['actual_seq_lengths_kv'].item())
    for row, count in enumerate(counts):
        threshold = length - len(counts) + row + 1
        forced = list(dict.fromkeys([127, 128, 8191, threshold - 1]))
        if threshold > 8192:
            forced.append(8192)
        forced = list(dict.fromkeys(forced))
        selected = forced + [x for x in torch.randperm(
            threshold, generator=generator).tolist() if x not in forced]
        selected = selected[:count]
        selected = torch.tensor(selected, dtype=torch.int32)
        selected = selected[torch.randperm(count, generator=generator)]
        raw[row, 0, :count] = selected
        future = list(range(threshold, min(length, threshold + 2)))
        future = future[:raw.shape[-1] - count]
        if future:
            raw[row, 0, count:count + len(future)] = torch.tensor(future)
    return case


def selection_audit(case):
    result = []
    raw = case['sparse_indices']
    length = int(case['actual_seq_lengths_kv'].item())
    rows = case['query'].shape[0]
    for row, values in enumerate(raw[:, 0].tolist()):
        threshold = length - rows + row + 1
        trailing = values.index(-1) if -1 in values else len(values)
        assert all(x == -1 for x in values[trailing:])
        selected = [x for x in values[:trailing] if 0 <= x < threshold]
        assert selected and len(selected) == len(set(selected))
        assert all(x >= 0 for x in values[:trailing])
        blocks = list(dict.fromkeys(x // 4 for x in selected))
        expanded = [x for b in blocks
                    for x in range(4 * b, min(4 * b + 4, threshold))]
        assert selected != expanded
        result.append({
            'row': row, 'causal_threshold': threshold,
            'valid_tokens': len(selected),
            'future_tokens': sum(x >= threshold for x in values[:trailing]),
            'sentinels': len(values) - trailing,
            'four_block_closed': False,
            'raw_indices_sha256': hashlib.sha256(
                raw[row].contiguous().numpy().tobytes()).hexdigest(),
            'token_representation_unchanged': True,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int)
    parser.add_argument('--ordinary-indices', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    case = (torch.load(args.input, map_location='cpu', weights_only=True)
            if args.input else make_case())
    rows = selection_audit(case)
    fixture = args.output / 'input.pt'
    torch.save(case, fixture)
    report = {
        'status': 'cpu_selection_verified',
        'input_kind': 'synthetic_arbitrary_tokens',
        'input_sha256': sha256(args.input or fixture),
        'fixture_sha256': sha256(fixture),
        'source_sha256': sha256(__file__),
        'execution_source_original_sha256': ORIGINAL_HELPER_SHA256,
        'wrapper_source_original_sha256': ORIGINAL_WRAPPER_SHA256,
        'rows': rows, 'sparse_shape': list(case['sparse_indices'].shape),
        'input_tensor_bytes': sum(x.numel() * x.element_size()
                                  for x in case.values()),
        'indices_backing_bytes': (case['sparse_indices'].numel()
                                 + (not args.ordinary_indices)) * 4,
        'indices_storage': ('ordinary' if args.ordinary_indices
                            else 'one_int32_guard'),
        'torch_allocator_limit_bytes': 256 * 2**20,
        'quota_scope': 'Torch allocator; excludes NPU runtime/context',
        'npu_launched': False, 'performance_measured': False,
        'decoder_precision_verified': False,
    }
    if args.device is not None:
        outputs, checks, runtime = execute(
            case, case['sparse_indices'], SimpleNamespace(
                device=args.device, memory_limit_mib=256, block_size=1,
                ordinary_indices=args.ordinary_indices))
        torch.save(outputs, args.output / 'outputs.pt')
        equal = all(x['bit_equal'] and x['finite']
                    for group in (checks['graph_vs_eager'], checks['reset'])
                    for x in group.values())
        report.update(status='device_reproducible' if equal else 'device_diff',
                      npu_launched=True, comparisons=checks, runtime=runtime,
                      outputs_sha256=sha256(args.output / 'outputs.pt'))
    (args.output / 'result.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return int(report['status'] == 'device_diff')


if __name__ == '__main__':
    raise SystemExit(main())

