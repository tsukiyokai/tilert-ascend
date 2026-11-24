"""Replay CANN SFA, gathered FIA and split-KV on frozen 8K MLA inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def prepare(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)["actual"]
    query, query_rope = saved["query_nope"], saved["query_rope"]
    width = query.shape[0]
    assert width in (1, 4) and query.shape == (width, 4, 512)
    assert query_rope.shape == (width, 4, 64)
    indices = saved["indices"]
    assert indices.shape == (width, 1, 2048)
    assert indices.dtype == torch.int32
    for row in range(width):
        assert indices[row].min() >= 0
        assert indices[row].max() <= 8192 + row
    length, pages = 8192 + width, 65
    table = torch.arange(pages - 1, -1, -1, dtype=torch.int32)[None]
    case = {"query": query, "query_rope": query_rope,
            "sparse_indices": indices, "block_table": table,
            "actual_seq_lengths_query": torch.tensor([width],
                                                     dtype=torch.int32),
            "actual_seq_lengths_kv": torch.tensor([length],
                                                  dtype=torch.int32)}
    for name, source, dim in (("key", "latent", 512),
                              ("key_rope", "rope", 64)):
        assert saved[source].shape == (length, dim)
        assert saved[source].dtype == torch.bfloat16
        logical = torch.zeros(pages * 128, dim, dtype=torch.bfloat16)
        logical[:length].copy_(saved[source])
        paged = torch.empty(pages, 128, 1, dim, dtype=torch.bfloat16)
        paged[table[0].long()] = logical.view(pages, 128, 1, dim)
        case[name] = paged
    assert all(value.dtype == torch.bfloat16 for key, value in case.items()
               if key in ("query", "query_rope", "key", "key_rope"))
    return case


def sfa(case, *, statistics=False):
    return torch.ops._C_ascend.npu_sparse_flash_attention(
        **case, value=case["key"], scale_value=1 / 16,
        sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2, return_softmax_lse=statistics,
    )


def gathered_fia(case):
    import torch_npu

    width = case["query"].shape[0]
    logical = case["sparse_indices"].reshape(-1).long()
    pages = torch.div(logical, 128, rounding_mode="floor")
    physical = case["block_table"][0].long().index_select(0, pages)
    physical = physical * 128 + logical.remainder(128)
    key = case["key"].view(-1, 512).index_select(0, physical)
    key_rope = case["key_rope"].view(-1, 64).index_select(0, physical)
    key = key.view(width, 1, 2048, 512)
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        case["query"].view(width, 4, 1, 512), key, key,
        query_rope=case["query_rope"].view(width, 4, 1, 64),
        key_rope=key_rope.view(width, 1, 2048, 64),
        num_query_heads=4, num_key_value_heads=1, softmax_scale=1 / 16,
        input_layout="BNSD", sparse_mode=0, block_table=None, block_size=0,
        actual_seq_qlen=[1] * width, actual_seq_kvlen=[2048] * width,
    )
    return output.view(width, 4, 512)


def partitioned_case(case):
    width = case["query"].shape[0]
    split = dict(case)
    split.update(
        query=case["query"].repeat(4, 1, 1),
        query_rope=case["query_rope"].repeat(4, 1, 1),
        sparse_indices=case["sparse_indices"].reshape(
            width, 4, 512).permute(1, 0, 2).contiguous().reshape(
                4 * width, 1, 512),
        block_table=case["block_table"].repeat(4 * width, 1),
        actual_seq_lengths_query=torch.arange(
            1, 4 * width + 1, dtype=torch.int32),
        actual_seq_lengths_kv=torch.arange(
            8193, 8193 + width, dtype=torch.int32).repeat(4),
    )
    return split


def split_kv(case):
    output, maximum, total = sfa(case, statistics=True)
    width = case["query"].shape[0] // 4
    partials = output.reshape(4, width * 4, 512).float()
    lse = maximum.reshape(4, width * 4) + total.reshape(
        4, width * 4).log()
    merged, _ = torch.ops.npu.npu_attention_update(
        list(lse.unbind(0)), list(partials.unbind(0)), 0)
    return merged.reshape(width, 4, 512).bfloat16()


def difference(value, reference):
    assert value.shape == reference.shape
    assert value.dtype == reference.dtype == torch.bfloat16
    assert torch.isfinite(value).all() and torch.isfinite(reference).all()
    delta = value.double() - reference.double()
    return {
        "bit_equal": torch.equal(value.view(torch.int16),
                                 reference.view(torch.int16)),
        "different_elements": int((value != reference).sum()),
        "max_abs": delta.abs().max().item(),
        "nrmse": (delta.norm() / reference.double().norm()).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    case = prepare(args.input)
    split = partitioned_case(case)

    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    torch_npu.npu.set_device(args.device)
    total = torch_npu.npu.get_device_properties(args.device).total_memory
    torch_npu.npu.set_per_process_memory_fraction(1.5 * 2**30 / total)
    torch.npu.config.allow_internal_format = True
    assert enable_custom_op()
    device = torch.device("npu", args.device)
    case = {key: value.to(device) for key, value in case.items()}
    split = {key: (case[key] if key in ("key", "key_rope")
                   else value.to(device)) for key, value in split.items()}
    outputs = {}
    with torch.inference_mode():
        for name, operation, inputs in (
            ("sfa", lambda values: sfa(values)[0], case),
            ("gathered_fia", gathered_fia, case),
            ("split_kv", split_kv, split),
        ):
            for iteration in range(2):
                value = operation(inputs)
                torch_npu.npu.synchronize()
                outputs[f"{name}_{iteration}"] = value.cpu().clone()
    report = {
        "scope": "Frozen-input attention diagnostic; cross-kernel differences",
        "precision_verified": False, "performance_measured": False,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "query_count": case["query"].shape[0],
        "comparisons": {name: difference(outputs[f"{name}_0"],
                                         outputs["sfa_0"])
                        for name in ("gathered_fia", "split_kv")},
        "repeats": {name: difference(outputs[f"{name}_1"],
                                     outputs[f"{name}_0"])
                    for name in ("sfa", "gathered_fia", "split_kv")},
        "peak_reserved_bytes": torch_npu.npu.max_memory_reserved(),
    }
    assert all(value["bit_equal"] for value in report["repeats"].values())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(outputs, args.output / "outputs.pt")
    report["outputs_sha256"] = hashlib.sha256(
        (args.output / "outputs.pt").read_bytes()).hexdigest()
    (args.output / "result.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
