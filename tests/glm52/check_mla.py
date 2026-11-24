"""Check rank-local CANN compressed MLA, BF16 boundaries, and paged state."""

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as functional

from tilert.models.glm_5_2_ascend.ops import mla


HEADS = 4
LATENT_DIM = 512
NOPE_DIM = 192
ROPE_DIM = 64
VALUE_DIM = 256
TOP_K = 2048
LENGTH = 8192


def record(checks, name, passed, category, **details):
    checks.append({
        "name": name, "passed": bool(passed), "category": category, **details,
    })


def metrics(actual, expected):
    left, right = actual.float(), expected.float()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {"finite": False}
    error = (left - right).norm().item()
    norm = right.norm().item()
    cosine = (
        functional.cosine_similarity(
            left.flatten(), right.flatten(), dim=0
        ).item() if norm else float(torch.count_nonzero(left).item() == 0)
    )
    return {
        "finite": True, "max_abs": (left - right).abs().max().item(),
        "nrmse": error / max(norm, 1e-12), "cosine": cosine,
        "unequal": int(torch.count_nonzero(actual != expected)),
    }


def compare(checks, name, actual, expected, category, exact=False):
    actual = actual.detach().cpu()
    details = {
        "shape": list(actual.shape), "expected_shape": list(expected.shape),
        "dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
        "mode": "exact" if exact else "nrmse_and_cosine",
    }
    compatible = (
        actual.shape == expected.shape and actual.dtype == expected.dtype
    )
    passed = False
    if compatible and exact:
        details["unequal"] = int(torch.count_nonzero(actual != expected))
        passed = torch.equal(actual, expected)
    elif compatible:
        details.update(metrics(actual, expected))
        passed = (
            details["finite"] and details["nrmse"] <= 0.01
            and details["cosine"] >= 0.999
        )
    record(checks, name, passed, category, **details)


def split_weight(weight):
    heads = weight.float().reshape(HEADS, NOPE_DIM + VALUE_DIM, LATENT_DIM)
    return heads[:, :NOPE_DIM], heads[:, NOPE_DIM:]


def query_reference(q_nope, weight):
    key_weight, _ = split_weight(weight)
    return torch.einsum("thd,hdl->thl", q_nope.float(), key_weight)


def attention_reference(query, q_rope, latent, rope, indices, scale):
    """Independent FP32 scores, softmax, and latent weighted sum on CPU."""
    selected = indices.to(torch.int64)
    keys, key_rope = latent[selected].float(), rope[selected].float()
    scores = query[0].float() @ keys.T
    scores = scores + q_rope[0].float() @ key_rope.T
    probabilities = torch.softmax(scores * scale, dim=-1)
    return (probabilities @ keys).unsqueeze(0)


def value_reference(latent_output, weight):
    _, value_weight = split_weight(weight)
    return torch.einsum(
        "thl,hvl->thv", latent_output.float(), value_weight
    ).reshape(1, HEADS * VALUE_DIM)


def compressed_reference(
    weight, q_nope, q_rope, latent, rope, indices, scale, bf16_boundaries
):
    query = query_reference(q_nope, weight)
    if bf16_boundaries:
        query = query.to(torch.bfloat16)
    context = attention_reference(query, q_rope, latent, rope, indices, scale)
    if bf16_boundaries:
        context = context.to(torch.bfloat16)
    output = value_reference(context, weight)
    if bf16_boundaries:
        output = output.to(torch.bfloat16)
    return {"projected_query": query, "latent_output": context,
            "output": output}


def expanded_reference(weight, q_nope, q_rope, latent, rope, indices, scale):
    """HF-style BF16 KV expansion, QK, scale, probabilities, and PV."""
    selected = indices.to(torch.int64)
    selected_latent, selected_rope = latent[selected], rope[selected]
    heads = weight.float().reshape(HEADS, NOPE_DIM + VALUE_DIM, LATENT_DIM)
    expanded = torch.einsum(
        "sl,hol->sho", selected_latent.float(), heads
    ).to(torch.bfloat16)
    key_nope, value = expanded.split((NOPE_DIM, VALUE_DIM), dim=-1)
    query = torch.cat((q_nope[0], q_rope[0]), dim=-1).float()
    key = torch.cat((
        key_nope, selected_rope[:, None, :].expand(-1, HEADS, -1)
    ), dim=-1).float()
    scores = torch.einsum("hd,shd->hs", query, key).to(torch.bfloat16)
    scores = (scores.float() * scale).to(torch.bfloat16)
    probability = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
    return torch.einsum(
        "hs,shv->hv", probability.float(), value.float()
    ).to(torch.bfloat16).reshape(1, HEADS * VALUE_DIM)


def tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "device": str(tensor.device), "stride": list(tensor.stride()),
    }


def snapshot_cache(cache):
    return {
        "latent": cache.latent_cache.cpu().clone(),
        "rope": cache.rope_cache.cpu().clone(),
        "length": cache.key_length.cpu().clone(),
        "block_table": cache.block_table.cpu().clone(),
    }


def check_cache(cache, latent, rope, name, checks):
    record(checks, name + ".length", cache.length == len(latent), "cache",
           actual=cache.length, expected=len(latent))
    compare(checks, name + ".length_tensor", cache.key_length,
            torch.tensor([len(latent)], dtype=torch.int32), "cache", True)
    compare(checks, name + ".latent",
            cache.latent_cache.view(-1, LATENT_DIM)[:len(latent)],
            latent, "cache", True)
    compare(checks, name + ".rope",
            cache.rope_cache.view(-1, ROPE_DIM)[:len(rope)],
            rope, "cache", True)


def write_cache(cache, latent, rope, name, checks, append=False):
    device_latent, device_rope = latent.to(cache.device), rope.to(cache.device)
    if append:
        cache.append(device_latent, device_rope)
    else:
        cache.inject(device_latent, device_rope)
    compare(checks, name + ".input_latent", device_latent, latent,
            "input_preservation", True)
    compare(checks, name + ".input_rope", device_rope, rope,
            "input_preservation", True)


def selected_indices(length, generator):
    count = min(length, TOP_K)
    required = [position for position in (0, 127, 128, length - 1)
                if 0 <= position < length]
    required = list(dict.fromkeys(required))
    remaining = [position for position in torch.randperm(
        length, generator=generator
    ).tolist() if position not in required]
    values = torch.tensor(
        required + remaining[:count - len(required)], dtype=torch.int64
    )
    backing = torch.full((2 * count,), -77, dtype=torch.int64)
    backing[::2] = values
    return backing[::2]


def trace_with_capture(cache, query, q_rope, indices):
    original = mla._sparse_attention_operator()
    captured = {}

    def invoke(**arguments):
        captured["inputs"] = {
            name: tensor_metadata(value) for name, value in arguments.items()
            if torch.is_tensor(value)
        }
        captured["attributes"] = {
            name: value for name, value in arguments.items()
            if not torch.is_tensor(value)
        }
        captured["sparse_indices"] = arguments["sparse_indices"].cpu().clone()
        captured["query_length"] = (
            arguments["actual_seq_lengths_query"].cpu().clone()
        )
        captured["key_length"] = (
            arguments["actual_seq_lengths_kv"].cpu().clone()
        )
        captured["key_value_alias"] = (
            arguments["key"].data_ptr() == arguments["value"].data_ptr()
        )
        output = original(**arguments)
        captured["outputs"] = [tensor_metadata(item) for item in output]
        return output

    with patch.object(mla, "_sparse_attention_operator", return_value=invoke):
        trace = cache.trace(query, q_rope, indices)
    return trace, captured


def check_case(cache, weight, latent, rope, query, q_rope, indices, name,
               args, report):
    checks = report["checks"]
    check_cache(cache, latent, rope, name + ".before", checks)
    before = snapshot_cache(cache)
    device_query, device_rope = query.to(args.device), q_rope.to(args.device)
    index_storage = torch.full(
        (2 * len(indices),), -77, dtype=indices.dtype, device=args.device
    )
    device_indices = index_storage[::2]
    device_indices.copy_(indices.to(args.device))
    trace, captured = trace_with_capture(
        cache, device_query, device_rope, device_indices
    )
    actual = {
        field.name: getattr(trace, field.name).detach().cpu().clone()
        for field in fields(trace)
    }
    boundary = compressed_reference(
        weight, query, q_rope, latent, rope, indices, cache.scale, True
    )
    fp32 = compressed_reference(
        weight, query, q_rope, latent, rope, indices, cache.scale, False
    )
    expanded = expanded_reference(
        weight, query, q_rope, latent, rope, indices, cache.scale
    )
    for stage, expected in boundary.items():
        compare(checks, name + ".chain." + stage, actual[stage], expected,
                "compressed_bf16")
    local_attention = attention_reference(
        actual["projected_query"], q_rope, latent, rope, indices, cache.scale
    ).to(torch.bfloat16)
    compare(checks, name + ".local.sparse_attention", actual["latent_output"],
            local_attention, "native_stages")
    local_output = value_reference(
        actual["latent_output"], weight
    ).to(torch.bfloat16)
    compare(checks, name + ".local.value_projection", actual["output"],
            local_output, "native_stages")
    compare(checks, name + ".hf_expanded", actual["output"], expanded,
            "hf_expanded")
    for label, device_value, expected in (
        ("query", device_query, query), ("query_rope", device_rope, q_rope),
        ("indices", device_indices, indices),
    ):
        compare(checks, name + ".input." + label, device_value, expected,
                "input_preservation", True)
    for label, value in snapshot_cache(cache).items():
        compare(checks, name + ".unchanged." + label, value, before[label],
                "cache", True)
    check_cache(cache, latent, rope, name + ".after", checks)
    expected_indices = torch.full((1, 1, TOP_K), -1, dtype=torch.int32)
    expected_indices[0, 0, :len(indices)] = indices.to(torch.int32)
    compare(checks, name + ".wire.indices", captured["sparse_indices"],
            expected_indices, "operator_contract", True)
    compare(checks, name + ".wire.query_length", captured["query_length"],
            torch.ones(1, dtype=torch.int32), "operator_contract", True)
    compare(checks, name + ".wire.key_length", captured["key_length"],
            torch.tensor([len(latent)], dtype=torch.int32),
            "operator_contract", True)
    record(checks, name + ".wire.key_value_alias",
           captured["key_value_alias"], "operator_contract")
    artifact = args.output.parent / (args.output.stem + "-" + name + ".pt")
    torch.save({
        "actual": actual, "cpu_bf16_boundaries": boundary,
        "cpu_fp32": fp32, "cpu_hf_expanded": expanded,
        "query": query, "query_rope": q_rope, "indices": indices,
        "sparse_indices": captured.pop("sparse_indices"),
        "query_length": captured.pop("query_length"),
        "key_length": captured.pop("key_length"),
    }, artifact)
    report["cases"].append({
        "name": name, "cache_length": len(latent), "selected": len(indices),
        "index_input": tensor_metadata(device_indices),
        "native": captured,
        "stages": {name: tensor_metadata(getattr(trace, name))
                   for name in actual},
        "fp32_output_difference": metrics(actual["output"], fp32["output"]),
        "compressed_vs_hf_rounding": metrics(boundary["output"], expanded),
        "artifact": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    })


def load_weight(args, generator):
    if args.checkpoint is None:
        return (torch.randn(
            HEADS * (NOPE_DIM + VALUE_DIM), LATENT_DIM, generator=generator
        ) / math.sqrt(LATENT_DIM)).to(torch.bfloat16)
    from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint

    source = ModelSlimCheckpoint(args.checkpoint)
    name = f"model.layers.{args.layer}.self_attn.kv_b_proj.weight"
    rows = HEADS * (NOPE_DIM + VALUE_DIM)
    selection = (slice(args.rank * rows, (args.rank + 1) * rows), slice(None))
    return source.tensor(name, selection)


def run(args, report):
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    report["versions"]["torch_npu"] = torch_npu.__version__
    generator = torch.Generator().manual_seed(args.seed)
    weight = load_weight(args, generator)
    latent = torch.randn(LENGTH + 1, LATENT_DIM, generator=generator)
    rope = torch.randn(LENGTH + 1, ROPE_DIM, generator=generator)
    latent, rope = (latent / 2).to(torch.bfloat16), rope.to(torch.bfloat16)
    query = (torch.randn(
        1, HEADS, NOPE_DIM, generator=generator
    ) / 2).to(torch.bfloat16)
    q_rope = torch.randn(1, HEADS, ROPE_DIM, generator=generator).to(
        torch.bfloat16
    )
    corpus = args.output.parent / (args.output.stem + "-inputs.pt")
    torch.save({"weight": weight, "latent": latent, "rope": rope,
                "query": query, "query_rope": q_rope}, corpus)
    report["input_artifact"] = {
        "path": str(corpus),
        "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
    }
    device_weight = weight.to(args.device)
    cache = mla.NativeMla(device_weight, capacity=LENGTH + 1,
                          device=args.device, scale=args.scale)
    cache.latent_cache.fill_(23)
    cache.rope_cache.fill_(-19)
    report["storage"] = {
        name: tensor_metadata(getattr(cache, name)) for name in (
            "key_weight", "value_weight", "latent_cache", "rope_cache",
            "block_table", "query_length", "key_length",
        )
    }
    key_weight, value_weight = split_weight(weight)
    compare(report["checks"], "weight.key_layout", cache.key_weight,
            key_weight.to(torch.bfloat16), "operator_contract", True)
    compare(report["checks"], "weight.value_layout", cache.value_weight,
            value_weight.transpose(1, 2).contiguous().to(torch.bfloat16),
            "operator_contract", True)
    for length in (1, 8, 127, 128, 129, LENGTH, LENGTH + 1):
        if length in (128, 129, LENGTH + 1):
            write_cache(cache, latent[length - 1:length],
                        rope[length - 1:length], f"append-{length}",
                        report["checks"], append=True)
        else:
            write_cache(cache, latent[:length], rope[:length],
                        f"inject-{length}", report["checks"])
        indices = selected_indices(length, generator)
        check_case(cache, weight, latent[:length], rope[:length], query,
                   q_rope, indices, f"mixed-{length}", args, report)
    for label, sign in (("positive", 1), ("negative", -1)):
        signed_latent, signed_rope = (
            latent[:LENGTH].abs() * sign, rope[:LENGTH].abs() * sign
        )
        write_cache(cache, signed_latent, signed_rope, "inject-" + label,
                    report["checks"])
        indices = selected_indices(LENGTH, generator)
        check_case(cache, weight, signed_latent, signed_rope, query.abs(),
                   q_rope.abs(), indices, label + "-8192", args, report)
    cache.reset()
    record(report["checks"], "reset.empty", cache.length == 0, "cache")
    compare(report["checks"], "reset.length_tensor", cache.key_length,
            torch.zeros(1, dtype=torch.int32), "cache", True)
    try:
        cache.attend(query.to(args.device), q_rope.to(args.device),
                     torch.tensor([0], dtype=torch.int32))
    except ValueError:
        rejected_empty = True
    else:
        rejected_empty = False
    record(report["checks"], "reset.reject_empty", rejected_empty, "cache")
    write_cache(cache, latent[-1:], rope[-1:], "append-after-reset",
                report["checks"], append=True)
    check_case(cache, weight, latent[-1:], rope[-1:], query, q_rope,
               torch.tensor([0], dtype=torch.int32), "reset-refill", args,
               report)
    try:
        cache.attend(query.to(args.device), q_rope.to(args.device),
                     torch.tensor([-1], dtype=torch.int32))
    except ValueError:
        rejected_negative = True
    else:
        rejected_negative = False
    record(report["checks"], "indices.reject_negative", rejected_negative,
           "operator_contract")
    compare(report["checks"], "weight.input_preserved", device_weight,
            weight, "input_preservation", True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, choices=range(16), default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--scale", type=float, default=1 / 16)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), Path(mla.__file__)]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "error", "checks": [], "cases": [],
        "device": args.device, "seed": args.seed, "scale": args.scale,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "layer": args.layer, "rank": args.rank, "tp_size": 16,
        "weight_source": "checkpoint" if args.checkpoint else "synthetic",
        "hostname": platform.node(), "git_head": args.source_commit,
        "command": [sys.executable, *sys.argv],
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest() for path in sources
        },
        "versions": {"torch": torch.__version__, "python": sys.version},
        "scope": "M1, four heads, rank-local compressed MLA; no TP reduction",
        "operators": ["torch_npu.npu_transpose_batchmatmul",
                      "torch.ops._C_ascend.npu_sparse_flash_attention"],
        "reference": (
            "Independent CPU FP32 arithmetic. Compressed BF16 checks round "
            "query projection, latent attention output, and value projection; "
            "the fused SFA interior uses FP32 softmax in the reference. "
            "HF checks also round expanded KV, QK, scale, probability and PV."
        ),
        "acceptance": {
            "numerical": "NRMSE <= 0.01 and cosine >= 0.999",
            "hf_expanded": "same gate; differences remain failures",
            "input_cache_indices": "exact dtype, shape, and values",
            "fp32_diagnostics": "reported separately; never override gates",
        },
    }
    try:
        with torch.inference_mode():
            run(args, report)
        report["status"] = (
            "pass" if all(check["passed"] for check in report["checks"])
            else "fail"
        )
    except Exception:
        report["error"] = traceback.format_exc()
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(
        not item["passed"] for item in report["checks"]
    )
    report["category_results"] = {}
    for category in sorted({item["category"] for item in report["checks"]}):
        checks = [item for item in report["checks"]
                  if item["category"] == category]
        failed = sum(not item["passed"] for item in checks)
        report["category_results"][category] = {
            "status": "fail" if failed else "pass", "checks": len(checks),
            "failed": failed,
        }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(json.dumps({name: report[name] for name in (
        "status", "checks_total", "checks_failed"
    )}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
