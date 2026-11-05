"""Check 8K-to-2K native index selection and paged cache state on A3."""

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch

from tilert.models.glm_5_2_ascend.ops.indexer import (
    CANN_ARCH22_SOURCE_DIRECTORY,
    NativeIndexer,
    cann_arch22_index_scores,
    cann_arch22_input_domain,
    cann_arch22_scoring_contract,
)
from tilert.models.glm_5_2_ascend.ops.quantized import quantize_a8_reference


LENGTH = 8192
DIMENSION = 128
HEADS = 32
TOP_K = 2048


def check(checks, name, passed, category="end_to_end", **details):
    checks.append({
        "name": name, "category": category, "passed": bool(passed), **details,
    })


def exact(checks, name, actual, expected, category="end_to_end"):
    actual = actual.detach().cpu()
    metadata = {
        "shape": list(actual.shape), "expected_shape": list(expected.shape),
        "dtype": str(actual.dtype), "expected_dtype": str(expected.dtype),
    }
    compatible = (
        actual.shape == expected.shape and actual.dtype == expected.dtype
    )
    if compatible:
        metadata["unequal"] = int(torch.count_nonzero(actual != expected))
    check(checks, name, compatible and torch.equal(actual, expected),
          category=category, **metadata)


def reference_hadamard():
    """Independent Walsh construction from the parity of row/column bits."""
    indices = torch.arange(DIMENSION, dtype=torch.int32)
    products = indices[:, None] & indices[None, :]
    parity = torch.zeros_like(products)
    for shift in range(7):
        parity ^= (products >> shift) & 1
    return ((1 - 2 * parity).float() / math.sqrt(DIMENSION)).to(torch.bfloat16)


def encode_reference(rows, dtype):
    if dtype == torch.bfloat16:
        return rows.clone(), None
    rotated = (rows.float() @ reference_hadamard().float()).to(torch.bfloat16)
    values, scale = quantize_a8_reference(rotated)
    return values, scale.to(torch.float16)


def scores_reference(query, keys, query_scale, key_scale, weights):
    """Ideal FP32 dequantization, retained as an algorithmic diagnostic."""
    if query.dtype == torch.int8:
        products = query.to(torch.int32) @ keys.to(torch.int32).T
        products = products.float() * query_scale.float()[:, None]
        products = products * key_scale.float()[None, :]
    else:
        products = query.float() @ keys.float().T
    return (products.relu() * weights.float()[:, None]).sum(dim=0)


def source_scores(query, keys, query_scale, key_scale, weights):
    if query.dtype == torch.int8:
        return cann_arch22_index_scores(
            query, keys, query_scale, key_scale, weights
        )
    return scores_reference(query, keys, query_scale, key_scale, weights)


def scoring_domain(checks, name, query_scale, key_scale, weights, category):
    if query_scale is None:
        return
    domain = cann_arch22_input_domain(
        query_scale.tolist(), key_scale.tolist(), weights.tolist()
    )
    check(checks, name, domain["accepted"], category=category, **domain)


def topk_check(checks, name, actual, scores, category="end_to_end"):
    if not torch.isfinite(scores).all():
        check(checks, name, False, category=category,
              nonfinite_scores=int((~torch.isfinite(scores)).sum()))
        return
    actual = actual.detach().cpu().flatten()
    output_dtype = str(actual.dtype)
    integer_output = actual.dtype in (torch.int32, torch.int64)
    actual = actual.to(torch.int64)
    length = scores.numel()
    count = min(TOP_K, length)
    order = torch.argsort(scores, descending=True, stable=True)
    expected = order[:count]
    observed_set, expected_set = set(actual.tolist()), set(expected.tolist())
    valid = integer_output and len(actual) == count
    valid = valid and len(observed_set) == count
    valid = valid and all(0 <= index < length for index in observed_set)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    cutoff = scores[order[count - 1]]
    next_score = scores[order[count]] if count < length else None
    changed = missing + extra
    tie_only = bool(changed) and valid and all(
        scores[index] == cutoff for index in changed
    )
    check(
        checks, name, valid and observed_set == expected_set,
        category=category, dtype=output_dtype, key_length=length,
        count=len(actual), unique=len(observed_set), intersection=len(
            observed_set & expected_set
        ), missing=missing[:32], extra=extra[:32],
        cutoff_score=cutoff.item(),
        next_score=next_score.item() if next_score is not None else None,
        boundary_gap=(cutoff - next_score).item()
        if next_score is not None else None,
        tied_at_cutoff=int(torch.count_nonzero(scores == cutoff)),
        selected_at_cutoff=count - int(torch.count_nonzero(scores > cutoff)),
        tie_only_difference=tie_only,
        tie_rule="equal CPU scores prefer lower index; exact set required",
        expected_sha256=hashlib.sha256(
            expected.sort().values.numpy().tobytes()
        ).hexdigest(),
        actual_sha256=hashlib.sha256(
            actual.sort().values.numpy().tobytes()
        ).hexdigest(),
    )


def cache_contents(cache):
    keys = cache.keys.view(-1, DIMENSION)[:cache.length].cpu().clone()
    scales = None
    if cache.scales is not None:
        scales = cache.scales.view(-1)[:cache.length].cpu().clone()
    return keys, scales


def cache_check(
    checks, name, cache, expected_keys, expected_scales, category="end_to_end"
):
    keys, scales = cache_contents(cache)
    check(checks, name + ".length", cache.length == len(expected_keys),
          category=category, length=cache.length,
          expected_length=len(expected_keys))
    exact(checks, name + ".keys", keys, expected_keys, category)
    if expected_scales is None:
        check(checks, name + ".no_scale", scales is None, category=category)
    else:
        exact(checks, name + ".scales", scales, expected_scales, category)


class CapturedIndexer(NativeIndexer):
    """Snapshot encoder outputs before cache writes and native selection."""

    def encode(self, rows):
        values, scales = super().encode(rows)
        self.last_encoding = (
            values.detach().cpu().clone(),
            scales.detach().cpu().clone() if scales is not None else None,
        )
        return values, scales


def tensor_metadata(value):
    if not torch.is_tensor(value):
        return {"type": type(value).__name__}
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "device": str(value.device),
    }


def capture_arguments(arguments):
    inputs = {
        name: value.detach().cpu().clone()
        for name, value in arguments.items() if torch.is_tensor(value)
    }
    length = int(inputs["actual_seq_lengths_key"][0])
    page_size = inputs["key"].shape[1]
    page_count = math.ceil(length / page_size)
    pages = inputs["block_table"][0, :page_count].to(torch.int64)
    keys = inputs["key"][pages].reshape(-1, DIMENSION)[:length]
    scales = inputs.get("key_dequant_scale")
    if scales is not None:
        scales = scales[pages].reshape(-1)[:length]
    query_scale = inputs.get("query_dequant_scale")
    return {
        "input_metadata": {
            name: tensor_metadata(value) for name, value in arguments.items()
            if torch.is_tensor(value)
        },
        "attributes": {
            name: value for name, value in arguments.items()
            if not torch.is_tensor(value)
        },
        "wire": {
            "query": inputs["query"].reshape(HEADS, DIMENSION),
            "keys": keys,
            "query_scale": query_scale.reshape(-1)
            if query_scale is not None else None,
            "key_scale": scales,
            "weights": inputs["weights"].reshape(HEADS),
            "block_table": inputs["block_table"],
            "actual_seq_lengths_query": inputs["actual_seq_lengths_query"],
            "actual_seq_lengths_key": inputs["actual_seq_lengths_key"],
        },
    }


def select_with_capture(cache, query, weights):
    import torch_npu

    operator = (
        "npu_quant_lightning_indexer" if cache.key_dtype == torch.int8
        else "npu_lightning_indexer"
    )
    original = getattr(torch_npu, operator)
    captured = {"operator": "torch_npu." + operator}

    def invoke(**arguments):
        captured.update(capture_arguments(arguments))
        result = original(**arguments)
        outputs = (result,) if torch.is_tensor(result) else result
        captured["output_metadata"] = [
            tensor_metadata(item) for item in outputs
        ]
        captured["indices"] = outputs[0].detach().cpu().clone()
        return result

    with patch.object(torch_npu, operator, invoke):
        selected = cache.select(query, weights)
    return selected, captured


def append_with_capture(cache, key, encoded_keys, encoded_scales):
    cache.append(key)
    new_keys, new_scales = cache.last_encoding
    encoded_keys = torch.cat((encoded_keys, new_keys))
    if encoded_scales is not None:
        encoded_scales = torch.cat((encoded_scales, new_scales))
    return encoded_keys, encoded_scales


def check_selections(
    cache, query_input, query, weights, expected, label, report
):
    expected_query, expected_keys, query_scale, key_scale = expected
    checks = report["checks"]
    for sign, head_weights in weights.items():
        weight_dtype = (
            torch.float16 if cache.key_dtype == torch.int8 else torch.bfloat16
        )
        rounded_weights = head_weights.to(weight_dtype)
        device_weights = rounded_weights.to(cache.device)
        reference_inputs = (
            expected_query, expected_keys[:cache.length], query_scale,
            key_scale[:cache.length] if key_scale is not None else None,
            rounded_weights,
        )
        expected_scores = source_scores(*reference_inputs)
        ideal_expected_scores = scores_reference(*reference_inputs)
        before_keys, before_scales = cache_contents(cache)
        selected, captured = select_with_capture(
            cache, query_input, device_weights
        )
        name = f"{label}.{sign}"
        scoring_domain(
            checks, name + ".reference_fp16_domain",
            reference_inputs[2], reference_inputs[3], rounded_weights,
            "end_to_end",
        )
        topk_check(checks, name + ".topk", selected, expected_scores)
        wire = captured.pop("wire")
        wire_inputs = (
            wire["query"], wire["keys"], wire["query_scale"],
            wire["key_scale"], wire["weights"],
        )
        wire_scores = source_scores(*wire_inputs)
        ideal_wire_scores = scores_reference(*wire_inputs)
        scoring_domain(
            checks, name + ".wire_fp16_domain", wire["query_scale"],
            wire["key_scale"], wire["weights"], "native_scoring_selection",
        )
        topk_check(checks, name + ".native_topk", selected, wire_scores,
                   category="native_scoring_selection")
        topk_check(report["ideal_comparisons"], name + ".ideal_topk",
                   selected, ideal_expected_scores, category="ideal_encoder")
        topk_check(report["ideal_comparisons"], name + ".ideal_wire_topk",
                   selected, ideal_wire_scores, category="ideal_wire")
        actual_indices = captured.pop("indices")
        artifact = Path(report["artifact_directory"]) / f"{name}.pt"
        torch.save({
            **wire, "cpu_scores": wire_scores,
            "end_to_end_cpu_scores": expected_scores,
            "ideal_cpu_scores": ideal_wire_scores,
            "ideal_end_to_end_cpu_scores": ideal_expected_scores,
            "native_indices": actual_indices,
        }, artifact)
        captured.update({
            "name": name, "length": cache.length, "artifact": str(artifact),
            "artifact_sha256": hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest(),
        })
        report["native_calls"].append(captured)
        cache_check(checks, name + ".cache_preserved", cache,
                    before_keys, before_scales, category="cache")
        exact(checks, name + ".query_preserved", query_input, query,
              category="encoding")
        exact(checks, name + ".weights_preserved", device_weights,
              rounded_weights, category="native_scoring_selection")


def check_dtype(dtype, args, keys, query, weights, report):
    label = str(dtype).removeprefix("torch.")
    checks = report["checks"]
    expected_keys, expected_scales = encode_reference(keys, dtype)
    expected_query, expected_query_scale = encode_reference(query, dtype)
    cache = CapturedIndexer(LENGTH + 1, dtype, args.device)
    key_input, query_input = keys.to(args.device), query.to(args.device)
    encoded, scales = cache.encode(key_input)
    encoded_query, query_scale = cache.encode(query_input)
    exact(checks, label + ".encode.keys", encoded, expected_keys,
          category="encoding")
    exact(checks, label + ".encode.query", encoded_query, expected_query,
          category="encoding")
    if dtype == torch.int8:
        exact(checks, label + ".hadamard", cache.hadamard,
              reference_hadamard(), category="encoding")
        exact(checks, label + ".encode.key_scale", scales, expected_scales,
              category="encoding")
        exact(checks, label + ".encode.query_scale", query_scale,
              expected_query_scale, category="encoding")
    else:
        check(checks, label + ".encode.no_scale",
              scales is None and query_scale is None, category="encoding")
    encoded_before = encoded.cpu().clone()
    scales_before = scales.cpu().clone() if scales is not None else None
    cache.inject(encoded[:LENGTH - 1],
                 scales[:LENGTH - 1] if scales is not None else None)
    cache_check(checks, label + ".inject", cache,
                expected_keys[:LENGTH - 1],
                expected_scales[:LENGTH - 1]
                if expected_scales is not None else None)
    wire_keys = encoded_before[:LENGTH - 1].clone()
    wire_scales = (
        scales_before[:LENGTH - 1].clone()
        if scales_before is not None else None
    )
    cache_check(checks, label + ".cache.inject", cache, wire_keys, wire_scales,
                category="cache")
    wire_keys, wire_scales = append_with_capture(
        cache, key_input[LENGTH - 1], wire_keys, wire_scales
    )
    cache_check(checks, label + ".append", cache, expected_keys[:LENGTH],
                expected_scales[:LENGTH]
                if expected_scales is not None else None)
    cache_check(checks, label + ".cache.append", cache, wire_keys, wire_scales,
                category="cache")
    report["cache_layouts"][label] = {
        "keys": list(cache.keys.shape), "key_dtype": str(cache.keys.dtype),
        "scales": (
            list(cache.scales.shape) if cache.scales is not None else None
        ),
        "scale_dtype": (
            str(cache.scales.dtype) if cache.scales is not None else None
        ),
        "capacity": cache.capacity, "length": cache.length,
    }
    expected = (
        expected_query, expected_keys, expected_query_scale, expected_scales
    )
    check_selections(
        cache, query_input, query, weights, expected, label, report
    )
    wire_keys, wire_scales = append_with_capture(
        cache, key_input[LENGTH], wire_keys, wire_scales
    )
    cache_check(checks, label + ".cross_page_append", cache,
                expected_keys, expected_scales)
    cache_check(checks, label + ".cache.cross_page_append", cache,
                wire_keys, wire_scales, category="cache")
    check_selections(cache, query_input, query, weights, expected,
                     f"{label}.length{LENGTH + 1}", report)
    try:
        cache.append(key_input[LENGTH])
    except ValueError:
        check(checks, label + ".capacity_guard", True, category="cache")
    else:
        check(checks, label + ".capacity_guard", False, category="cache")
    cache.reset()
    check(checks, label + ".reset.length", cache.length == 0, category="cache")
    reset_weights = weights["negative"].to(args.device)
    try:
        cache.select(query_input, reset_weights)
    except ValueError:
        check(checks, label + ".reset.empty_select", True, category="cache")
    else:
        check(checks, label + ".reset.empty_select", False, category="cache")
    cache.append(key_input[-1])
    cache_check(checks, label + ".reset.append", cache, expected_keys[-1:],
                expected_scales[-1:] if expected_scales is not None else None)
    cache_check(checks, label + ".cache.reset.append", cache,
                *cache.last_encoding, category="cache")
    selected = cache.select(query_input, reset_weights).cpu().to(torch.int64)
    exact(checks, label + ".reset.new_prefix", selected, torch.tensor([0]),
          category="cache")
    exact(checks, label + ".key_input_preserved", key_input, keys,
          category="encoding")
    exact(checks, label + ".encoded_input_preserved", encoded, encoded_before,
          category="cache")
    if scales is not None:
        exact(checks, label + ".scale_input_preserved", scales, scales_before,
              category="cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts = args.output.parent / (args.output.stem + "-tensors")
    artifacts.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "hostname": platform.node(), "device": args.device, "seed": args.seed,
        "command": [sys.executable, *sys.argv], "git_head": args.source_commit,
        "shape": {"query": [HEADS, DIMENSION], "keys": [LENGTH, DIMENSION],
                  "key_lengths": [LENGTH, LENGTH + 1],
                  "top_k": TOP_K, "page_size": 128},
        "scope": "synthetic post-RoPE encoding, cache operations, selection",
        "reference": (
            "CPU encoder; C8 arch22 FP16 intermediates and FP32 reduction; "
            "BF16 index scores retain the FP32 formula"
        ),
        "selection_rule": "exact sets; equal CPU scores prefer lower index",
        "head_weights": "positive/negative scaled by 1/64; zero cutoff ties",
        "categories": {
            "encoding": "native encoder against independent CPU encoder",
            "cache": "cache writes against captured native encoder outputs",
            "native_scoring_selection": (
                "source-bound CPU scores from actual CANN inputs"
            ),
            "end_to_end": "CPU encoder, source-bound scores, exact top-k",
        },
        "ideal_comparison_scope": (
            "diagnostic exact-set comparison; ideal FP32 scores omit proven "
            "arch22 FP16 boundaries and do not define CANN acceptance"
        ),
        "ideal_comparisons": [],
        "artifact_directory": str(artifacts), "native_calls": [],
        "versions": {"python": platform.python_version(),
                     "torch": torch.__version__},
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (Path(__file__),
                         root / "tilert/models/glm_5_2_ascend/ops/indexer.py",
                         root / "tilert/models/glm_5_2_ascend/ops/quantized.py"
                         )
        },
        "cache_layouts": {}, "checks": [],
    }
    try:
        import torch_npu

        report["c8_scoring_contract"] = cann_arch22_scoring_contract(
            CANN_ARCH22_SOURCE_DIRECTORY
        )
        torch.set_num_threads(4)
        torch.npu.set_device(args.device)
        report["versions"]["torch_npu"] = torch_npu.__version__
        generator = torch.Generator().manual_seed(args.seed)
        keys = torch.randn(
            LENGTH + 1, DIMENSION, generator=generator
        ).to(torch.bfloat16)
        query = torch.randn(
            HEADS, DIMENSION, generator=generator
        ).to(torch.bfloat16)
        positive = (torch.rand(HEADS, generator=generator) + 0.5) / 64
        weights = {
            "positive": positive, "negative": -positive,
            "zero_ties": torch.zeros_like(positive),
        }
        with torch.inference_mode():
            for dtype in (torch.int8, torch.bfloat16):
                check_dtype(dtype, args, keys, query, weights, report)
        report["status"] = (
            "pass" if all(item["passed"] for item in report["checks"])
            else "fail"
        )
    except Exception:
        report["error"] = traceback.format_exc()
    report["checks_total"] = len(report["checks"])
    report["checks_failed"] = sum(
        not item["passed"] for item in report["checks"]
    )
    report["category_results"] = {}
    for category in report["categories"]:
        checks = [item for item in report["checks"]
                  if item["category"] == category]
        failed = sum(not item["passed"] for item in checks)
        status = ("fail" if failed else "pass") if checks else "not_run"
        report["category_results"][category] = {
            "status": status, "checks_total": len(checks),
            "checks_failed": failed,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(json.dumps({name: report[name] for name in (
        "status", "checks_total", "checks_failed"
    )}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
