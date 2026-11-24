"""Capture M1 TP16 CANN decode from a full-P prefix and check exact reset.

Run with torchrun --nproc-per-node=16. A capture case supplies the original
prompt, main KV with P rows, and the prefill first token. No prompt replay or
MTP model is used. An optional matching HF trace adds cross-configuration
diagnostics. Execution checks and exact reset determine capture status.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import traceback
from unittest.mock import patch

from prefix_io import prefix_prompt_identity, prompt_identity
from run_native_decode import compare_tensor
from run_native_decode import (
    validate_reference_manifest as validate_hf_manifest,
)


def validate_reference_manifest(reference, metadata, *, fp32_head=False):
    validate_hf_manifest(reference, metadata, fp32_head=fp32_head)
    if (reference.get("layer_count") != 78
            or reference.get("cpu_matmul", {}).get("mkldnn_enabled")
            is not False):
        raise ValueError(
            "HF diagnostics require 78 layers and oneDNN disabled"
        )
    checks = reference.get("checks", [])
    if not checks or any(item.get("passed") is not True for item in checks):
        raise ValueError("HF generation and exact reset checks must complete")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--capture-case", type=Path)
    inputs.add_argument("--tokens", nargs="+", type=int,
                        help="Cold-cache teacher tokens, then greedy decode")
    parser.add_argument("--steps", type=int, default=2,
                        help="Total M1 forwards, including teacher tokens")
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--device-offset", type=int, choices=range(16),
                        default=0, help="Physical device=(rank+offset)%%16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--init-method", default="env://")
    parser.add_argument("--reference", type=Path,
                        help="Matching independent HF trace directory")
    parser.add_argument("--layer-trace", action="store_true",
                        help="Capture 79 hidden states and 78 layer logits")
    parser.add_argument("--tp-reduction", default="default",
                        choices=("default", "bfloat16", "float32"),
                        help="Copy BF16 TP inputs and select sum dtype")
    parser.add_argument("--trace-mlp-layer", type=int, choices=range(3, 78),
                        help="Capture one MoE layer's routed/shared stages")
    args = parser.parse_args()
    if args.steps < 1 or args.capacity < 128:
        parser.error("Positive steps and capacity >= 128 are required")
    if args.tokens and (
        len(args.tokens) > args.steps
        or any(not 0 <= token < 154880 for token in args.tokens)
    ):
        parser.error("Cold teacher tokens must fit steps and vocabulary")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.trace_mlp_layer is not None and not args.layer_trace:
        parser.error("trace-mlp-layer requires layer-trace")
    return args


def load_capture(case, rank, checkpoint_hashes):
    import torch

    manifest = json.loads((case / "manifest.json").read_text())
    if (manifest.get("status") != "capture_complete"
            or manifest.get("reset_complete") is not True):
        raise ValueError("Capture case is not complete")
    identity = prompt_identity(manifest["prompt_token_ids"])
    if (identity["prompt_length"] != manifest["seq_len"]
            or identity["prompt_sha256"] != manifest["prompt_sha256"]):
        raise ValueError("Captured prompt identity differs")
    first = manifest["first_token_id"]
    if type(first) is not int or not 0 <= first < 154880:
        raise ValueError("Capture lacks the prefill first token")
    if manifest.get("delivered_tokens") != [first]:
        raise ValueError("Capture did not deliver exactly the first token")
    receipts = manifest["ranks"]
    if (len(receipts) != 16
            or {entry["rank"] for entry in receipts} != set(range(16))):
        raise ValueError("Capture needs all 16 rank receipts")
    for receipt in receipts:
        if (receipt["status"] != "saved"
                or receipt["seq_len"] != manifest["seq_len"]
                or receipt["mtp_seq_len"] != manifest["seq_len"] - 1
                or receipt["request_id"] != manifest["request_id"]
                or receipt["generation"] != manifest["generation"]
                or receipt["prompt_sha256"] != identity["prompt_sha256"]
                or receipt["checkpoint_sha256"] != checkpoint_hashes):
            raise ValueError("Rank receipts describe different captures")
    receipt = next(item for item in receipts if item["rank"] == rank)
    path = case / "main-prefix" / f"rank-{rank}.pt"
    file_hash = digest(path)
    if file_hash != receipt["files"]["main"]["sha256"]:
        raise ValueError("Full-P main prefix file hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["metadata_sha256"] != checkpoint_hashes
            or prefix_prompt_identity(saved, manifest["seq_len"]) != identity
            or saved["prompt_token_ids"] != manifest["prompt_token_ids"]):
        raise ValueError("Prefix payload identity differs from capture")
    return saved["layers"], [first], {
        **identity, "initial_position": manifest["seq_len"],
        "first_token_id": first, "request_id": manifest["request_id"],
        "generation": manifest["generation"], "prefix_file": str(path),
        "prefix_sha256": file_hash,
        "capture_manifest_sha256": digest(case / "manifest.json"),
    }


def load_reference(args, metadata, inputs, rank):
    if args.reference is None:
        return None
    manifest = json.loads((args.reference / "manifest.json").read_text())
    validate_reference_manifest(manifest, metadata)
    if manifest["initial_position"] != inputs["initial_position"]:
        raise ValueError("HF reference starts at a different position")
    teacher = manifest["teacher_tokens"]
    if len(teacher) < args.steps:
        raise ValueError("HF trace does not cover every requested forward")
    if args.capture_case:
        hashes = manifest.get("prefix_files_sha256", {})
        if (hashes.get(f"rank-{rank}.pt") != inputs["prefix_sha256"]
                or teacher[0] != inputs["first_token_id"]):
            raise ValueError("HF trace uses a different full-P continuation")
    elif teacher[:len(args.tokens)] != args.tokens:
        raise ValueError("HF teacher inputs differ from cold inputs")
    return manifest


def snapshot(engine, result, token, position):
    def cpu(value):
        return value.detach().cpu().clone()

    return {
        "token": token, "position": position,
        "committed_position": engine.position,
        "token_ids": cpu(result.token_ids),
        "logits": cpu(result.logits),
        "last_hidden": cpu(result.last_hidden),
        "cache_rows": tuple({
            "latent": cpu(layer["latent"][position]),
            "rope": cpu(layer["rope"][position]),
            "index": cpu(layer["index_keys"][position])
            if "index_keys" in layer else None,
            "index_scale": cpu(layer["index_scales"][position:position + 1])
            if "index_scales" in layer else None,
        } for layer in engine.cache),
    }


def capture_step(engine, token, position, layer_trace, tp_reduction="default",
                 mlp_layer=None):
    if tp_reduction == "default":
        return capture_forward(engine, token, position, layer_trace, mlp_layer)

    import torch
    from cann_reduction_probe import TpReductionProbe

    captures = 3 if mlp_layer is None else 2 * (mlp_layer + 1) + 1
    with TpReductionProbe(getattr(torch, tp_reduction),
                          capture_first=captures) as r:
        actual = capture_forward(
            engine, token, position, layer_trace, mlp_layer,
        )
    actual["tp_reductions"] = r.metadata()
    actual["tp_reduction_samples"] = [
        {name: value.detach().cpu().clone() for name, value in sample.items()}
        for sample in r.captures
    ]
    return actual


@contextmanager
def capture_mlp(engine, layer):
    stages = {}
    if layer is None:
        yield stages
        return
    from vllm_ascend.ops.fused_moe.moe_comm_method import (
        MoECommType, get_moe_comm_method,
    )

    def before(module, inputs):
        stages["input"] = inputs[0].detach().clone()

    def after(module, inputs, output):
        stages["output"] = output.detach().clone()

    mlp = engine.model.model.layers[layer].mlp
    runner = mlp.experts
    scheme = runner.routed_experts.quant_method.quant_method
    dispatcher = get_moe_comm_method(
        MoECommType.ALLGATHER, runner.moe_config,
    ).token_dispatcher
    original_apply = scheme.apply
    original_gmm1 = scheme.apply_gmm1_act_quant
    original_gmm2 = scheme.apply_gmm2
    original_combine = dispatcher.token_combine
    original_forward = runner._forward_impl

    def combine(hidden_states, combine_metadata, bias=None):
        if (tuple(combine_metadata.restore_shape)
                != tuple(stages["input"].shape)):
            raise ValueError("Expected M1 AllGather restore shape")
        stages["expanded_row_idx"] = (
            combine_metadata.expanded_row_idx.detach().clone()
        )
        stages["finalize_probs_fp32"] = (
            combine_metadata.topk_weights.detach().clone()
        )
        result = original_combine(hidden_states, combine_metadata, bias)
        stages["finalize"] = result.detach().clone()
        return result

    def apply(*args, **kwargs):
        stages["probs_fp32"] = kwargs["topk_weights"].detach().clone()
        stages["topk_ids"] = kwargs["topk_ids"].detach().clone()
        # The dispatcher is shared; patch only this layer's apply.
        with patch.object(dispatcher, "token_combine", combine):
            return original_apply(*args, **kwargs)

    def gmm1(value):
        if value.group_list_type != 1:
            raise ValueError("Expected AllGather expert-count groups")
        stages["dispatch_a8"] = value.hidden_states.detach().clone()
        stages["dispatch_scale"] = value.dynamic_scale.detach().clone()
        stages["group_counts"] = value.group_list.detach().clone()
        result = original_gmm1(value)
        stages["fused_a8"] = result[0].detach().clone()
        stages["fused_scale"] = result[1].detach().clone()
        return result

    def gmm2(value, hidden_states, act_out_scale):
        if value.group_list_type != 1:
            raise ValueError("Expected AllGather expert-count groups")
        stages["gmm2_group_counts"] = value.group_list.detach().clone()
        result = original_gmm2(value, hidden_states, act_out_scale)
        stages["down"] = result.detach().clone()
        return result

    def forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        shared, routed = result
        stages["combine"] = routed.detach().clone()
        stages["shared"] = shared.detach().clone()
        return result

    handles = [mlp.register_forward_pre_hook(before),
               mlp.register_forward_hook(after)]
    try:
        with (
            patch.object(scheme, "apply", apply),
            patch.object(scheme, "apply_gmm1_act_quant", gmm1),
            patch.object(scheme, "apply_gmm2", gmm2),
            patch.object(runner, "_forward_impl", forward),
        ):
            yield stages
    finally:
        for handle in handles:
            handle.remove()


def capture_forward(engine, token, position, layer_trace, mlp_layer=None):
    if not layer_trace:
        return snapshot(engine, engine.verify((token,)), token, position)

    import torch
    from vllm.config import set_current_vllm_config
    from tilert.models.glm_5_2_ascend.cann_choices import CannChoices
    from tilert.models.glm_5_2_ascend.cann_trace import CannTrace

    with torch.npu.device(engine.device), torch.inference_mode(), (
        set_current_vllm_config(engine.config)
    ):
        with (CannTrace(engine.model) as trace, CannChoices(engine.model) as c,
              capture_mlp(engine, mlp_layer) as mlp):
            result = engine.verify((token,))
            hidden = trace.finish(result)
            indices, routes = c.finish()
        normalized = engine.model.model.norm(torch.cat(hidden[1:-1]))
        projected = engine.model.compute_logits(normalized)
        if projected.shape != (77, result.logits.shape[-1]):
            raise ValueError("Layer projection must retain all 77 rows")
        logits = (*projected.split(1, dim=0), result.logits)
        actual = snapshot(engine, result, token, position)
        actual.update(
            hidden_states=tuple(value.detach().cpu().clone()
                                for value in hidden),
            layer_logits=tuple(value.detach().cpu().clone()
                               for value in logits),
            indices=tuple(value[0].detach().cpu().clone()
                          for value in indices),
            routes=tuple(None if value is None else
                         value[0].detach().cpu().clone() for value in routes),
            choice_source_sha256=c.source_metadata(),
        )
        if mlp_layer is not None:
            actual["mlp_layer"] = mlp_layer
            actual["mlp_stages"] = {
                name: value.detach().cpu().clone()
                for name, value in mlp.items()
            }
    return actual


def diagnose_tensor(name, actual, expected, checks, diagnostics, *,
                    logits=False):
    import torch
    import torch.nn.functional as functional

    if actual is None or expected is None:
        checks.append({"name": name + ".presence",
                       "passed": actual is None and expected is None})
        return
    layout = actual.shape == expected.shape and actual.dtype == expected.dtype
    checks.append({"name": name + ".layout", "passed": layout})
    item = {"name": name, "shape": list(actual.shape),
            "expected_shape": list(expected.shape), "dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype)}
    diagnostics.append(item)
    if not layout:
        return
    left, right = actual.float(), expected.float()
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    checks.append({"name": name + ".finite", "passed": finite})
    if not finite:
        return
    norm = right.norm().item()
    error = left - right
    cosine = functional.cosine_similarity(
        left.flatten(), right.flatten(), dim=0,
    ).item() if norm else float(torch.count_nonzero(left) == 0)
    item.update(
        nrmse=error.norm().item() / max(norm, 1e-12), cosine=cosine,
        max_abs=error.abs().max().item(), elements=actual.numel(),
        equal_elements=int(torch.count_nonzero(actual == expected)),
        evaluation_dtype="float32",
    )
    if logits:
        item["argmax_equal"] = torch.equal(
            actual.argmax(-1), expected.argmax(-1),
        )


def compare_choices(name, actual, expected, checks, *, exact=False,
                    diagnostics=None):
    import torch

    for field in ("indices", "routes"):
        left, right = actual.get(field, ()), expected.get(field, ())
        checks.append({"name": name + "." + field + ".count",
                       "passed": len(left) == len(right) == 78})
        for layer, (value, reference) in enumerate(zip(left, right)):
            label = f"{name}.{field}.{layer}"
            if value is None or reference is None:
                compare_tensor(label, value, reference, checks, exact=True)
                continue
            integer = (value.dtype in (torch.int32, torch.int64)
                       and reference.dtype in (torch.int32, torch.int64))
            single_row = all(t.ndim == 1 or (t.ndim == 2 and t.shape[0] == 1)
                             for t in (value, reference))
            checks.append({"name": label + ".integer_row",
                           "passed": integer and single_row})
            if not integer or not single_row:
                continue
            raw_value, raw_reference = value, reference
            value, reference = value.flatten(), reference.flatten()
            if field == "indices":
                valid = all(bool(torch.all((row == -1) | (
                    (row >= 0) & (row <= actual["position"])
                ))) for row in (value, reference))
                checks.append({"name": label + ".causal_range",
                               "passed": valid})
                value, reference = value[value >= 0], reference[reference >= 0]
                count = min(actual["position"] + 1, 2048)
            else:
                count = 8
                checks.append({"name": label + ".expert_range",
                               "passed": all(bool(torch.all(
                                   (row >= 0) & (row < 256)
                               )) for row in (value, reference))})
            checks.append({"name": label + ".valid_selection",
                           "passed": all(row.numel() == count
                                         and row.unique().numel() == count
                                         for row in (value, reference))})
            if exact:
                compare_tensor(label, raw_value, raw_reference, checks,
                               exact=True)
                continue
            if diagnostics is not None:
                left_set = set(value.tolist())
                right_set = set(reference.tolist())
                diagnostics.append({
                    "name": label, "overlap": len(left_set & right_set),
                    "actual_count": len(left_set),
                    "reference_count": len(right_set),
                    "sets_equal": left_set == right_set,
                    "order_equal": torch.equal(value.to(torch.int64),
                                               reference.to(torch.int64)),
                })
                continue
            value = value.to(torch.int64).sort().values
            reference = reference.to(torch.int64).sort().values
            if value.numel() == 0 or reference.numel() == 0:
                checks.append({"name": label + ".nonempty", "passed": False})
                continue
            compare_tensor(label, value, reference, checks, exact=True)


def compare_step(name, actual, expected, checks, *, exact=False, max_ulp=None,
                 diagnostics=None):
    """Keep exact/ULP checks; collect explicit HF differences separately."""
    import torch

    if exact or max_ulp is not None:
        diagnostics = None

    def compare(label, value, reference, *, tensor_exact=False, logits=False):
        if diagnostics is not None:
            diagnose_tensor(label, value, reference, checks, diagnostics,
                            logits=logits)
        else:
            compare_tensor(label, value, reference, checks,
                           exact=exact or tensor_exact, logits=logits,
                           max_ulp=max_ulp)

    checks.append({
        "name": name + ".position_and_input",
        "passed": all(actual[key] == expected[key]
                      for key in ("token", "position", "committed_position")),
    })
    if diagnostics is not None:
        checks.append({
            "name": name + ".causal_length",
            "passed": all(trace["committed_position"] == trace["position"] + 1
                          for trace in (actual, expected)),
        })
    compare(name + ".logits", actual["logits"], expected["logits"],
            logits=True)
    hidden = (expected["last_hidden"] if "last_hidden" in expected
              else expected["hidden_states"][-1])
    compare(name + ".last_hidden", actual["last_hidden"], hidden)
    if "indices" in actual or "routes" in actual:
        compare_choices(name, actual, expected, checks,
                        exact=exact or max_ulp is not None,
                        diagnostics=diagnostics)
    for field, count in (("hidden_states", 79), ("layer_logits", 78)):
        if field not in actual:
            continue
        left, right = actual[field], expected[field]
        checks.append({
            "name": name + "." + field + ".count",
            "passed": len(left) == len(right) == count,
        })
        for layer, (value, reference) in enumerate(zip(left, right)):
            compare(f"{name}.{field}.{layer}", value, reference,
                    logits=field == "layer_logits")
    checks.append({"name": name + ".cache_layer_count",
                   "passed": len(actual["cache_rows"]) == 78
                   and len(expected["cache_rows"]) == 78})
    for layer, (left, right) in enumerate(zip(
        actual["cache_rows"], expected["cache_rows"],
    )):
        for field in ("latent", "rope", "index", "index_scale"):
            value = left[field]
            integer = value is not None and value.dtype == torch.int8
            compare(f"{name}.cache.{layer}.{field}", value, right[field],
                    tensor_exact=integer or field == "index_scale")


def check_prefix(engine, prefixes, checks, label):
    import torch

    for number, (saved, live) in enumerate(zip(prefixes, engine.cache)):
        for field, value in saved.items():
            if value is None:
                continue
            current = live[field][:value.shape[0]].cpu()
            checks.append({
                "name": f"{label}.prefix.{number}.{field}",
                "passed": torch.equal(current, value), "exact": True,
            })


def execute(args):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    import torch.distributed as distributed
    from vllm.distributed import get_world_group
    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = int(os.environ["RANK"])
    device = (rank + args.device_offset) % 16
    metadata = checkpoint_fingerprint(args.checkpoint)
    prefixes = []
    if args.capture_case:
        prefixes, teacher, inputs = load_capture(
            args.capture_case, rank, metadata,
        )
    else:
        teacher = args.tokens
        inputs = {"initial_position": 0, "cold_teacher_tokens": teacher}
    if inputs["initial_position"] + args.steps > args.capacity:
        raise ValueError("Prefix and decode exceed capacity")
    reference = load_reference(args, metadata, inputs, rank)
    if reference:
        teacher = reference["teacher_tokens"][:args.steps]
    engine = CannModel(
        args.checkpoint, f"npu:{device}", rank, args.init_method,
        args.capacity, args.gpu_memory_utilization,
    )
    if args.tp_reduction != "default":
        engine._restore_tp_sum()
    if prefixes:
        engine.inject(prefixes)
    initial = engine.position
    rank_dir = args.output / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    checks, diagnostics, trace_hashes, consumed, predicted = [], [], {}, [], []
    reset_hashes = {}
    token = teacher[0]
    for step in range(args.steps):
        if step < len(teacher):
            token = teacher[step]
        position = engine.position
        actual = capture_step(engine, token, position, args.layer_trace,
                              args.tp_reduction, args.trace_mlp_layer)
        path = rank_dir / f"step-{position:05d}.pt"
        torch.save(actual, path)
        trace_hashes[path.name] = digest(path)
        consumed.append(token)
        token = int(actual["token_ids"].item())
        predicted.append(token)
        checks.append({
            "name": f"step.{step}.finite",
            "passed": bool(torch.isfinite(actual["logits"]).all()
                           and torch.isfinite(actual["last_hidden"]).all()),
        })
        if reference:
            expected_path = args.reference / path.name
            if digest(expected_path) != reference["trace_sha256"][path.name]:
                raise ValueError("HF trace file hash differs")
            expected = torch.load(
                expected_path, map_location="cpu", weights_only=True,
            )
            compare_step(f"hf.{step}", actual, expected, checks,
                         diagnostics=diagnostics)
    check_prefix(engine, prefixes, checks, "decode")
    engine.reset()
    checks.append({"name": "reset.position", "passed": engine.position == 0})
    if prefixes:
        engine.inject(prefixes)
    for step, token in enumerate(consumed):
        position = engine.position
        actual = capture_step(engine, token, position, args.layer_trace,
                              args.tp_reduction, args.trace_mlp_layer)
        reset_path = rank_dir / f"reset-step-{position:05d}.pt"
        torch.save(actual, reset_path)
        reset_hashes[reset_path.name] = digest(reset_path)
        expected = torch.load(
            rank_dir / f"step-{position:05d}.pt", map_location="cpu",
            weights_only=True,
        )
        compare_step(f"reset.{step}", actual, expected, checks, exact=True)
    check_prefix(engine, prefixes, checks, "reset")
    passed = all(check["passed"] for check in checks)
    completed_status = "diagnostic" if reference else "captured"
    report = {
        "status": completed_status if passed else "fail",
        "precision_verified": False,
        "continuation_parity": False, "layerwise_logits_parity": False,
        "hf_diagnostics_complete": bool(reference and passed),
        "comparison_kind": "cross_configuration_hf" if reference else None,
        "repeatability_passed": all(check["passed"] for check in checks
                                    if check["name"].startswith("reset.")),
        "scope": "M1 logits/hidden/choices/new KV and frozen prefix",
        "coverage": {
            "independent_reference": reference is not None,
            "final_logits": True, "final_normalized_hidden": True,
            "all_layers_new_kv": True, "frozen_prefix_exact": bool(prefixes),
            "reset_exact": True,
            "intermediate_hidden_states": args.layer_trace,
            "intermediate_layer_logits": args.layer_trace,
            "routing_choices": args.layer_trace,
            "index_choices": args.layer_trace,
            "mtp": False,
        },
        "performance_measured": False, "draft_model_loaded": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rank": rank, "physical_device": device, "world_size": 16,
        "initial_position": initial, "final_position": engine.position,
        "steps": args.steps, "consumed_tokens": consumed,
        "predicted_tokens": predicted, "inputs": inputs,
        "checkpoint_sha256": metadata,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tp_reduction": args.tp_reduction,
        "model_tp_reduction": engine.tp_reduction,
        "trace_mlp_layer": args.trace_mlp_layer,
        "collective_env": {key: os.environ.get(key) for key in (
            "HCCL_DETERMINISTIC", "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
        )},
        "first_dense_projections": {
            name: {
                "class": type(module).__name__,
                "tp_size": module.tp_size,
                "reduce_results": module.reduce_results,
                "custom_op": (None if module.custom_op is None
                              else type(module.custom_op).__name__),
            }
            for name, module in (
                ("o_proj", engine.model.model.layers[0].self_attn.o_proj),
                ("down_proj", engine.model.model.layers[0].mlp.down_proj),
            )
        },
        "packages": {name: version(name) for name in
                     ("torch", "torch-npu", "vllm", "vllm-ascend")},
        "source_sha256": {
            **engine.source_metadata(),
            **actual.get("choice_source_sha256", {}),
            str(Path(__file__).resolve()): digest(Path(__file__)),
        },
        "trace_sha256": trace_hashes, "reset_trace_sha256": reset_hashes,
        "checks": checks, "diagnostics": diagnostics,
    }
    if args.tp_reduction != "default":
        import cann_reduction_probe

        path = Path(cann_reduction_probe.__file__).resolve()
        report["source_sha256"][str(path)] = digest(path)
    if args.layer_trace:
        path = (Path(__file__).resolve().parents[2]
                / "tilert/models/glm_5_2_ascend/cann_trace.py")
        report["source_sha256"][str(path)] = digest(path)
    report_path = args.output / f"rank-{rank}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    summaries = [None] * 16
    distributed.all_gather_object(
        summaries, {"rank": rank, "passed": passed,
                    "continuation_parity": report["continuation_parity"],
                    "layerwise_logits_parity": (
                        report["layerwise_logits_parity"]
                    ),
                    "hf_diagnostics_complete": (
                        report["hf_diagnostics_complete"]
                    ),
                    "consumed_tokens": consumed, "predicted_tokens": predicted,
                    "report_sha256": digest(report_path)},
        group=get_world_group().cpu_group,
    )
    same_tokens = all(item["consumed_tokens"] == consumed
                      and item["predicted_tokens"] == predicted
                      for item in summaries)
    success = all(item["passed"] for item in summaries) and same_tokens
    if rank == 0:
        summary = {
            "status": report["status"] if success else "fail",
            "all_ranks_agree": same_tokens,
            "precision_verified": False,
            "continuation_parity": False, "layerwise_logits_parity": False,
            "hf_diagnostics_complete": bool(success and reference),
            "performance_measured": False, "ranks": summaries,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
        )
        print(json.dumps(summary))
    engine.shutdown()
    return 0 if success else 1


def main():
    args = parse_arguments()
    try:
        return execute(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        error_path = args.output / (
            f"rank-{os.environ.get('RANK', 'unknown')}-error.txt"
        )
        error_path.write_text(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
