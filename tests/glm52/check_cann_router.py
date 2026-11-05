"""Replay a frozen HF router input and optionally its single-NPU CANN path.

The CPU replay stops before experts. The device probe separates expert
selection on fixed HF logits from the production gate-plus-selection path.
Neither path creates a process group or measures inference performance.
"""

import argparse
from datetime import datetime, timezone
import inspect
from importlib.util import find_spec
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

from check_saved_hf_boundary import (  # noqa: E402
    inject_layer, load_inputs, npu_initialized,
)
from run_native_decode import compare_tensor, digest  # noqa: E402
from tilert.models.glm_5_2_ascend.reference import (  # noqa: E402
    HfReference, StableTopk,
)


class RouterInputCaptured(Exception):
    """End the decoder call before any expert weights are read."""


def source_hashes(values):
    return {str(path): digest(path) for path in {
        Path(inspect.getsourcefile(inspect.unwrap(value)))
        for value in values
    }}


def production_sources():
    paths = {
        "vllm": (
            "model_executor/models/deepseek_v2.py",
            "model_executor/layers/fused_moe/runner/moe_runner.py",
        ),
        "vllm_ascend": (
            "patch/platform/patch_fused_moe.py",
            "ops/fused_moe/fused_moe.py",
            "ops/fused_moe/token_dispatcher.py",
            "ops/fused_moe/gate_linear.py",
            "ops/fused_moe/router/fused_topk_router.py",
        ),
    }
    hashes = {}
    for module, relatives in paths.items():
        root = Path(next(iter(find_spec(module).submodule_search_locations)))
        for relative in relatives:
            path = root / relative
            hashes[str(path)] = digest(path)
    return hashes


def routing(logits, bias, scale=1.0):
    scores = logits.sigmoid()
    choice_scores = scores + bias
    ids = choice_scores.argsort(dim=-1, descending=True, stable=True)
    ids = ids[:, :8]
    weights = scores.gather(1, ids)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * scale
    return {"logits": logits, "scores": scores,
            "choice_scores": choice_scores, "ids": ids,
            "weights": weights, "weights_bf16": weights.bfloat16()}


def selection_boundary(choice_scores):
    ordered, ids = choice_scores.sort(descending=True, stable=True)
    eighth, ninth = ordered[0, 7:9].tolist()
    return {"eighth_expert": int(ids[0, 7]),
            "ninth_expert": int(ids[0, 8]),
            "eighth_score": eighth, "ninth_score": ninth,
            "eighth_minus_ninth": eighth - ninth,
            "boundary_tie": eighth == ninth,
            "boundary_equal_scores": int((choice_scores == eighth).sum()),
            "adjacent_equal_scores": int(
                (ordered[:, 1:] == ordered[:, :-1]).sum())}


def scale_placement(saved):
    expected = saved["reference"]
    early = saved["hf_scaled_weights"].bfloat16()
    late = expected["weights_bf16"] * 2.5
    witnesses = []
    for row, column in (early != late).nonzero().tolist():
        witnesses.append({
            "expert": int(expected["ids"][row, column]),
            "normalized_probability": float(expected["weights"][row, column]),
            "hf_unit_expert_output": float(early[row, column]),
            "cann_unit_expert_output": float(late[row, column]),
        })
    return {
        "hf": "BF16(normalized_weights * 2.5) before expert combine",
        "cann": "BF16(normalized_weights); combine; BF16(output * 2.5)",
        "router_scale": 1.0,
        "scope": "Real probabilities; synthetic unit output of one expert",
        "bf16_weight_rounding_witnesses": witnesses,
        "max_abs": float((early.float() - late.float()).abs().max()),
        "expert_combine_validated": False,
    }


@torch.inference_mode()
def capture_input(engine, traces, number):
    saved = traces[-1]
    layer = engine.model.model.layers[number]
    stages = {}
    inject_layer(engine, number, traces[:-1], stages)
    engine.indices, engine.routes, engine.latents = (
        [None] * 78 for _ in range(3)
    )
    hidden = saved["hidden_states"][number].unsqueeze(1).clone()
    positions = torch.tensor([[saved["position"]]], dtype=torch.int64)
    angles = engine.model.model.rotary_emb(hidden, position_ids=positions)
    shared = layer.self_attn.indexer is None
    previous = saved["indices"][number].reshape(1, 1, -1).clone()

    def stop_before_experts(module, inputs):
        stages["router.input"] = inputs[0].detach().clone()
        raise RouterInputCaptured

    hook = layer.mlp.register_forward_pre_hook(stop_before_experts)
    try:
        layer(hidden, attention_mask=None, position_ids=positions,
              past_key_values=engine.cache, use_cache=True,
              position_embeddings=angles,
              prev_topk_indices=previous if shared else None)
    except RouterInputCaptured:
        pass
    finally:
        hook.remove()
    if "router.input" not in stages:
        raise ValueError("HF replay did not reach the MoE input")
    gate = layer.mlp.gate
    with StableTopk():
        logits, weights, ids = gate(stages["router.input"])
    expected = routing(logits, gate.e_score_correction_bias)
    hf_expected = routing(logits, gate.e_score_correction_bias,
                          scale=gate.routed_scaling_factor)
    checks = []
    compare_tensor("hf.router.ids", ids, expected["ids"], checks,
                   exact=True)
    compare_tensor("hf.router.weights", weights, hf_expected["weights"],
                   checks, exact=True)
    compare_tensor("frozen.routes", ids.reshape(-1), saved["routes"][number],
                   checks, exact=True)
    compare_tensor("frozen.indices", engine.indices[number],
                   saved["indices"][number], checks, exact=True)
    cache = engine.cache.layers[number]
    indexer = layer.self_attn.indexer
    rows = {"latent": engine.latents[number],
            "rope": cache.keys[0, 0, -1, 192:],
            "index": None if indexer is None else indexer.last_key,
            "index_scale": None if indexer is None else indexer.last_scale}
    for name, value in rows.items():
        compare_tensor("frozen.cache." + name, value,
                       saved["cache_rows"][number][name], checks, exact=True)
    checks.append({"name": "frozen.cache.length",
                   "passed": cache.get_seq_length() == saved["position"] + 1})
    prefix = f"model.layers.{number}.mlp.gate"
    weight = engine.weights.tensor(prefix + ".weight").clone()
    bias = engine.weights.tensor(prefix + ".e_score_correction_bias").clone()
    if (weight.dtype != torch.bfloat16 or weight.shape != (256, 6144)
            or bias.dtype != torch.float32 or bias.shape != (256,)):
        raise ValueError("Expected BF16 [256,6144] gate and FP32 [256] bias")
    compare_tensor("hf.gate.weight", gate.weight, weight.float(), checks,
                   exact=True)
    return {"hidden": stages["router.input"].reshape(1, 6144),
            "weight": weight, "bias": bias, "reference": expected,
            "hf_scaled_weights": routing(
                logits, gate.e_score_correction_bias, scale=2.5,
            )["weights"]}, checks


def compare_routing(label, actual, expected, checks):
    actual_ids = actual["ids"].long()
    expected_ids = expected["ids"].long()
    actual_order = actual_ids.argsort(dim=-1)
    expected_order = expected_ids.argsort(dim=-1)
    compare_tensor(label + ".expert_set", actual_ids.gather(1, actual_order),
                   expected_ids.gather(1, expected_order), checks, exact=True)
    for name, exact in (("weights", False), ("weights_bf16", True)):
        value = actual["weights"]
        if name == "weights_bf16":
            value = value.bfloat16()
        compare_tensor(label + "." + name, value.gather(1, actual_order),
                       expected[name].gather(1, expected_order), checks,
                       exact=exact)


def device_config(checkpoint):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    return EngineArgs(
        model=str(checkpoint), trust_remote_code=True, dtype="bfloat16",
        quantization="ascend", tensor_parallel_size=16,
        distributed_executor_backend="external_launcher",
        enforce_eager=True, max_model_len=128, max_num_seqs=1,
        max_num_batched_tokens=128,
        attention_config={"indexer_kv_dtype": "int8"},
        speculative_config=None,
        additional_config={"enable_flashcomm1": False,
                           "enable_dsa_cp": False,
                           "enable_shared_expert_dp": False,
                           "enable_fused_mc2": 0},
    ).create_engine_config()


@torch.inference_mode()
def run_device(saved, device, checks, checkpoint):
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm_ascend.ops.fused_moe.gate_linear import AscendGateLinear
    from vllm_ascend.ops.fused_moe.router.fused_topk_router import (
        AscendFusedTopKRouter,
    )
    from vllm_ascend.utils import enable_custom_op, maybe_trans_nz

    torch.npu.set_device(device)
    limit = 1536 * 1024 ** 2
    properties = torch.npu.get_device_properties(device)
    torch.npu.set_per_process_memory_fraction(limit / properties.total_memory,
                                             device)
    torch.npu.reset_peak_memory_stats(device)
    if not enable_custom_op():
        raise RuntimeError("Production custom topk operator is unavailable")
    config = device_config(checkpoint)
    with set_current_vllm_config(config):
        hidden = saved["hidden"].to(device)
        raw_weight = saved["weight"].to(device)
        gate = SimpleNamespace(weight=maybe_trans_nz(raw_weight), bias=None,
                               out_dtype=torch.float32, return_bias=True)
        router = AscendFusedTopKRouter(
            top_k=8, global_num_experts=256, renormalize=True,
            use_grouped_topk=True, num_expert_group=1, topk_group=1,
            scoring_func="sigmoid", routed_scaling_factor=1.0,
            e_score_correction_bias=saved["bias"].to(device),
        )
        actual = {}
        for name, logits in (
            ("fixed_hf_logits", saved["reference"]["logits"].to(device)),
            ("actual_gate", AscendGateLinear.forward(gate, hidden)[0]),
        ):
            weights, ids = router._compute_routing(hidden, logits,
                                                   torch.int32)
            actual[name] = {"logits": logits.cpu(), "weights": weights.cpu(),
                            "ids": ids.cpu()}
            compare_routing(name, actual[name], saved["reference"], checks)
    compare_tensor("actual_gate.logits", actual["actual_gate"]["logits"],
                   saved["reference"]["logits"], checks)
    torch.npu.synchronize()
    metadata = {
        "gate": "AscendGateLinear.forward; BF16 operands / FP32 output",
        "gate_context": "Explicit weight/bias/out_dtype/return_bias fields",
        "weight_format": int(torch_npu.get_npu_format(gate.weight)),
        "router": "AscendFusedTopKRouter._compute_routing",
        "router_scale": 1.0,
        "output_scale": "Production runner multiplies combined output by 2.5",
        "allocator_limit_bytes": limit,
        "source_sha256": source_hashes((
            AscendGateLinear, AscendFusedTopKRouter, maybe_trans_nz,
        )),
        "peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "peak_reserved_bytes": torch.npu.max_memory_reserved(),
    }
    root = Path(inspect.getsourcefile(AscendGateLinear)).parents[3]
    kernel = root / "csrc/moe/moe_gating_top_k"
    for suffix in ("moe_gating_top_k_torch_adpt.h",
                   "op_host/moe_gating_top_k_tiling.cpp",
                   "op_kernel/moe_gating_top_k_without_group.h"):
        path = kernel / suffix
        metadata["source_sha256"][str(path)] = digest(path)
    return actual, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=(0, 1), default=0)
    parser.add_argument("--layer", type=int, choices=range(3, 78), default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device != "cpu" and not args.device.startswith("npu:"):
        parser.error("device must be cpu or an explicit npu:N")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest, traces, artifacts, sources = load_inputs(
        args.reference, args.checkpoint, args.step,
    )
    if manifest.get("mla_preprocess") != "prolog_v3":
        raise ValueError("This diagnostic requires the CANN Prolog HF trace")
    root = Path(__file__).resolve().parents[2]
    for name in manifest["source_sha256"]:
        if name.startswith("tilert/models/glm_5_2_ascend/"):
            actual = digest(root / name)
            sources[name] = actual
    engine = HfReference(args.checkpoint, mla_preprocess="prolog_v3")
    installed = Path(inspect.getsourcefile(type(engine.model)))
    installed_hashes = manifest["installed_hf_source_sha256"].values()
    if digest(installed) not in installed_hashes:
        raise ValueError("Installed HF source differs from the frozen trace")
    saved, checks = capture_input(engine, traces, args.layer)
    if not all(item["passed"] for item in checks):
        raise ValueError("The frozen HF router input did not replay exactly")
    sources.update(source_hashes((HfReference, capture_input, inject_layer)))
    metadata = {"layer": args.layer, "position": args.step,
                "token": traces[-1]["token"], "mla_preprocess": "prolog_v3",
                "checkpoint_sha256": manifest["metadata_sha256"],
                "artifact_sha256": artifacts, "source_sha256": sources,
                "frozen_source_sha256": manifest["source_sha256"]}
    metadata["production_source_sha256"] = production_sources()
    saved["metadata"] = metadata
    device_metadata = None
    if args.device != "cpu":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        actual, device_metadata = run_device(saved, args.device, checks,
                                             args.checkpoint)
        saved["actual"] = actual
    else:
        checks.append({"name": "no_npu_initialization",
                       "passed": not npu_initialized()})
    tensor_path = args.output / "router.pt"
    torch.save(saved, tensor_path)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen HF attention/router replay; optional CANN router",
        "precision_verified": False, "performance_measured": False,
        "experts_executed": False, "collectives_executed": False,
        "device": args.device, "metadata": metadata,
        "device_metadata": device_metadata,
        "selection_boundary": selection_boundary(
            saved["reference"]["choice_scores"]),
        "input_shape": list(saved["hidden"].shape),
        "weight_shape": list(saved["weight"].shape),
        "routes": saved["reference"]["ids"].tolist(),
        "scale_placement": scale_placement(saved),
        "tensor_sha256": digest(tensor_path), "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    report = json.dumps(result, indent=2) + "\n"
    (args.output / "report.json").write_text(report)
    print(json.dumps({"passed": result["passed"], "checks": len(checks),
                      "selection_boundary": result["selection_boundary"],
                      "output": str(args.output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
