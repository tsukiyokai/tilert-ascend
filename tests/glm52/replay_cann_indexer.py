"""Replay layer-zero BF16 indexing on a captured 8192-token prefix.

CPU preparation reproduces the saved HF choices. An explicit --device runs
the installed CANN backend and its selector with fixed HF inputs separately.
Reported scores are CPU FP32 reconstructions: paged LightningIndexer does
not expose its internal scores. No model, process group or HCCL is created.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402
import torch.nn.functional as functional  # noqa: E402

from check_cann_prolog import (  # noqa: E402
    common_arguments, digest, load_rotary, prolog_reference,
)
from check_cann_router import device_config, source_hashes  # noqa: E402
from tilert.models.glm_5_2_ascend.reference import (  # noqa: E402
    interleaved_rope, prolog_input_norm,
)
from tilert.models.glm_5_2_ascend.reference_prolog import (  # noqa: E402
    project_codes,
)
from tilert.models.glm_5_2_ascend.reference_weights import (  # noqa: E402
    ReferenceWeights, quantize_dynamic_a8,
)


ATTENTION = "model.layers.0.self_attn"
INDEXER = ATTENTION + ".indexer"
POSITION = 8192


def difference(actual, expected):
    delta = actual.float() - expected.float()
    return {"equal": torch.equal(actual, expected),
            "shape": list(actual.shape), "dtype": str(actual.dtype),
            "unequal": int((actual != expected).sum()),
            "max_abs": float(delta.abs().max()),
            "nrmse": float(delta.norm()
                           / expected.float().norm().clamp_min(1e-30))}


def selection_difference(actual, expected):
    left = set(actual.flatten().tolist())
    right = set(expected.flatten().tolist())
    return {"equal": left == right, "intersection": len(left & right),
            "actual_only": sorted(left - right),
            "expected_only": sorted(right - left)}


def score(query, keys, weights):
    products = query.reshape(32, 128).float() @ keys.float().T
    return (products.relu() * weights.reshape(32, 1).float()).sum(0)


def boundary(scores):
    order = scores.argsort(descending=True, stable=True)
    selected, excluded = int(order[2047]), int(order[2048])
    return {"cutoff_index": selected, "next_index": excluded,
            "cutoff_gap": float(scores[selected] - scores[excluded]),
            "scores": {str(i): float(scores[i])
                       for i in sorted({5950, 3782, selected, excluded})}}


@torch.inference_mode()
def prepare(args):
    weights = ReferenceWeights(args.checkpoint)
    report_path = args.trace.parent.parent / "rank-0.json"
    report = json.loads(report_path.read_text())
    manifest_path = args.reference.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    hashes = weights.contract["metadata_sha256"]
    assert report["checkpoint_sha256"] == manifest["metadata_sha256"] == hashes
    assert manifest["status"] == "pass"
    assert manifest["mla_preprocess"] == "prolog_v3"
    assert digest(args.trace) == report["trace_sha256"][args.trace.name]
    assert (digest(args.reference)
            == manifest["trace_sha256"][args.reference.name])
    assert digest(args.prefix) == report["inputs"]["prefix_sha256"]
    assert (manifest["prefix_files_sha256"]["rank-0.pt"]
            == report["inputs"]["prefix_sha256"])
    assert manifest["prompt_sha256"] == report["inputs"]["prompt_sha256"]
    assert manifest["prompt_length"] == report["initial_position"] == POSITION
    native, hf, prefix = [torch.load(
        path, map_location="cpu", weights_only=True, mmap=True,
    ) for path in (args.trace, args.reference, args.prefix)]
    assert prefix["metadata_sha256"] == hashes
    assert native["position"] == hf["position"] == POSITION
    assert native["token"] == hf["token"] == 6500
    hidden = hf["hidden_states"][0].clone()
    assert torch.equal(hidden, native["hidden_states"][0])
    keys = prefix["layers"][0]["index_keys"].clone()
    assert keys.shape == (POSITION, 128) and keys.dtype == torch.bfloat16
    assert "index_scales" not in prefix["layers"][0]
    cosine, sine, rotary = load_rotary(
        args.checkpoint, args.reference, hidden, POSITION,
    )

    def read(name, rows=None):
        return weights.tensor(name, rows=rows).clone()

    def projection(stem, rows=None):
        name = ATTENTION + "." + stem
        assert not torch.count_nonzero(read(name + ".weight_offset", rows))
        return {"weight": read(name + ".weight", rows),
                "scale": read(name + ".weight_scale", rows).flatten()}

    raw = {"q_a": projection("q_a_proj"),
           "kv_a": projection("kv_a_proj_with_mqa"),
           "q_b": projection("q_b_proj", (0, 4 * 256)),
           "kv_b": read(ATTENTION + ".kv_b_proj.weight", (0, 4 * 448)),
           "q_gamma": read(ATTENTION + ".q_a_layernorm.weight").bfloat16(),
           "kv_gamma": read(ATTENTION + ".kv_a_layernorm.weight").bfloat16(),
           "input_gamma": read("model.layers.0.input_layernorm.weight")
           .bfloat16()}
    norm = SimpleNamespace(weight=raw["input_gamma"], variance_epsilon=1e-5)
    normalized = prolog_input_norm(norm, hidden)
    codes, scale = quantize_dynamic_a8(normalized)
    prolog = prolog_reference(codes, scale, raw, cosine, sine)
    query = project_codes(
        weights, INDEXER + ".wq_b", prolog["query_norm"],
        prolog["query_norm_scale"], scale_order="channel_first",
    ).bfloat16().reshape(1, 1, 32, 128)
    key = weights.linear(INDEXER + ".wk", normalized, "replicated")
    key = functional.layer_norm(
        key.float(), (128,), read(INDEXER + ".k_norm.weight").float(),
        read(INDEXER + ".k_norm.bias").float(), eps=1e-6,
    ).bfloat16().reshape(1, 1, 1, 128)

    def rotate(value):
        angles = cosine.unsqueeze(0), sine.unsqueeze(0)
        return torch.cat((interleaved_rope(value[..., :64], *angles),
                          value[..., 64:]), dim=-1)

    query, key = rotate(query).reshape(1, 32, 128), rotate(key).reshape(1, 128)
    head_weights = weights.linear(
        INDEXER + ".weights_proj", normalized, "replicated",
    )
    scores = score(query, torch.cat((keys, key)), head_weights)
    indices = scores.argsort(descending=True, stable=True)[:2048]
    assert torch.equal(indices.int(), hf["indices"][0].flatten())
    assert torch.equal(key.flatten(), hf["cache_rows"][0]["index"].flatten())
    reference = {"input_norm": normalized,
                 "input_codes": codes, "input_scale": scale.flatten(),
                 "q_codes": prolog["query_norm"],
                 "q_scale": prolog["query_norm_scale"].flatten(),
                 "query": query, "key": key, "weights": head_weights,
                 "scores": scores, "indices": indices}
    identity = {"checkpoint_sha256": hashes, "inputs": report["inputs"],
                "artifacts": {str(p): digest(p) for p in (
                    args.trace, args.reference, args.prefix,
                    report_path, manifest_path)},
                "source_sha256": source_hashes([
                    prepare, prolog_input_norm, project_codes,
                    ReferenceWeights, prolog_reference]), "rotary": rotary,
                "hf_replay_exact": True,
                "frozen_cann_choices": selection_difference(
                    native["indices"][0], indices),
                "cpu_reference_boundary": boundary(scores)}
    return {"hidden": hidden, "prefix_keys": keys, "cos": cosine,
            "sin": sine, "prolog_weights": raw, "reference": reference,
            "frozen_cann_indices": native["indices"][0].clone(),
            "identity": identity}, weights


def load_backend(weights, config):
    from vllm.model_executor import parameter as parameter_module
    from vllm.model_executor.layers.layernorm import LayerNorm
    from vllm.model_executor.model_loader.utils import (
        configure_quant_config, get_model_architecture,
    )
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
    from vllm_ascend.ops.linear import (
        AscendMergedColumnParallelLinear, AscendReplicatedLinear,
    )

    model_class, _ = get_model_architecture(config.model_config)
    configure_quant_config(config.quant_config, model_class)
    owner = SimpleNamespace(n_head=32, head_dim=128, topk_tokens=2048,
                            q_lora_rank=2048, softmax_scale=128 ** -0.5)
    owner.wq_b = AscendReplicatedLinear(
        2048, 4096, bias=False, params_dtype=torch.bfloat16,
        quant_config=config.quant_config, prefix=INDEXER + ".wq_b",
        disable_tp=True,
    )
    # Replicated parameters still query TP metadata during construction.
    with (
        patch.object(parameter_module, "get_tensor_model_parallel_rank",
                     return_value=0),
        patch.object(parameter_module, "get_tensor_model_parallel_world_size",
                     return_value=1),
    ):
        owner.wk_weights_proj = AscendMergedColumnParallelLinear(
            6144, [128, 32], bias=False, params_dtype=torch.bfloat16,
            quant_config=None, prefix=INDEXER + ".wk_weights_proj",
            disable_tp=True,
        )
    owner.k_norm = LayerNorm(128, eps=1e-6, dtype=torch.float32)
    for suffix in ("weight", "weight_scale", "weight_offset"):
        parameter = getattr(owner.wq_b, suffix)
        parameter.weight_loader(parameter, weights.tensor(
            INDEXER + ".wq_b." + suffix,
        ))
    parameter = owner.wk_weights_proj.weight
    for shard, stem in enumerate(("wk", "weights_proj")):
        parameter.weight_loader(parameter, weights.tensor(
            INDEXER + "." + stem + ".weight",
        ), shard)
    for suffix in ("weight", "bias"):
        getattr(owner.k_norm, suffix).data.copy_(weights.tensor(
            INDEXER + ".k_norm." + suffix,
        ))
    owner.k_cache = SimpleNamespace(prefix=INDEXER + ".k_cache", kv_cache=())
    backend = AscendSFAIndexerBackend(owner, 64)
    assert not backend.enable_sparse_li_quant
    return backend


@torch.inference_mode()
def device_replay(args, saved, weights):
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton,
    )
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_per_process_memory_fraction(
        1536 * 1024 ** 2 / torch.npu.get_device_properties(args.device)
        .total_memory, args.device,
    )
    init_device_properties_triton()
    assert enable_custom_op()
    config = device_config(args.checkpoint)
    with set_current_vllm_config(config):
        backend = load_backend(weights, config).to(args.device)
        for layer in (backend.wq_b, backend.wk_weights_proj):
            layer.quant_method.process_weights_after_loading(layer)
        page = torch.zeros(65, 128, 1, 128, dtype=torch.bfloat16,
                           device=args.device)
        page.flatten(0, 2)[:POSITION].copy_(saved["prefix_keys"])
        backend.k_cache.kv_cache = (page,)
        cosine = saved["cos"].to(args.device)
        sine = saved["sin"].to(args.device)
        lengths = torch.tensor([POSITION + 1], dtype=torch.int32,
                               device=args.device)
        query_lengths = torch.ones(1, dtype=torch.int32, device=args.device)
        metadata = AscendSFAIndexerMetadata(
            1, torch.tensor([POSITION], dtype=torch.int32, device=args.device),
            lengths, query_lengths,
            torch.arange(65, dtype=torch.int32, device=args.device)[None],
            sine, cosine, block_size=128,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_key=lengths, num_decode_tokens=1,
        )
        captured = {}

        def select_values(query, head_weights):
            return DeviceOperator.indexer_select_post_process(
                q_li=query, q_li_scale=None, q_li_shape_ori=None,
                weights=head_weights, kv_cache=(page,),
                indexer_k_cache_idx=0, indexer_scale_cache_idx=1,
                attn_metadata=metadata,
                actual_seq_lengths_query=query_lengths,
                actual_seq_lengths_key=lengths,
                enable_sparse_li_c8=False, enable_sparse_li_c4=False,
                use_torch_npu_lightning_indexer=True,
            )

        def select(query, head_weights, indexer, meta, query_scale):
            assert query_scale is None
            captured.update(query=query.clone(), weights=head_weights.clone(),
                            key=page.flatten(0, 2)[POSITION].clone()[None])
            return select_values(query, head_weights)

        metadata.topk_selector = select
        raw = saved["prolog_weights"]
        hidden = torch_npu.npu_rms_norm(
            saved["hidden"].to(args.device),
            raw["input_gamma"].to(args.device),
            1e-5,
        )[0]
        codes, scales = torch_npu.npu_dynamic_quant(hidden)
        arguments = common_arguments(raw, saved["cos"], saved["sin"], POSITION)
        arguments = {k: v.to(args.device) if torch.is_tensor(v) else v
                     for k, v in arguments.items()}
        for name in ("weight_dq", "weight_dkv_kr", "weight_uq_qr"):
            arguments[name] = torch_npu.npu_format_cast(arguments[name], 29)
        latent = torch.zeros(65, 128, 1, 512, dtype=torch.bfloat16,
                             device=args.device)
        rope = torch.zeros(65, 128, 1, 64, dtype=torch.bfloat16,
                           device=args.device)
        _, _, _, q_codes, q_scale = torch_npu.npu_mla_prolog_v3(
            token_x=codes, dequant_scale_x=scales.reshape(1, 1),
            kv_cache=latent, kr_cache=rope, **arguments,
        )
        indices = backend(hidden, (q_codes, q_scale.flatten()),
                          hidden, metadata)
        captured.update(input_norm=hidden.clone(), input_codes=codes.clone(),
                        input_scale=scales.flatten().clone(),
                        q_codes=q_codes.clone(),
                        q_scale=q_scale.flatten().clone(), indices=indices)
        reference = saved["reference"]
        page.flatten(0, 2)[POSITION].copy_(reference["key"].flatten())
        fixed = select_values(
            reference["query"].to(args.device),
            reference["weights"].to(args.device),
        )
        torch.npu.synchronize()
        captured = {k: v.cpu() for k, v in captured.items()}
        captured["fixed_hf_indices"] = fixed.cpu()
        captured["scores"] = score(captured["query"], torch.cat((
            saved["prefix_keys"], captured["key"],
        )), captured["weights"])
        prolog_same_input = prolog_reference(
            captured["input_codes"], captured["input_scale"], raw,
            saved["cos"], saved["sin"],
        )
        result = {"stage_differences": {
            name: difference(captured[name].reshape_as(reference[name]),
                             reference[name])
            for name in ("input_norm", "input_codes", "input_scale",
                         "q_codes", "q_scale", "query", "key", "weights")},
            "prolog_same_input": {
                "q_codes": difference(captured["q_codes"],
                                      prolog_same_input["query_norm"]),
                "q_scale": difference(captured["q_scale"],
                                      prolog_same_input["query_norm_scale"]
                                      .flatten())},
            "full_backend_vs_hf": selection_difference(indices.cpu(),
                                                        reference["indices"]),
            "fixed_hf_selector_vs_hf": selection_difference(
                fixed.cpu(), reference["indices"]),
            "full_backend_vs_frozen": selection_difference(
                indices.cpu(), saved["frozen_cann_indices"]),
            "actual_input_cpu_boundary": boundary(captured["scores"]),
            "actual_input_cpu_selector": selection_difference(
                indices.cpu(), captured["scores"].argsort(
                    descending=True, stable=True)[:2048]),
            "source_sha256": source_hashes([
                type(backend), type(backend.wq_b), type(backend.k_norm),
                DeviceOperator.indexer_select_post_process]),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(args.device)}
    return captured, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "trace", "reference", "prefix", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", help="Explicit single NPU, e.g. npu:15")
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved, weights = prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "input.pt"
    torch.save(saved, path)
    result = {**saved["identity"], "prepared": str(path),
              "prepared_sha256": digest(path), "device_executed": False,
              "score_scope": "CPU FP32 reconstruction; no device score output",
              "constructor_metadata_patch": "Replicated weight rank=0,size=1",
              "status": "cpu_prepared", "precision_verified": False}
    if args.device:
        captured, measured = device_replay(args, saved, weights)
        torch.save(captured, args.output / "device.pt")
        result.update(measured, device_executed=True, device=args.device,
                      status="captured",
                      device_sha256=digest(args.output / "device.pt"))
    result["npu_initialized"] = (hasattr(torch, "npu")
                                 and torch.npu.is_initialized())
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
    )
    print(json.dumps({k: result[k] for k in (
        "status", "device_executed", "npu_initialized",
    )}))


if __name__ == "__main__":
    main()
