"""Replay frozen layers 66..73 through real rank-0 attention, stopping before O/TP."""

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch

BASE = Path("/workspace/experiments/layer-major-local-attention-20261020")
SOURCE = Path("/workspace/experiments/layer-major-m32-20261020/source")
sys.path[:0] = [str(SOURCE), str(SOURCE / "tests/glm52")]
WIDTH, CAPACITY, POSITION = 32, 9344, 8192
LAYERS, PRODUCERS = range(66, 74), (66, 70)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def difference(value, reference):
    value, reference = value.detach().cpu(), reference.detach().cpu()
    a, b = value.double(), reference.double()
    return {"bitexact": torch.equal(value.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)),
            "finite": bool(torch.isfinite(value).all()),
            "maxabs": (a - b).abs().max().item(),
            "nrmse": ((a - b).norm() / b.norm()).item()}


def raw_weights(checkpoint, number):
    stem = f"model.layers.{number}"
    values = {}
    for short, name, rows in (("q_a", "q_a_proj", None),
                              ("kv_a", "kv_a_proj_with_mqa", None),
                              ("q_b", "q_b_proj", slice(0, 1024))):
        prefix = stem + ".self_attn." + name
        select = None if rows is None else (rows, slice(None))
        values[short] = checkpoint.tensor(prefix + ".weight", select)
        values[short + "_scale"] = checkpoint.tensor(prefix + ".weight_scale", rows).reshape(1, -1)
        assert not bool(checkpoint.tensor(prefix + ".weight_offset", rows).any())
    for short, name in (("input_gamma", "input_layernorm"),
                        ("q_gamma", "self_attn.q_a_layernorm"),
                        ("kv_gamma", "self_attn.kv_a_layernorm")):
        values[short] = checkpoint.tensor(stem + "." + name + ".weight").bfloat16()
    values["kv_b"] = checkpoint.tensor(stem + ".self_attn.kv_b_proj.weight",
                                       (slice(0, 1792), slice(None))).bfloat16()
    assert [values[k].shape for k in ("q_a", "kv_a", "q_b", "kv_b")] == [
        (2048, 6144), (576, 6144), (1024, 2048), (1792, 512)]
    return values


def prolog_owner(raw, number, device, sfa):
    values = {name: value.to(device) for name, value in raw.items()}
    owner = SimpleNamespace(
        _quant_type=sfa.AscendW8A8DynamicLinearMethod,
        enable_sparse_sfa_c8=False, enable_sparse_sfa_turboquant=False,
        has_indexer=number in PRODUCERS, skip_topk=number not in PRODUCERS,
        local_num_heads=4, q_lora_rank=2048, kv_lora_rank=512,
        qk_rope_head_dim=64, v_head_dim=256, scale=1 / 16,
        q_a_layernorm=SimpleNamespace(weight=values["q_gamma"], variance_epsilon=1e-5),
        kv_a_layernorm=SimpleNamespace(weight=values["kv_gamma"], variance_epsilon=1e-5),
        weight_dq=values["q_a"].T.contiguous(),
        weight_dkv_kr=values["kv_a"].T.contiguous(),
        weight_uq_qr=values["q_b"].T.contiguous(),
        dequant_scale_w_dq=values["q_a_scale"],
        dequant_scale_w_dkv_kr=values["kv_a_scale"],
        dequant_scale_w_uq_qr=values["q_b_scale"],
        W_UK_T=values["kv_b"].view(4, 448, 512)[:, :192].contiguous(),
        W_UV=values["kv_b"].view(4, 448, 512)[:, 192:].transpose(1, 2).contiguous(),
    )
    if device.type == "npu":
        import torch_npu
        for name in ("weight_dq", "weight_dkv_kr", "weight_uq_qr"):
            setattr(owner, name, torch_npu.npu_format_cast(getattr(owner, name), 29))
    owner._execute_sparse_flash_attention_process = MethodType(
        sfa.AscendSFAImpl._execute_sparse_flash_attention_process, owner)
    return owner, values["input_gamma"]


def indexer(checkpoint, number, config, device):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers.layernorm import LayerNorm
    from vllm_ascend.ops.linear import AscendMergedColumnParallelLinear, AscendReplicatedLinear
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend

    prefix = f"model.layers.{number}.self_attn.indexer"
    with (torch.device(device),
          patch.object(parameter, "get_tensor_model_parallel_rank", return_value=0),
          patch.object(parameter, "get_tensor_model_parallel_world_size", return_value=1)):
        owner = SimpleNamespace(n_head=32, head_dim=128, topk_tokens=2048,
                                q_lora_rank=2048, softmax_scale=128 ** -0.5)
        owner.wq_b = AscendReplicatedLinear(2048, 4096, bias=False,
            params_dtype=torch.bfloat16, quant_config=config.quant_config,
            prefix=prefix + ".wq_b", disable_tp=True)
        owner.wk_weights_proj = AscendMergedColumnParallelLinear(6144, [128, 32],
            bias=False, params_dtype=torch.bfloat16, quant_config=None,
            prefix=prefix + ".wk_weights_proj", disable_tp=True)
        owner.k_norm = LayerNorm(128, eps=1e-6, dtype=torch.float32)
        for suffix in ("weight", "weight_scale", "weight_offset"):
            target = getattr(owner.wq_b, suffix)
            target.weight_loader(target, checkpoint.tensor(prefix + ".wq_b." + suffix))
        target = owner.wk_weights_proj.weight
        for shard, name in enumerate(("wk", "weights_proj")):
            target.weight_loader(target, checkpoint.tensor(prefix + "." + name + ".weight"), shard)
        for suffix in ("weight", "bias"):
            getattr(owner.k_norm, suffix).data.copy_(checkpoint.tensor(prefix + ".k_norm." + suffix))
        owner.k_cache = SimpleNamespace(prefix=prefix + ".k_cache", kv_cache=())
        backend = AscendSFAIndexerBackend(owner, 64)
    assert backend.enable_sparse_li_c8 is (number == 66)
    assert not backend._pcp_active and not backend._dsa_cp_active
    if device.type == "npu":
        for layer in (backend.wq_b, backend.wk_weights_proj):
            layer.quant_method.process_weights_after_loading(layer)
        backend.process_weights_after_loading()
    return backend


def main(args):
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import configure_quant_config, get_model_architecture
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm_ascend.attention import sfa_v1 as sfa
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata
    from vllm_ascend.ops.layernorm import AscendRMSNorm
    from vllm_ascend.ops import rotary_embedding as rope
    from tilert.models.glm_5_2_ascend.cann_attention import SharedKVAttention, selection_mask, shared_kv_fia
    from tilert.models.glm_5_2_ascend.cann_model import create_config
    from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "rank0 frozen teacher inputs, attention-only layers66..73; no O-projection, TP reduction, MoE, head or timing",
              "device_executed": bool(args.device), "graph_captured": False,
              "direct_inputs": ["M1 per-layer input_hidden/input_residual", "8K prefix", "checkpoint weights"],
              "executed": ["actual AscendRMSNorm", "actual AscendSFAImpl PrologV3", "actual producer indexer", "SFA/shared/rebuilt FIA", "actual V-up projection"],
              "original_M1_preO_saved": False, "layers": [], "source_sha256": {},
              "quota_bytes": args.quota_mib * 2 ** 20}
    config = create_config(args.checkpoint, CAPACITY, 0.02, graph_width=WIDTH)
    model_class, _ = get_model_architecture(config.model_config)
    configure_quant_config(config.quant_config, model_class)
    checkpoint = ModelSlimCheckpoint(args.checkpoint)
    original = json.loads((args.raw / "rank-0.json").read_text())
    prefix = torch.load(args.prefix, map_location="cpu", weights_only=True, mmap=True)
    assert prefix["metadata_sha256"] == original["checkpoint_sha256"]
    assert all(digest(args.checkpoint / name) == sha for name, sha in original["checkpoint_sha256"].items())
    assert original["teacher_source"]["sha256"] == "82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655"
    report["original_report_sha256"] = digest(args.raw / "rank-0.json")
    report["prefix_sha256"] = digest(args.prefix)
    assert report["prefix_sha256"] == original["inputs"]["prefix_sha256"]
    device = torch.device(args.device or "cpu")
    if args.device:
        torch.npu.set_device(device)
        torch.npu.config.allow_internal_format = True
        torch.npu.set_per_process_memory_fraction(report["quota_bytes"] /
            torch.npu.get_device_properties(device).total_memory, device)
        torch.npu.reset_peak_memory_stats(device)
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
        init_device_properties_triton()
        from vllm_ascend.utils import enable_custom_op
        assert enable_custom_op()
    with (set_current_vllm_config(config), set_default_torch_dtype(torch.bfloat16), torch.inference_mode()):
        with torch.device(device):
            rotary = rope.AscendRotaryEmbedding(64, 64, CAPACITY, 8000000, False, torch.bfloat16)
        positions = torch.arange(POSITION, POSITION + WIDTH, dtype=torch.int64, device=device)
        cosine, sine = rope._cos_cache[positions], rope._sin_cache[positions]
        table = torch.arange(CAPACITY // 128, dtype=torch.int32, device=device)[None]
        slots, qlens = positions.int(), torch.tensor([WIDTH], dtype=torch.int32, device=device)
        klens = torch.tensor([POSITION + WIDTH], dtype=torch.int32, device=device)
        shared = SharedKVAttention(WIDTH, CAPACITY, device)
        previous_ids = None
        previous_oracle = None
        for number in LAYERS:
            path = args.raw / f"rank-0/layer-{number:02d}.pt"
            sha = digest(path)
            assert sha == next(v for k, v in original["artifact_sha256"].items() if k.endswith(f"/rank-0/layer-{number:02d}.pt"))
            frozen = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            assert frozen["positions"] == list(range(POSITION, POSITION + WIDTH))
            assert frozen["skip_topk"] is (number not in PRODUCERS)
            oracle = frozen["indices"]
            assert oracle.shape == (WIDTH, 2048)
            assert bool(((oracle >= 0) & (oracle <= torch.arange(POSITION, POSITION + WIDTH)[:, None])).all())
            assert bool((oracle.sort(1).values.diff() > 0).all())
            if number in PRODUCERS:
                previous_oracle = oracle
            else:
                assert torch.equal(oracle, previous_oracle)
            raw = raw_weights(checkpoint, number)
            owner, gamma = prolog_owner(raw, number, device, sfa)
            backend = indexer(checkpoint, number, config, device) if number in PRODUCERS else None
            entry = {"layer": number, "input_sha256": sha, "indexer_c8": number == 66,
                     "weight_tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                         "sha256": hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                         for k, v in raw.items()}}
            if backend is not None:
                for value in (type(backend.wq_b), type(backend.wq_b.quant_method),
                              type(backend.wq_b.quant_method.quant_method), type(backend.wk_weights_proj)):
                    path = Path(inspect.getfile(value)).resolve()
                    report["source_sha256"][str(path)] = digest(path)
            if not args.device:
                assert frozen["input_hidden"].shape == frozen["input_residual"].shape == (WIDTH, 6144)
                assert bool(torch.isfinite(frozen["input_hidden"]).all())
                assert bool(torch.isfinite(frozen["input_residual"]).all())
                mask = selection_mask(oracle, shared.columns > torch.arange(POSITION, POSITION + WIDTH)[:, None])
                assert torch.equal((~mask).sum(1), torch.full((WIDTH,), 2048))
                entry["cpu_contract"] = "pass"
                report["layers"].append(entry)
                del backend, owner, raw
                gc.collect()
                continue
            fields = prefix["layers"][number]
            cache = {name: torch.zeros((CAPACITY // 128, 128, 1, width), dtype=value.dtype, device=device)
                     for name, width in (("latent", 512), ("rope", 64), ("index_keys", 128), ("index_scales", 1))
                     if isinstance(value := fields.get(name), torch.Tensor)}
            for name, value in fields.items():
                if isinstance(value, torch.Tensor):
                    cache[name].view(CAPACITY, -1)[:POSITION].copy_(value.reshape(POSITION, -1))
            kv = (cache["latent"], cache["rope"])
            norm = SimpleNamespace(weight=gamma, bias=torch.zeros_like(gamma), variance_epsilon=1e-5)
            hidden, _ = AscendRMSNorm.forward_oot(norm, frozen["input_hidden"].to(device), frozen["input_residual"].to(device))
            _, q, qr, qc = sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(owner, hidden, kv, cosine, sine, positions)
            if backend is not None:
                backend.k_cache.kv_cache = tuple(cache[n] for n in ("index_keys", "index_scales") if n in cache)
                meta = AscendSFAIndexerMetadata(WIDTH, slots, klens, qlens, table, sine, cosine,
                    block_size=128, actual_seq_lengths_query=qlens, actual_seq_lengths_key=klens, num_decode_tokens=0)
                previous_ids = backend(hidden, qc, hidden, meta)
            indices = previous_ids
            assert indices is not None
            assert bool(((indices.reshape(WIDTH, -1) >= 0) & (indices.reshape(WIDTH, -1) <= positions[:, None])).all())
            assert bool((indices.reshape(WIDTH, -1).sort(1).values.diff() > 0).all())
            metadata = SimpleNamespace(slot_mapping=slots, block_table=table)
            control = owner._execute_sparse_flash_attention_process(q, qr, kv, indices, metadata, qlens, klens)
            shared.bind(owner)
            previous_mask = shared.mask
            candidate = owner._execute_sparse_flash_attention_process(q, qr, kv, indices, metadata, qlens, klens)
            assert ((shared.mask is not previous_mask) if number in PRODUCERS
                    else (shared.mask is previous_mask))
            rebuilt_mask = selection_mask(indices, shared.columns > slots[:, None])
            rebuilt = shared_kv_fia(q, qr, kv, rebuilt_mask, owner.scale)
            assert torch.equal(shared.mask, rebuilt_mask) and torch.equal(candidate, rebuilt)
            pre_o = {name: sfa.AscendSFAImpl._v_up_proj(owner, value) for name, value in
                     (("sfa", control), ("shared_fia", candidate), ("rebuilt_fia", rebuilt))}
            assert all(bool(torch.isfinite(value).all()) for value in pre_o.values())
            assert torch.equal(pre_o["shared_fia"], pre_o["rebuilt_fia"])
            rows = {name: value.view(CAPACITY, -1)[POSITION:POSITION + WIDTH].cpu().clone() for name, value in cache.items()}
            entry.update(pre_o_difference=difference(pre_o["shared_fia"], pre_o["sfa"]),
                cache_differences={name: difference(value.reshape_as(frozen["cache_rows"][name]), frozen["cache_rows"][name]) for name, value in rows.items()},
                indices_same_set=torch.equal(indices.cpu().reshape(WIDTH, -1).sort(1).values, frozen["indices"].sort(1).values),
                shared_mask_lifecycle="refreshed" if number in PRODUCERS else "reused",
                peak_allocated=torch.npu.max_memory_allocated(device), peak_reserved=torch.npu.max_memory_reserved(device))
            torch.save({"q": q.cpu(), "qr": qr.cpu(), "indices": indices.cpu(), "new_cache_rows": rows,
                        "pre_o": {k: v.cpu() for k, v in pre_o.items()},
                        "attention": {"sfa": control.cpu(), "shared_fia": candidate.cpu(), "rebuilt_fia": rebuilt.cpu()}},
                       args.output / f"layer-{number:02d}.pt")
            report["layers"].append(entry)
            del backend, owner, raw, kv, cache, hidden, control, candidate, rebuilt, pre_o
            gc.collect()
            torch.npu.empty_cache()
    for value in (main, SharedKVAttention, AscendRMSNorm, sfa.AscendSFAImpl,
                  sfa.AscendW8A8DynamicLinearMethod, type(rotary), indexer,
                  ModelSlimCheckpoint):
        path = Path(inspect.getfile(value)).resolve()
        report["source_sha256"][str(path)] = digest(path)
    from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
    report["source_sha256"][inspect.getfile(AscendSFAIndexerBackend)] = digest(inspect.getfile(AscendSFAIndexerBackend))
    report["npu_initialized"] = torch_npu.npu._initialized
    report["process_group_initialized"] = torch.distributed.is_initialized()
    assert not report["process_group_initialized"]
    assert bool(args.device) or not report["npu_initialized"]
    report["status"] = "attention_only_complete" if args.device else "cpu_import_shape_pass"
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "layers": len(report["layers"]), "npu_initialized": report["npu_initialized"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/models/GLM-5.2-w4a8c8"))
    parser.add_argument("--raw", type=Path, default=BASE / "raw")
    parser.add_argument("--prefix", type=Path, default=Path("/workspace/offline/prefix8k-allocation-20261017/capture/case-00/main-prefix/rank-0.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", help="Explicit NPU only; omission performs CPU import/shape checks")
    parser.add_argument("--quota-mib", type=int, default=512)
    main(parser.parse_args())
