import hashlib
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import traceback

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
REPORT = Path("/tmp/cann-adapter-cpu-api-20261017.json")
result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "checks": [],
    "scope": (
        "CPU API verification with real installed packages and CPU tensors; "
        "parallel world-size stubs only; no checkpoint weights, "
        "NPU initialization or model forward"
    ),
}


def record(name, **data):
    result["checks"].append({"name": name, **data})
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"][-1]), flush=True)


# Keep imports staged to record each initialization boundary.
try:
    import torch
    import torch_npu

    torch.set_num_threads(1)
    record(
        "torch", torch=torch.__version__, torch_npu=torch_npu.__version__,
        initialized=torch_npu.npu._initialized,
    )
    assert not torch_npu.npu._initialized

    from vllm.engine.arg_utils import EngineArgs
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder
    from vllm_ascend.attention.indexer import AscendSFAIndexerMetadataBuilder
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.worker.worker import NPUWorker
    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
    )
    from vllm_ascend.patch.worker.patch_bind_kv_cache import (
        bind_kv_cache_to_layers,
    )
    from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding

    assert not torch_npu.npu._initialized
    record("imports", initialized=torch_npu.npu._initialized)
    objects = [
        EngineArgs, AscendCommonAttentionMetadata, AscendSFAMetadataBuilder,
        AscendSFAIndexerMetadataBuilder, set_ascend_forward_context,
        NPUWorker, AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
        bind_kv_cache_to_layers, AscendRotaryEmbedding,
    ]
    for value in objects:
        path = Path(inspect.getfile(inspect.unwrap(value)))
        record(
            "signature", object=value.__name__,
            signature=str(inspect.signature(value)), path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    args = EngineArgs(
        model="/workspace/models/GLM-5.2-w4a8c8", trust_remote_code=True,
        dtype="bfloat16", quantization="ascend", tensor_parallel_size=16,
        pipeline_parallel_size=1, data_parallel_size=1,
        distributed_executor_backend="external_launcher",
        attention_config={"indexer_kv_dtype": "int8"},
        kv_cache_dtype="auto", block_size=128,
        max_num_seqs=1, max_num_batched_tokens=16, max_model_len=9344,
        gpu_memory_utilization=0.85, enable_prefix_caching=False,
        enable_chunked_prefill=True, enforce_eager=True,
        async_scheduling=False,
        speculative_config=None, kv_transfer_config=None,
        additional_config={
            "enable_flashcomm1": False, "enable_dsa_cp": False,
            "enable_shared_expert_dp": False, "enable_fused_mc2": 0,
        },
    )
    record(
        "engine_args",
        attention_config_type=type(args.attention_config).__name__,
        initialized=torch_npu.npu._initialized,
    )
    assert not torch_npu.npu._initialized

    os.environ["WORLD_SIZE"] = "16"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    config = args.create_engine_config()
    record(
        "engine_config", architecture=config.model_config.architectures,
        model_type=config.model_config.hf_text_config.model_type,
        use_mla=config.model_config.use_mla,
        tp=config.parallel_config.tensor_parallel_size,
        use_v2=config.use_v2_model_runner,
        initialized=torch_npu.npu._initialized,
    )
    assert not torch_npu.npu._initialized

    from vllm.config import set_current_vllm_config
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.ops import rotary_embedding as rope_module
    from vllm_ascend.attention.sfa_v1 import _int64_kv_slots

    common = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([8193], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([8193], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([8193], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([8192], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([8193], dtype=torch.int32),
        _num_computed_tokens_cpu=torch.tensor([8192], dtype=torch.int32),
        num_reqs=1, num_actual_tokens=1, max_query_len=1, max_seq_len=8193,
        block_table_tensor=torch.arange(73, dtype=torch.int32).unsqueeze(0),
        slot_mapping=torch.tensor([8192], dtype=torch.int32),
        positions=torch.tensor([8192]), positions_cpu=torch.tensor([8192]),
        is_prefilling=torch.tensor([False]), decode_token_per_req=1,
        actual_seq_lengths_q=[1], attn_state=AscendAttentionState.DecodeOnly,
        num_input_tokens=1,
    )
    slots64 = _int64_kv_slots(common.slot_mapping, common)
    assert slots64.dtype == torch.int64 and slots64.tolist() == [8192]
    assert _int64_kv_slots(common.slot_mapping, common) is slots64
    record(
        "common_metadata", position=8192, seq_lens=common.seq_lens.tolist(),
        slot_dtype=str(common.slot_mapping.dtype),
        prolog_slot_dtype=str(slots64.dtype),
        initialized=torch_npu.npu._initialized,
    )

    context = {
        "model.layers.0.self_attn.attn": torch.nn.Module(),
        "model.layers.0.self_attn.indexer": torch.nn.Module(),
    }
    caches = {
        next(iter(context)): (
            torch.empty((1, 128, 1, 512), dtype=torch.bfloat16),
            torch.empty((1, 128, 1, 64), dtype=torch.bfloat16),
        ),
    }
    name = list(context)[1]
    caches[name] = (
        torch.empty((1, 128, 1, 128), dtype=torch.int8),
        torch.empty((1, 128, 1, 1), dtype=torch.float16),
    )
    bind_kv_cache_to_layers(caches, context)
    assert all(
        context[name].kv_cache is cache for name, cache in caches.items()
    )
    record(
        "tuple_cache_binding", identity_preserved=True,
        shapes={
            name: [list(x.shape) for x in cache]
            for name, cache in caches.items()
        },
    )

    with set_current_vllm_config(config), torch.device("cpu"):
        rope_module.set_cos_and_sin(
            config, 1, 1, torch.bfloat16, torch.device("cpu"),
        )
        rotary = AscendRotaryEmbedding(
            256, 64, 9344,
            config.model_config.hf_text_config.rope_parameters["rope_theta"],
            False, torch.bfloat16,
        )
        cos, sin = rope_module.get_cos_and_sin_mla(
            torch.tensor([8192]), use_cache=True,
        )
        expected_cos = rope_module._cos_cache[8192].view(1, 1, 1, 64)
        expected_sin = rope_module._sin_cache[8192].view(1, 1, 1, 64)
        assert (
            torch.equal(cos, expected_cos)
            and torch.equal(sin, expected_sin)
        )
        record(
            "rope_lookup_cpu", cos_shape=list(cos.shape),
            cos_dtype=str(cos.dtype), position=8192,
            cpu_tensors=all(
                x.device.type == "cpu"
                for x in (cos, sin, rotary.cos_sin_cache)
            ),
            initialized=torch_npu.npu._initialized,
        )
    assert not torch_npu.npu._initialized

    from unittest.mock import patch
    from types import SimpleNamespace
    from vllm_ascend.attention.mla_v1 import AscendMLAAttention
    from vllm_ascend.attention.sfa_v1 import AscendSFABackend
    from vllm_ascend import ascend_forward_context as forward_module
    from vllm.forward_context import get_forward_context

    mla_name = "model.layers.4.self_attn.attn"
    index_name = "model.layers.4.self_attn.indexer"
    with (
        set_current_vllm_config(config),
        torch.device("cpu"),
        patch(
            "vllm_ascend.attention.sfa_v1."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        torch.set_default_dtype(torch.bfloat16)
        kv_b = torch.nn.Linear(512, 1792, bias=False, dtype=torch.bfloat16)
        q_b = torch.nn.Linear(2048, 1024, bias=False, dtype=torch.bfloat16)
        o_proj = torch.nn.Linear(1024, 6144, bias=False, dtype=torch.bfloat16)
        attention = AscendMLAAttention(
            num_heads=4, scale=0.0625, qk_nope_head_dim=192,
            qk_rope_head_dim=64, v_head_dim=256, q_lora_rank=2048,
            kv_lora_rank=512, kv_b_proj=kv_b, cache_config=config.cache_config,
            prefix=mla_name,
            use_sparse=True, indexer=None,
            topk_indices_buffer=torch.empty((16, 2048), dtype=torch.int32),
            q_b_proj=q_b, o_proj=o_proj, skip_topk=True, layer_name=mla_name,
        )
        spec = AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, num_heads=4,
        )
        builder = AscendSFAMetadataBuilder(
            spec, [mla_name], config, torch.device("cpu"),
        )
        metadata = builder.build(0, common)
        assert metadata.num_decodes == 1 and metadata.num_prefills == 0
        assert metadata.num_decode_tokens == 1
        assert metadata.seq_lens.tolist() == [8193]
        assert metadata.cum_query_lens.tolist() == [1]
        assert metadata.slot_mapping.tolist() == [8192]
        record(
            "sfa_builder_cpu", num_decodes=metadata.num_decodes,
            num_prefills=metadata.num_prefills,
            num_decode_tokens=metadata.num_decode_tokens,
            seq_lens=metadata.seq_lens.tolist(),
            query_lengths=metadata.cum_query_lens.tolist(),
            cos_shape=list(metadata.cos.shape),
            sin_shape=list(metadata.sin.shape),
            slot_dtype=str(metadata.slot_mapping.dtype),
            initialized=torch_npu.npu._initialized,
            stub=(
                "Only SFA TP world-size query returns 1; real CPU "
                "AscendMLAAttention uses unused CPU nn.Linear projections, "
                "no checkpoint weights"
            ),
        )

        index_spec = AscendSFAIndexerCacheSpec(
            block_size=128, num_kv_heads=1, head_size=128,
            dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
        )
        index_builder = AscendSFAIndexerMetadataBuilder(
            index_spec, [index_name], config, torch.device("cpu"),
        )
        index_meta = index_builder.build(0, common)
        assert index_meta.actual_seq_lengths_query.tolist() == [1]
        assert index_meta.actual_seq_lengths_key.tolist() == [8193]
        record(
            "indexer_builder_cpu",
            query_lengths=index_meta.actual_seq_lengths_query.tolist(),
            key_lengths=index_meta.actual_seq_lengths_key.tolist(),
            slots=index_meta.slot_mapping.tolist(),
            cos_shape=list(index_meta.cos.shape),
            initialized=torch_npu.npu._initialized,
        )

        import sys

        sys.path.insert(0, "/workspace/src/TileRT-Ascend")
        from tilert.models.glm_5_2_ascend.cann_model import CannModel

        adapter = object.__new__(CannModel)
        adapter.device = torch.device("cpu")
        adapter.layout = {"capacity_tokens": 9344}
        adapter.builders = [
            ([mla_name], builder), ([index_name], index_builder),
        ]
        adapter._allocate_inputs()
        records = []
        for position in (0, 1, 8192, 8193):
            adapter.position = position
            built = adapter._metadata()
            main_meta, idx_meta = built[mla_name], built[index_name]
            assert main_meta.seq_lens.tolist() == [position + 1]
            assert main_meta.positions.tolist() == [position]
            assert main_meta.slot_mapping.tolist() == [position]
            assert main_meta.cum_query_lens.tolist() == [1]
            assert main_meta.num_decodes == 1 and main_meta.num_prefills == 0
            assert idx_meta.actual_seq_lengths_key.tolist() == [position + 1]
            assert idx_meta.actual_seq_lengths_query.tolist() == [1]
            assert torch.equal(
                main_meta.cos,
                rope_module._cos_cache[position].view(1, 1, 1, 64),
            )
            assert torch.equal(
                idx_meta.sin,
                rope_module._sin_cache[position].view(1, 1, 1, 64),
            )
            assert _int64_kv_slots(
                main_meta.slot_mapping, main_meta,
            ).tolist() == [position]
            records.append({
                "position": position, "kv_length": position + 1,
                "query_length": 1, "decode_count": main_meta.num_decodes,
            })
        path = Path(inspect.getfile(CannModel))
        record(
            "actual_cann_metadata_cpu", cases=records, path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            initialized=torch_npu.npu._initialized,
            scope=(
                "Actual _allocate_inputs and _metadata methods; constructor "
                "bypassed, device=cpu, real builders, no weights or forward"
            ),
        )
        path = Path(
            "/workspace/src/TileRT-Ascend/tests/glm52/run_cann_model.py"
        )
        record(
            "runner_source", path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

        forward_module.set_mc2_tokens_capacity(config, 1, 1)
        with (
            patch.object(
                forward_module, "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch.object(
                forward_module, "get_dp_group",
                return_value=SimpleNamespace(world_size=1),
            ),
        ):
            with set_ascend_forward_context(
                {mla_name: metadata, index_name: index_meta}, config,
                num_tokens=1, num_actual_tokens=1, model_instance=attention,
            ):
                current = get_forward_context()
                assert current.model_instance is attention
                record(
                    "forward_context_cpu", num_tokens=current.num_tokens,
                    padded_num_tokens=current.padded_num_tokens,
                    is_first_layer=current.is_first_layer,
                    initialized=torch_npu.npu._initialized,
                    stub=(
                        "Only TP world-size and DP group world-size queries "
                        "return 1; no communication or model forward"
                    ),
                )
    assert not torch_npu.npu._initialized
    result["not_executed"] = [
        "NPUWorker constructor/init_device/load_model",
        "checkpoint weight loading",
        "HCCL TP16 execution",
        "NPU tensors/kernels or model forward",
        "numerical parity and performance",
    ]
    result["stubs"] = [
        (
            "vllm_ascend.attention.sfa_v1."
            "get_tensor_model_parallel_world_size returns 1"
        ),
        (
            "ascend_forward_context.get_tensor_model_parallel_world_size "
            "returns 1"
        ),
        "ascend_forward_context.get_dp_group returns world_size=1",
    ]
    result["cpu_fixture"] = [
        (
            "Real AscendMLAAttention with unused CPU torch.nn.Linear "
            "projection modules"
        ),
        (
            "RoPE table uses checkpoint theta=8000000 and truncated "
            "9344-position capacity"
        ),
        (
            "Actual CannModel._allocate_inputs/_metadata on device=cpu; "
            "constructor bypassed"
        ),
        "No distributed process group initialized",
    ]
    result["status"] = "pass"
except BaseException as exc:
    result["status"] = "failure"
    record(
        "exception", type=type(exc).__name__, message=str(exc),
        traceback=traceback.format_exc(),
    )
finally:
    if "torch_npu" in globals():
        result["npu_initialized_final"] = torch_npu.npu._initialized
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    print("REPORT", str(REPORT), result.get("status"), flush=True)
