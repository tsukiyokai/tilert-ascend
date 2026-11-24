#!/usr/bin/env python3
"""Exercise actual vendor constructors on CPU and record the device boundary."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import resource
import tempfile
import traceback

import torch
import torch_npu


MODEL_CONFIG = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "torch_dtype": "bfloat16",
    "hidden_size": 6144,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 78,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "rms_norm_eps": 1e-5,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    "use_index_cache": True,
    "indexer_types": ["full"] * 78,
    "max_position_embeddings": 8192,
    "vocab_size": 154880,
    "rope_theta": 10000.0,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
}


def source_identity(value):
    path = Path(inspect.getfile(value))
    return {"class": f"{value.__module__}.{value.__name__}", "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=("worker", "indexer"), default="worker")
    args = parser.parse_args()
    report = {
        "scope": "actual constructors with CPU tensors and a real TP1 Gloo group",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "initialized_before": torch.npu.is_initialized(),
        "component": args.component,
        "synthetic_model_config": MODEL_CONFIG, "stages": [],
    }

    def refuse_npu_init(*unused_args, **unused_kwargs):
        raise RuntimeError("CPU constructor audit refuses NPU initialization")

    torch.npu._lazy_init = refuse_npu_init
    torch_npu._C._npu_init = refuse_npu_init

    def stage(name, action):
        item = {"name": name}
        try:
            value, detail = action()
            item.update(success=True, **detail)
        except Exception:
            item.update(success=False, error=traceback.format_exc())
            report["stages"].append(item)
            print(json.dumps(item), flush=True)
            raise
        report["stages"].append(item)
        print(json.dumps(item), flush=True)
        return value

    try:
        from vllm.platforms import current_platform

        stage("platform_registration", lambda: (
            current_platform.pre_register_and_update(),
            source_identity(type(current_platform)),
        ))

        def make_metadata():
            from vllm.forward_context import (
                ForwardContext, get_forward_context, override_forward_context,
            )
            from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata
            from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata

            rows = 2
            lengths = torch.tensor([128], dtype=torch.int32)
            query = torch.tensor([rows], dtype=torch.int32)
            sine = torch.zeros(rows, 1, 64, dtype=torch.bfloat16)
            cosine = torch.ones_like(sine)
            main = AscendSFAMetadata(
                rows, torch.tensor([126, 127], dtype=torch.int32),
                lengths, lengths.clone(), query,
                torch.tensor([[0]], dtype=torch.int32), sine, cosine,
                num_input_tokens=rows, block_size=128,
            )
            index = AscendSFAIndexerMetadata(
                rows, torch.tensor([254, 255], dtype=torch.int32),
                lengths.clone(), query.clone(),
                torch.tensor([[1]], dtype=torch.int32), sine.clone(), cosine.clone(),
                block_size=128, actual_seq_lengths_query=query.clone(),
                actual_seq_lengths_key=lengths.clone(),
            )
            context = ForwardContext(
                no_compile_layers={}, attn_metadata={"main": main, "index": index},
                slot_mapping={"main": main.slot_mapping, "index": index.slot_mapping},
            )
            with override_forward_context(context):
                assert get_forward_context() is context
                assert context.attn_metadata["index"].slot_mapping is index.slot_mapping
                assert main.slot_mapping.data_ptr() != index.slot_mapping.data_ptr()
            return None, {"main": source_identity(type(main)),
                          "index": source_identity(type(index)),
                          "context": source_identity(type(context)),
                          "independent_cache_metadata": True, "rows": rows}

        stage("actual_metadata_and_forward_context", make_metadata)
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs

        with tempfile.TemporaryDirectory(prefix="vendor-hook-cpu-") as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(MODEL_CONFIG))
            prefix = "model.layers.0.self_attn"
            quant_description = {
                f"{prefix}.{name}.weight": "W8A8_DYNAMIC"
                for name in ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj",
                             "o_proj", "indexer.wq_b")
            }
            (path / "quant_model_description.json").write_text(
                json.dumps(quant_description))

            def make_config():
                config = EngineArgs(
                    model=str(path), dtype="bfloat16", quantization="ascend",
                    skip_tokenizer_init=True, load_format="dummy",
                    tensor_parallel_size=1, pipeline_parallel_size=1,
                    data_parallel_size=1, max_model_len=8192,
                    max_num_seqs=1, max_num_batched_tokens=32,
                    block_size=128, kv_cache_dtype="auto",
                    attention_config={"indexer_kv_dtype": "int8"},
                    enforce_eager=True, enable_prefix_caching=False,
                    enable_chunked_prefill=True, async_scheduling=False,
                    additional_config={"enable_mlapo": False,
                                       "enable_cpu_binding": False,
                                       "ascend_log_path": "/tmp/vendor-hook-log"},
                ).create_engine_config()
                return config, {
                    "config": source_identity(type(config)),
                    "hf_config": source_identity(type(config.model_config.hf_config)),
                    "tp_size": config.parallel_config.tensor_parallel_size,
                }

            config = stage("engine_configuration", make_config)
            with set_current_vllm_config(config):
                from vllm.distributed.parallel_state import (
                    init_distributed_environment, initialize_model_parallel,
                )

                def make_groups():
                    init_distributed_environment(
                        world_size=1, rank=0, local_rank=0, backend="gloo",
                        distributed_init_method=f"file://{path / 'rendezvous'}",
                    )
                    initialize_model_parallel(backend="gloo")
                    return None, {"world_size": torch.distributed.get_world_size(),
                                  "backend": torch.distributed.get_backend()}

                stage("real_tp1_groups", make_groups)
                if args.component == "indexer":
                    def make_indexer():
                        from vllm.model_executor.models.deepseek_v2 import Indexer
                        from vllm_ascend.ops.mla import IndexerWrapper
                        from vllm_ascend.quantization.configs.modelslim_config import (
                            AscendModelSlimConfig,
                        )

                        previous = torch.get_default_dtype()
                        torch.set_default_dtype(torch.bfloat16)
                        try:
                            with torch.device("cpu"):
                                source = Indexer(
                                    config, config.model_config.hf_config, 6144, 2048,
                                    AscendModelSlimConfig(quant_description),
                                    config.cache_config,
                                    torch.zeros(32, 2048, dtype=torch.int32),
                                    prefix=f"{prefix}.indexer",
                                )
                                wrapper = IndexerWrapper(source, 64)
                        finally:
                            torch.set_default_dtype(previous)
                        backend = wrapper.impl
                        assert wrapper.wq_b is source.wq_b is backend.wq_b
                        assert wrapper.k_cache is source.k_cache is backend.k_cache
                        assert not hasattr(wrapper, "q_hadamard")
                        return wrapper, {
                            "source": source_identity(type(source)),
                            "wrapper": source_identity(type(wrapper)),
                            "backend": source_identity(type(backend)),
                            "quant_method": source_identity(type(backend.wq_b.quant_method)),
                            "quant_scheme": source_identity(type(backend.wq_b.quant_method.quant_method)),
                            "wq_shape_before_postprocess": list(backend.wq_b.weight.shape),
                            "scale_shape_before_postprocess": list(backend.wq_b.weight_scale.shape),
                            "scale_dtype": str(backend.wq_b.weight_scale.dtype),
                            "wk_shape": list(backend.wk_weights_proj.weight.shape),
                            "shared_weight_and_cache_objects": True,
                            "c8": backend.enable_sparse_li_c8,
                            "rope_neox": backend.is_rope_neox_style,
                            "parameter_bytes": sum(p.numel() * p.element_size()
                                                   for p in wrapper.parameters()),
                        }

                    stage("actual_indexer_constructor", make_indexer)
                    report["completed_scope"] = "replicated indexer constructor before worker patches or weight postprocessing"
                    return 0
                from vllm_ascend.utils import adapt_patch, register_ascend_customop

                stage("worker_registration", lambda: (
                    (adapt_patch(), register_ascend_customop(config)), {},
                ))
                from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
                from vllm_ascend.quantization.configs.modelslim_config import (
                    AscendModelSlimConfig,
                )

                def make_attention():
                    quant = AscendModelSlimConfig(quant_description)
                    previous = torch.get_default_dtype()
                    torch.set_default_dtype(torch.bfloat16)
                    try:
                        with torch.device("cpu"):
                            layer = DeepseekV2MLAAttention(
                                config, config.model_config.hf_config,
                                6144, 64, 192, 64, 256, 2048, 512,
                                cache_config=config.cache_config, quant_config=quant,
                                prefix=prefix, reduce_results=False,
                                topk_indices_buffer=torch.zeros(32, 2048,
                                                               dtype=torch.int32),
                            )
                    finally:
                        torch.set_default_dtype(previous)
                    impl = layer.mla_attn.mla_attn.impl
                    return layer, {
                        "outer": source_identity(type(layer)),
                        "wrapper": source_identity(type(layer.mla_attn)),
                        "implementation": source_identity(type(impl)),
                        "indexer": source_identity(type(impl.indexer)),
                        "indexer_backend": source_identity(type(impl.indexer.impl)),
                        "tp_size": impl.tp_size, "heads": impl.local_num_heads,
                        "preprocess": str(impl.preprocess_type),
                        "parameter_bytes": sum(p.numel() * p.element_size()
                                               for p in layer.parameters()),
                    }

                layer = stage("actual_attention_constructor", make_attention)
                impl = layer.mla_attn.mla_attn.impl

                def profiling_forward():
                    hidden = torch.ones(2, 6144, dtype=torch.bfloat16)
                    output = torch.full_like(hidden, 7)
                    result = impl.forward(impl.layer_name, hidden, (), None, output)
                    assert result is output and torch.count_nonzero(output) == 0
                    return None, {"rows": 2, "metadata": None,
                                  "in_place_zero_output": True}

                stage("vendor_profiling_forward", profiling_forward)
                stage("sfa_weight_postprocessing", lambda: (
                    impl.process_weights_after_loading(torch.bfloat16), {},
                ))
    except Exception:
        report["stopped_at"] = report["stages"][-1]["name"] if report["stages"] else "imports"
        report["error"] = traceback.format_exc()
    finally:
        if torch.distributed.is_initialized():
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment, destroy_model_parallel,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        report["initialized_after"] = torch.npu.is_initialized()
        report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["memory_snapshot"] = Path("/proc/self/smaps_rollup").read_text()
        report["cpu_stages_completed"] = [
            item["name"] for item in report["stages"] if item["success"]
        ]
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int("stopped_at" in report or report["initialized_after"])


if __name__ == "__main__":
    raise SystemExit(main())
