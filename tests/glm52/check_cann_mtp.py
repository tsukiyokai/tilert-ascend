"""Installed CPU API contracts for CannMtp; no checkpoint payload or NPU."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid state was accepted")


def run(checkpoint, report):
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ.update(WORLD_SIZE="16", RANK="0", LOCAL_RANK="0")

    import torch
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.model_loader import utils as loader_utils
    from vllm.model_executor.model_loader.default_loader import (
        DefaultModelLoader,
    )
    from vllm.model_executor.models.deepseek_mtp import (
        DeepSeekMultiTokenPredictorLayer,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm.platforms import current_platform
    from vllm_ascend import ascend_forward_context as context_module
    from vllm_ascend.attention import sfa_v1
    from vllm_ascend.attention.mla_v1 import AscendMLAAttention
    from vllm_ascend.models.deepseek_mtp import AscendDeepSeekMTP
    from vllm_ascend.ops import rotary_embedding as rope_module

    from tilert.models.glm_5_2_ascend import cann_mtp
    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    torch.set_num_threads(4)
    assert not torch_npu.npu._initialized
    current_platform.pre_register_and_update()
    config = EngineArgs(
        model=str(checkpoint), trust_remote_code=True, dtype="bfloat16",
        quantization="ascend", tensor_parallel_size=16,
        pipeline_parallel_size=1, data_parallel_size=1,
        distributed_executor_backend="external_launcher",
        attention_config={"indexer_kv_dtype": "int8"},
        kv_cache_dtype="auto", block_size=128,
        max_num_seqs=1, max_num_batched_tokens=16, max_model_len=9344,
        gpu_memory_utilization=0.85, enable_prefix_caching=False,
        enable_chunked_prefill=True, enforce_eager=True,
        async_scheduling=False, speculative_config=None,
        kv_transfer_config=None,
        additional_config={
            "enable_flashcomm1": False, "enable_dsa_cp": False,
            "enable_shared_expert_dp": False, "enable_fused_mc2": 0,
        },
    ).create_engine_config()
    config.compilation_config.static_forward_context["target-sentinel"] = 1
    target = SimpleNamespace(
        config=config, device=torch.device("cpu"), capacity=9344,
        layout={"capacity_tokens": 9344}, checkpoint=str(checkpoint),
        position=4, attention_kernel="sfa", fp32_head=False,
        model=SimpleNamespace(model=SimpleNamespace(
            embed_tokens=torch.nn.Embedding(8, 8),
        )),
        views={name: torch.zeros((73, 128, 1, size), dtype=torch.bfloat16)
               for name, size in (("mla_latent.78", 512),
                                  ("mla_rope.78", 64), ("index.78", 128))},
    )
    calls = []

    def create_model(loader, candidate_config, model_config, prefix=""):
        assert candidate_config.model_config.runner_type == "draft"
        architecture, _ = loader_utils.get_model_architecture(model_config)
        assert architecture is AscendDeepSeekMTP
        assert prefix == ""
        calls.append("official_create_model_boundary")
        return make_model_double(
            torch, candidate_config, get_forward_context,
            AscendMLAAttention, DeepseekV32IndexerCache, rope_module,
            cann_mtp,
        )

    def load_weights(loader, model, model_config):
        sources = [DefaultModelLoader.Source(
            model_config.model, model_config.revision,
            allow_patterns_overrides=model.allow_patterns_overrides,
        ), *model.secondary_weights]
        files = []
        for source in sources:
            _, paths, safetensors, _ = loader._prepare_weights(
                source.model_or_path, source.subfolder, source.revision,
                source.fall_back_to_pt, source.allow_patterns_overrides,
            )
            assert safetensors
            files.extend(Path(path).name for path in paths)
        assert sorted(files) == sorted(cann_mtp.weight_sources(checkpoint))
        report["selected_files"] = files
        model.has_own_lm_head = True
        calls.append("weight_payload_boundary")

    def postprocess(model, model_config, device):
        assert model.has_own_lm_head
        assert device.type == "cpu"
        assert model.model.embed_tokens is not target.model.model.embed_tokens
        calls.append("official_postprocess_boundary")

    with (
        patch.object(torch.npu, "device", return_value=nullcontext()),
        patch.object(DefaultModelLoader, "create_model", create_model),
        patch.object(DefaultModelLoader, "load_weights", load_weights),
        patch.object(loader_utils, "process_weights_after_loading",
                     postprocess),
        patch.object(sfa_v1, "get_tensor_model_parallel_world_size",
                     return_value=1),
        patch.object(context_module, "get_tensor_model_parallel_world_size",
                     return_value=16),
        patch.object(context_module, "get_dp_group",
                     return_value=SimpleNamespace(world_size=1)),
    ):
        draft = cann_mtp.CannMtp(target)
        assert calls == ["official_create_model_boundary",
                         "weight_payload_boundary",
                         "official_postprocess_boundary"]
        assert (draft.model.model.embed_tokens
                is target.model.model.embed_tokens)
        assert draft.model.lm_head is draft.model.own_head
        assert not draft.model.training
        add_check(report, "actual_constructor_loader_sequence_and_sharing")

        spec = draft.config.speculative_config
        assert spec.method == "mtp" and spec.num_speculative_tokens == 3
        assert spec.draft_tensor_parallel_size == 16
        assert spec.draft_model_config.runner_type == "draft"
        assert spec.draft_model_config.quantization == "ascend"
        assert config.speculative_config is None
        assert config.model_config.runner_type != "draft"
        assert config.model_config.hf_config.model_type == "glm_moe_dsa"
        assert config.compilation_config.static_forward_context == {
            "target-sentinel": 1,
        }
        assert draft.config.quant_config is not config.quant_config
        add_check(report, "real_speculative_config_and_target_isolation")

        context = draft.config.compilation_config.static_forward_context
        assert context[cann_mtp.MAIN_CACHE].kv_cache[0] is target.views[
            "mla_latent.78"
        ]
        assert context[cann_mtp.INDEX_CACHE].kv_cache == (
            target.views["index.78"],
        )
        prefix = {name: torch.ones_like(value[:3])
                  for name, value in draft.cache.items()}
        draft.inject(prefix)
        for name, destination in draft.cache.items():
            assert torch.equal(destination[:3], prefix[name])
            prefix[name].zero_()
            assert torch.all(destination[:3] == 1)
        assert draft.position == 3 and target.position == 4
        add_check(report, "raw_tuple_binding_and_P_minus_one_injection")

        for position in (0, 1, 8192):
            for width in (1, 2, 4):
                draft.position = position
                parent = torch.zeros((width, 6144), dtype=torch.bfloat16)
                result = draft.forward(tuple(range(width)), parent)
                assert draft.position == position + width
                assert result.start_position == position
                assert result.token_ids.tolist() == list(range(width))
                assert result.pre_norm_hidden.shape == (width, 6144)
                assert torch.equal(result.recycle_hidden,
                                   result.pre_norm_hidden + 2)
                assert result.indices.shape == (width, 2048)
                metadata = draft.model.seen_metadata
                main = metadata[cann_mtp.MAIN_CACHE]
                assert main.slot_mapping.tolist() == list(
                    range(position, position + width)
                )
                assert main.seq_lens.tolist() == [position + width]
                assert main.cum_query_lens.tolist() == [width]
                assert main.attn_state.name == "SpecDecoding"
                assert main.num_decode_tokens == width
                assert main.num_prefills == 0
                for item in metadata.values():
                    assert item.cos.shape == (width, 1, 1, 64)
                assert target.position == 4
                add_check(report,
                          f"metadata_and_dispatch.P{position}.M{width}")

        draft.position = 8192
        seed = draft.forward((5,), torch.zeros((1, 6144),
                                              dtype=torch.bfloat16))
        saved_indices = seed.indices.clone()
        chain = draft.forward((6,), seed.recycle_hidden,
                              external_indices=seed.indices,
                              full_logits=False)
        assert chain.logits is None
        assert draft.model.model.skip_topk
        assert torch.equal(chain.indices, saved_indices)
        assert torch.equal(seed.indices, saved_indices)
        draft.forward((7,), chain.recycle_hidden)
        assert not draft.model.model.skip_topk
        assert torch.equal(seed.indices, saved_indices)
        add_check(report, "explicit_chain_indices_and_cloned_output")

        graph = SimpleNamespace(forward=Mock(return_value=(
            seed.pre_norm_hidden, seed.recycle_hidden, seed.logits,
            seed.token_ids, seed.indices,
        )), close=Mock())
        draft.graph = graph
        result = draft.forward((6,), seed.recycle_hidden,
                               external_indices=seed.indices)
        graph.forward.assert_called_once()
        assert graph.forward.call_args.args[1] is seed.recycle_hidden
        assert result.recycle_hidden is seed.recycle_hidden
        draft.forward((7,), seed.recycle_hidden)
        draft.forward((1, 2, 3, 4), torch.zeros((4, 6144),
                                             dtype=torch.bfloat16))
        assert graph.forward.call_count == 1
        draft.close()
        graph.close.assert_called_once()
        assert draft.graph is None
        add_check(report, "chain_graph_dispatch_and_eager_seed_conditioned")

        for length in (-1, draft.position + 1, True, 1.5):
            rejects(lambda: draft.truncate(length))
        draft.truncate(2)
        result = draft.forward((8,), torch.zeros((1, 6144),
                                                dtype=torch.bfloat16))
        assert result.start_position == 2 and draft.position == 3
        for tokens in ((), (1, 2, 3), (True,), (-1,), (154880,)):
            rejects(lambda: draft.forward(tokens, seed.recycle_hidden))
        rejects(lambda: draft.forward((1,), seed.recycle_hidden.float()))
        rejects(lambda: draft.forward((1,), seed.recycle_hidden,
                                      external_indices=seed.indices[:, :1]))
        draft.position = draft.capacity
        rejects(lambda: draft.forward((1,), seed.recycle_hidden))
        bad_prefix = {name: value[:4] for name, value in draft.cache.items()}
        rejects(lambda: draft.inject(bad_prefix))
        draft.reset()
        assert draft.position == 0 and not draft.model.model.skip_topk
        add_check(report, "truncate_reset_capacity_and_invalid_inputs")

    assert not torch_npu.npu._initialized
    report["npu_initialized"] = bool(torch_npu.npu._initialized)
    report["packages"] = {"torch": torch.__version__,
                          "torch_npu": torch_npu.__version__}
    paths = {Path(inspect.getfile(value)).resolve() for value in (
        cann_mtp, CannModel, AscendDeepSeekMTP, DefaultModelLoader,
        type(spec), sfa_v1, DeepSeekMultiTokenPredictorLayer,
    )}
    paths.add(Path(__file__).resolve())
    report["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def make_model_double(torch, config, get_context, attention_type,
                      index_cache_type, rope, module):
    """Synthetic model outputs with real installed cache/metadata modules."""
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(config), torch.device("cpu"):
        rope.set_cos_and_sin(config, 1, 1, torch.bfloat16, torch.device("cpu"))
        rope.AscendRotaryEmbedding(
            256, 64, 9344,
            config.model_config.hf_text_config.rope_parameters["rope_theta"],
            False, torch.bfloat16,
        )
        attention = attention_type(
            num_heads=4, scale=0.0625, qk_nope_head_dim=192,
            qk_rope_head_dim=64, v_head_dim=256, q_lora_rank=2048,
            kv_lora_rank=512,
            kv_b_proj=torch.nn.Linear(512, 1792, bias=False,
                                      dtype=torch.bfloat16),
            q_b_proj=torch.nn.Linear(2048, 1024, bias=False,
                                     dtype=torch.bfloat16),
            o_proj=torch.nn.Linear(1024, 6144, bias=False,
                                   dtype=torch.bfloat16),
            cache_config=config.cache_config, prefix=module.MAIN_CACHE,
            use_sparse=True, indexer=None, skip_topk=True,
            layer_name=module.MAIN_CACHE,
            topk_indices_buffer=torch.empty((16, 2048), dtype=torch.int32),
        )
        index_cache_type(128, torch.bfloat16, module.INDEX_CACHE,
                         config.cache_config)

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 8)
            self.layers = {"78": SimpleNamespace(mtp_block=SimpleNamespace(
                self_attn=SimpleNamespace(mla_attn=SimpleNamespace(
                    topk_indices_buffer=attention.impl.topk_indices_buffer,
                )),
            ))}
            self.skip_topk = False

        def set_skip_topk(self, skip):
            self.skip_topk = skip

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Predictor()
            self.own_head = torch.nn.Linear(4, 4)
            self.lm_head = self.own_head

        def forward(self, input_ids, positions, hidden_states, spec_step_idx):
            context = get_context()
            assert context.is_draft_model and context.model_instance is self
            assert context.num_tokens == len(input_ids)
            assert spec_step_idx == 0
            self.seen_metadata = context.attn_metadata
            self.pre_norm = hidden_states + input_ids[:, None]
            if not self.model.skip_topk:
                attention.impl.topk_indices_buffer[:len(input_ids)] = (
                    positions[:, None]
                )
            return self.pre_norm, self.pre_norm + 2

        def compute_logits(self, hidden, spec_step_idx):
            assert hidden is self.pre_norm and spec_step_idx == 0
            logits = torch.zeros((len(hidden), 16))
            logits.scatter_(1, hidden[:, :1].long().remainder(16), 1)
            return logits

    return Model()


def add_check(report, name):
    report["checks"].append({"name": name, "passed": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "installed CPU config, loader selection, metadata and state",
        "precision_verified": False, "checks": [],
        "not_executed": ["checkpoint tensor payload", "MTP model arithmetic",
                         "NPU initialization", "NPU kernels", "HCCL",
                         "model precision", "timing"],
        "fixtures": ["CPU model rows and postprocess boundary stub",
                     "graph dispatch stub; no graph capture",
                     "CPU projection modules and TP constructor query=1",
                     "TP/DP forward context queries=16/1",
                     "NPU device context replaced with nullcontext"],
    }
    try:
        run(args.checkpoint, report)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"],
                          "checks": len(report["checks"]),
                          "output": str(args.output)}))


if __name__ == "__main__":
    main()
