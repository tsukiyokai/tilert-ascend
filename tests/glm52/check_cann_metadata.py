"""CPU contracts for fixed-width CANN target metadata; no device precision."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def source_record(path, needles=()):
    path = Path(path)
    text = path.read_text()
    for needle in needles:
        if needle not in text:
            raise AssertionError(
                f"Source contract missing in {path}: {needle}"
            )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "anchors": [
            {"line": number, "text": line.strip()}
            for number, line in enumerate(text.splitlines(), 1)
            if any(needle in line for needle in needles)
        ],
    }


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
    from vllm.config.attention import AttentionConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import get_forward_context
    from vllm.platforms import current_platform
    from vllm_ascend import ascend_forward_context as context_module
    from vllm_ascend.attention import indexer, sfa_v1
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.mla_v1 import AscendMLAAttention
    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
    )
    from vllm_ascend.device.device_op import BaseDeviceAdaptor
    from vllm_ascend.ops import rotary_embedding as rope
    from vllm_ascend.worker.worker import NPUWorker

    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    torch.set_num_threads(4)
    assert not torch_npu.npu._initialized
    report["packages"] = {
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
    }
    try:
        AttentionConfig(indexer_kv_dtype="int8")
        accepted_before = True
    except ValueError:
        accepted_before = False
    current_platform.pre_register_and_update()
    assert AttentionConfig(indexer_kv_dtype="int8").indexer_kv_dtype == "int8"
    report["checks"].append({
        "name": "platform_registration_before_engine_args", "passed": True,
        "int8_accepted_before_registration": accepted_before,
        "npu_initialized": torch_npu.npu._initialized,
    })
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
    assert config.speculative_config is None
    assert config.parallel_config.tensor_parallel_size == 16
    assert config.parallel_config.prefill_context_parallel_size == 1
    assert config.parallel_config.decode_context_parallel_size == 1
    assert config.model_config.use_mla
    report["config"] = {
        "tp": 16, "speculative_config": None, "eager": True,
        "max_num_batched_tokens": 16, "capacity": 9344,
    }

    source_root = Path(inspect.getfile(sfa_v1)).parents[2]
    sources = [
        source_record(inspect.getfile(obj))
        for obj in (CannModel, sfa_v1, indexer, BaseDeviceAdaptor, NPUWorker,
                    type(current_platform))
    ]
    sources.append(source_record(
        source_root / "csrc/attention/sparse_flash_attention/op_kernel/"
        "arch22/sparse_flash_attention_service_vector_mla.h",
        ("s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size"
         " + runInfo.gS1Idx / constInfo.gSize + 1;",),
    ))
    sources.append(source_record(
        "/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/"
        "ops_transformer/ascendc/quant_lightning_indexer/op_kernel/"
        "arch22/quant_lightning_indexer_service_vector_arch22.h",
        ("cuRealAcSeq = info.actS2Size - "
         "(info.actS1Size - cuS1BeginIdxPerAiv);", "cuRealAcSeq += 1;"),
    ))
    report["sources"] = sources + [source_record(__file__)]
    name = "model.layers.4.self_attn.attn"

    with (
        set_current_vllm_config(config), torch.device("cpu"),
        patch.object(sfa_v1, "get_tensor_model_parallel_world_size",
                     return_value=1),
    ):
        rope.set_cos_and_sin(config, 1, 1, torch.bfloat16, torch.device("cpu"))
        rotary = rope.AscendRotaryEmbedding(
            256, 64, 9344,
            config.model_config.hf_text_config.rope_parameters["rope_theta"],
            False, torch.bfloat16,
        )
        assert rotary.cos_sin_cache.device.type == "cpu"
        attention = AscendMLAAttention(
            num_heads=4, scale=0.0625, qk_nope_head_dim=192,
            qk_rope_head_dim=64, v_head_dim=256, q_lora_rank=2048,
            kv_lora_rank=512,
            kv_b_proj=torch.nn.Linear(512, 1792, bias=False,
                                      dtype=torch.bfloat16),
            q_b_proj=torch.nn.Linear(2048, 1024, bias=False,
                                     dtype=torch.bfloat16),
            o_proj=torch.nn.Linear(1024, 6144, bias=False,
                                   dtype=torch.bfloat16),
            cache_config=config.cache_config, prefix=name,
            use_sparse=True, indexer=None, skip_topk=True, layer_name=name,
            topk_indices_buffer=torch.empty((16, 2048), dtype=torch.int32),
        )
        main_spec = AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16, num_heads=4,
        )
        builders = [([name], sfa_v1.AscendSFAMetadataBuilder(
            main_spec, [name], config, torch.device("cpu"),
        ))]
        for label, quantized in (("c8", True), ("bf16", False)):
            spec = AscendSFAIndexerCacheSpec(
                block_size=128, num_kv_heads=1, head_size=128,
                dtype=torch.int8 if quantized else torch.bfloat16,
                scale_dim=int(quantized), scale_dtype=torch.float16,
                cache_sparse_li_c8=quantized,
            )
            builders.append(([label], indexer.AscendSFAIndexerMetadataBuilder(
                spec, [label], config, torch.device("cpu"),
            )))

        model = object.__new__(CannModel)
        model.device = torch.device("cpu")
        model.capacity = 9344
        model.layout = {"capacity_tokens": 9344}
        model.builders = builders
        model.config = config
        model._allocate_inputs()
        assert model.input_ids.shape == (1,)
        for position in (0, 1, 8192, 8319):
            for width in (1, 2, 4):
                model.position = position
                metadata = model._metadata(width)
                main = metadata[name]
                slots = list(range(position, position + width))
                state = (AscendAttentionState.DecodeOnly if width == 1
                         else AscendAttentionState.ChunkedPrefill)
                assert main.attn_state == state
                assert main.seq_lens.tolist() == [position + width]
                assert main.cum_query_lens.tolist() == [width]
                assert main.query_start_loc.tolist() == [0, width]
                assert main.positions.tolist() == slots
                assert main.slot_mapping.tolist() == slots
                assert main.max_query_len == width
                assert main.num_actual_tokens == main.num_input_tokens == width
                assert main.num_decodes == int(width == 1)
                assert main.num_prefills == int(width > 1)
                assert main.num_decode_tokens == int(width == 1)
                assert model.seq_lens.shape == (1,)
                assert model.input_ids.shape == (width,)
                for entry in metadata.values():
                    assert entry.cos.shape == (width, 1, 1, 64)
                    assert torch.equal(entry.cos[:, 0, 0],
                                       rope._cos_cache[slots])
                    assert torch.equal(entry.sin[:, 0, 0],
                                       rope._sin_cache[slots])
                    assert entry.slot_mapping.tolist() == slots
                for label in ("c8", "bf16"):
                    entry = metadata[label]
                    assert entry.actual_seq_lengths_query.tolist() == [width]
                    assert entry.actual_seq_lengths_key.tolist() == [
                        position + width,
                    ]
                assert sfa_v1._int64_kv_slots(
                    main.slot_mapping, main,
                ).tolist() == slots
                total_kv = int(main.seq_lens[0])
                total_q = int(main.cum_query_lens[0])
                upper = [total_kv - total_q + row + 1
                         for row in range(total_q)]
                assert upper == list(range(position + 1, position + width + 1))
                selected = torch.full((width, 1, 2048), -1, dtype=torch.int32)
                for row, end in enumerate(upper):
                    count = min(2048, end)
                    selected[row, 0, :count] = torch.arange(end - count, end)
                    assert int(selected[row].max()) < end
                attention.impl._update_indexcache_topk_indices(selected)
                assert torch.equal(
                    attention.impl._get_indexcache_topk_indices(width),
                    selected,
                )
                check_dispatch(BaseDeviceAdaptor, torch, torch_npu, metadata,
                               main, selected, width)
                report["checks"].append({
                    "name": f"metadata.P{position}.M{width}", "passed": True,
                    "slots": slots, "kv_length": position + width,
                    "state": state.name, "query_length": width,
                    "causal_exclusive_upper": upper,
                    "shared_indices_shape": list(selected.shape),
                    "causal_scope": "source formula and CPU witness only",
                })

        model.model = CpuTarget(torch, get_forward_context)
        with (
            patch.object(torch.npu, "device", return_value=nullcontext()),
            patch.object(
                context_module, "get_tensor_model_parallel_world_size",
                return_value=16,
            ),
            patch.object(context_module, "get_dp_group",
                         return_value=SimpleNamespace(world_size=1)),
        ):
            model.position = 8192
            for width in (4, 2, 1, 4, 1):
                start = model.position
                value = model.verify(tuple(range(10, 10 + width)))
                assert value.last_hidden.shape == (width, 6144)
                assert value.logits.shape == (width, 154880)
                assert value.token_ids.tolist() == list(range(11, 11 + width))
                assert model.position == start + width
                model.truncate(start + 1)
                next_meta = model._metadata()
                assert next_meta[name].slot_mapping.tolist() == [start + 1]
                assert next_meta[name].seq_lens.tolist() == [start + 2]
            value = model.step(42)
            assert value.logits is None and value.token_ids.tolist() == [43]
        report["checks"].append({
            "name": "verify_dispatch_and_rollback", "passed": True,
            "scope": "real metadata/context; synthetic CPU model, no kernels",
        })
        position = model.position
        for length in (-1, position + 1, True, 1.5):
            rejects(lambda: model.truncate(length))
            assert model.position == position
        model.truncate(position)
        model.truncate(0)
        for tokens in ((), (1, 2, 3), (True,), (-1,), (154880,)):
            rejects(lambda: model.verify(tokens))
        model.position = model.capacity - 1
        for width in (2, 4):
            rejects(lambda: model.verify((1,) * width))
        model.position = model.capacity
        rejects(lambda: model.verify((1,)))
        model.reset()
        assert model.position == 0
        report["checks"].append({
            "name": "bounds_reset_and_invalid_tokens", "passed": True,
        })
    assert not torch_npu.npu._initialized
    report["npu_initialized"] = torch_npu.npu._initialized


class CpuTarget:
    """Synthetic rows exercise adapter dispatch without model computation."""

    def __init__(self, torch, get_context):
        self.torch = torch
        self.get_context = get_context

    def __call__(self, tokens, positions):
        assert self.get_context().num_tokens == len(tokens)
        assert len(positions) == len(tokens)
        hidden = self.torch.zeros((len(tokens), 6144),
                                  dtype=self.torch.bfloat16)
        hidden[:, 0] = tokens
        return hidden

    def compute_logits(self, hidden):
        logits = self.torch.zeros((len(hidden), 154880),
                                  dtype=self.torch.bfloat16)
        logits.scatter_(1, hidden[:, :1].long() + 1, 1)
        return logits


def check_dispatch(device, torch, torch_npu, metadata, main, selected, width):
    """Capture real wrapper arguments; device operators are replaced here."""
    calls = []

    def capture(**kwargs):
        calls.append(kwargs)
        return selected

    with (
        patch.object(torch_npu, "npu_quant_lightning_indexer", capture),
        patch.object(torch_npu, "npu_lightning_indexer",
                     side_effect=lambda **kw: (capture(**kw), None)),
        patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention",
                     capture, create=True),
    ):
        pages = main.block_table.shape[1]
        for label in ("c8", "bf16"):
            quantized = label == "c8"
            dtype = torch.int8 if quantized else torch.bfloat16
            query = torch.zeros((width, 32, 128), dtype=dtype)
            cache = (torch.zeros((pages, 128, 1, 128), dtype=dtype),)
            if quantized:
                cache += (
                    torch.ones((pages, 128, 1, 1), dtype=torch.float16),
                )
            device.indexer_select_post_process(
                query, torch.ones((width, 32), dtype=torch.float16),
                query.shape, torch.ones((width, 32), dtype=torch.bfloat16),
                cache, 0, 1, metadata[label],
                metadata[label].actual_seq_lengths_query,
                metadata[label].actual_seq_lengths_key,
                quantized, False, True,
            )
        latent = torch.zeros((pages, 128, 1, 512), dtype=torch.bfloat16)
        rope = torch.zeros((pages, 128, 1, 64), dtype=torch.bfloat16)
        device.execute_sparse_flash_attention_process(
            SimpleNamespace(scale=0.0625),
            torch.zeros((width, 4, 512), dtype=torch.bfloat16),
            torch.zeros((width, 4, 64), dtype=torch.bfloat16),
            (latent, rope), selected, main,
            main.cum_query_lens, main.seq_lens,
        )
    assert len(calls) == 3
    for call in calls:
        assert call["sparse_mode"] == 3
        assert call["layout_query"] == "TND"
        assert call["actual_seq_lengths_query"].tolist() == [width]
        lengths = call.get("actual_seq_lengths_key",
                           call.get("actual_seq_lengths_kv"))
        assert lengths.tolist() == main.seq_lens.tolist()
    assert calls[-1]["attention_mode"] == 2
    assert calls[-1]["layout_kv"] == "PA_BSND"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "installed CPU metadata API and source contracts",
        "precision_verified": False, "checks": [],
        "not_executed": ["NPUWorker", "checkpoint weight load",
                         "NPU kernels", "HCCL", "model precision", "timing"],
        "fixtures": ["CPU projection modules; TP query stub=1 in constructor",
                     "CPU target rows; TP/DP context query stubs=16/1",
                     "device operator capture stubs; no operator execution"],
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
