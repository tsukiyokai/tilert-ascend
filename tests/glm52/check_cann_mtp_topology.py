import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def run(checkpoint, tp_size):
    os.environ.update(WORLD_SIZE=str(tp_size), RANK="0", LOCAL_RANK="0")
    import torch
    import torch_npu
    from vllm_ascend.attention import indexer, sfa_v1
    from tilert.models.glm_5_2_ascend import cann_model
    from tilert.models.glm_5_2_ascend.cann_model import create_config
    from tilert.models.glm_5_2_ascend.cann_mtp import (
        CannMtp, INDEX_CACHE, MAIN_CACHE,
    )

    config = create_config(checkpoint, expert_parallel=True, tp_size=tp_size)
    config.compilation_config.static_forward_context["target"] = object()
    target = SimpleNamespace(
        config=config, device=torch.device("cpu"), capacity=9344,
        layout={"capacity_tokens": 9344}, checkpoint=str(checkpoint),
        attention_kernel="sfa", hoist_metadata=True,
        views={name: torch.zeros(73, 128, 1, size, dtype=torch.bfloat16)
               for name, size in (("mla_latent.78", 512),
                                  ("mla_rope.78", 64), ("index.78", 128))},
    )

    def load(draft):
        context = draft.config.compilation_config.static_forward_context
        context.update({name: SimpleNamespace(kv_cache=None)
                        for name in (MAIN_CACHE, INDEX_CACHE)})
        attention = SimpleNamespace(mla_attn=SimpleNamespace(
            topk_indices_buffer=torch.empty(
                config.scheduler_config.max_num_batched_tokens, 2048,
                dtype=torch.int32,
            ),
        ))
        return SimpleNamespace(model=SimpleNamespace(layers={
            "78": SimpleNamespace(mtp_block=SimpleNamespace(
                self_attn=attention,
            )),
        }))

    zeros = torch.zeros

    def unpinned_zeros(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return zeros(*args, **kwargs)

    with (
        patch.object(CannMtp, "_load_model", load),
        patch.object(torch.npu, "device", return_value=nullcontext()),
        patch.object(torch.Tensor, "pin_memory", lambda self: self),
        patch.object(torch, "zeros", unpinned_zeros),
        patch.object(sfa_v1, "AscendSFAMetadataBuilder") as main_builder,
        patch.object(indexer, "AscendSFAIndexerMetadataBuilder"),
    ):
        draft = CannMtp(target)

    assert draft.config.speculative_config.draft_tensor_parallel_size == tp_size
    assert main_builder.call_args.args[0].num_heads == 64 // tp_size
    assert draft.config.parallel_config.enable_expert_parallel
    assert config.speculative_config is None
    assert list(config.compilation_config.static_forward_context) == ["target"]
    context = draft.config.compilation_config.static_forward_context
    assert context[MAIN_CACHE].kv_cache[0] is target.views["mla_latent.78"]
    assert context[INDEX_CACHE].kv_cache[0] is target.views["index.78"]
    assert draft.cache["latent"].data_ptr() == target.views["mla_latent.78"].data_ptr()
    assert set(draft._input_buffers) == {1, 2, 4, 6, 8, 12, 16, 24, 32}
    assert draft._dyn_lens_dev[4][0, :2].tolist() == [4, 4]
    assert draft.whole_layer_path is None
    assert not draft.rope_graph
    draft._hoisted[1] = (None, None, ())
    with patch.object(cann_model, "refresh_rope") as refresh:
        cann_model.CannModel.refresh_hoisted_rope(draft, 1, draft.positions)
        refresh.assert_called_once_with(draft.positions, 1, ())
    assert not torch_npu.npu._initialized
    source = Path(inspect.getfile(CannMtp))
    return {
        "tp_size": tp_size, "ep_size": tp_size, "heads_per_rank": 64 // tp_size,
        "status": "pass", "npu_initialized": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tp-size", required=True, type=int, choices=(8, 16))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.checkpoint, args.tp_size)
    result["scope"] = "Installed config, cache binding and CPU input allocation"
    result["stubbed"] = ["model weights", "metadata builders", "NPU device", "pinning"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
