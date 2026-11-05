"""Compare native vLLM scheduling with same-configuration TileRT decode.

Run with torchrun on sixteen ranks. Both paths consume the captured full-P
prefix. They share loaded weights and use separate cache allocations and
input preparation. The native path calls the installed scheduler, worker,
model runner and sampler. Optional profiling starts after all ranks pass.
"""

import argparse
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import patch

from cann_reduction_probe import TpReductionProbe
from run_cann_model import (
    capture_forward, check_prefix, compare_step, digest, load_capture,
)
from run_native_decode import compare_tensor


LAYER_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def native_caches(runner):
    """Keep native cache views and their scheduler group IDs."""
    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec,
    )

    context = runner.compilation_config.static_forward_context
    specifications = runner.get_kv_cache_spec()
    layers = [{} for _ in range(78)]
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            number = int(LAYER_NAME.search(name)[1])
            spec = specifications[name]
            if isinstance(spec, AscendMLAAttentionSpec):
                fields = ("latent", "rope")
            elif isinstance(spec, AscendSFAIndexerCacheSpec):
                fields = (("index_keys", "index_scales") if spec.scale_dim
                          else ("index_keys",))
            else:
                raise ValueError(
                    f"Unexpected native cache specification: {name}"
                )
            tensors = context[name].kv_cache
            if len(tensors) != len(fields):
                raise ValueError(f"Native cache fields differ: {name}")
            layers[number].update(
                (field, (group_id, tensor))
                for field, tensor in zip(fields, tensors)
            )
    return layers


def tensor_contract(tensor):
    import torch_npu

    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "format": int(torch_npu.get_npu_format(tensor))}


def cache_contract(caches):
    return [{name: tensor_contract(tensor)
             for name, (_, tensor) in layer.items()} for layer in caches]


def copy_native_prefix(caches, blocks, prefixes):
    import torch

    for layer, prefix in zip(caches, prefixes):
        fields = {key for key, value in prefix.items() if value is not None}
        if layer.keys() != fields:
            raise ValueError("Native cache fields differ from captured prefix")
        for name, (group, destination) in layer.items():
            value = prefix[name]
            pages = value.shape[0] // 128
            indices = torch.tensor(blocks[group][:pages], dtype=torch.long,
                                   device=destination.device)
            shaped = value.reshape(pages, *destination.shape[1:])
            destination.index_copy_(0, indices, shaped.to(destination.device))


def native_rows(caches, blocks, position):
    fields = {"latent": "latent", "rope": "rope", "index": "index_keys",
              "index_scale": "index_scales"}
    rows = []
    for layer in caches:
        entry = {}
        for output, name in fields.items():
            value = None
            if name in layer:
                group, tensor = layer[name]
                page, offset = divmod(position, 128)
                value = tensor[blocks[group][page], offset].reshape(-1)
                value = value.detach().cpu().clone()
            entry[output] = value
        rows.append(entry)
    return tuple(rows)


def check_native_prefix(caches, blocks, prefixes, checks, label):
    import torch

    for number, (layer, prefix) in enumerate(zip(caches, prefixes)):
        for name, value in prefix.items():
            if value is None:
                continue
            group, tensor = layer[name]
            pages = value.shape[0] // 128
            indices = torch.tensor(blocks[group][:pages], dtype=torch.long,
                                   device=tensor.device)
            actual = tensor.index_select(0, indices).cpu().reshape(value.shape)
            compare_tensor(f"{label}.prefix.{number}.{name}", actual, value,
                           checks, exact=True)


def capture_native(worker, scheduled, caches, request_id, token, position):
    import torch
    from tilert.models.glm_5_2_ascend.cann_choices import CannChoices
    from tilert.models.glm_5_2_ascend.cann_trace import CannTrace

    runner, inputs = worker.model_runner, []
    model = runner.model

    def capture_input(module, args, kwargs):
        inputs.append((kwargs["input_ids"].detach().clone(),
                       kwargs["positions"].detach().clone()))

    with (model.register_forward_pre_hook(capture_input, with_kwargs=True),
          CannTrace(model) as trace, CannChoices(model) as choices):
        result = worker.execute_model(scheduled)
        if result is not None or runner.execute_model_state is None:
            raise ValueError("Expected native M1 forward followed by sampling")
        state = runner.execute_model_state
        logits, hidden = state[1].detach().clone(), state[5]
        layers = trace.finish(SimpleNamespace(last_hidden=hidden))
        indices, routes = choices.finish()
    if (len(inputs) != 1 or inputs[0][0].tolist() != [token]
            or inputs[0][1].tolist() != [position]):
        raise ValueError("Native forward consumed a different token/position")
    normalized = model.model.norm(torch.cat(layers[1:-1]))
    intermediate_logits = model.compute_logits(normalized)
    sampled = worker.sample_tokens(None)
    if sampled.sampled_token_ids != [[int(logits.argmax(-1).item())]]:
        raise ValueError("Native greedy sampler disagrees with raw logits")
    actual = {
        "token": token, "position": position,
        "committed_position": position + 1,
        "token_ids": torch.tensor(sampled.sampled_token_ids[0]),
        "logits": logits.detach().cpu().clone(),
        "last_hidden": hidden.detach().cpu().clone(),
        "hidden_states": tuple(value.detach().cpu().clone()
                               for value in layers),
        "layer_logits": tuple(value.detach().cpu().clone() for value in (
            *intermediate_logits.split(1), logits,
        )),
        "indices": tuple(value[0].detach().cpu().clone() for value in indices),
        "routes": tuple(None if value is None else
                        value[0].detach().cpu().clone() for value in routes),
        "cache_rows": native_rows(
            caches, runner.requests[request_id].block_ids, position,
        ),
        "choice_source_sha256": choices.source_metadata(),
        "input_contract": {
            "input_ids": tensor_contract(inputs[0][0]),
            "positions": tensor_contract(inputs[0][1]),
        },
    }
    return actual, sampled


def execute(args):
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    import torch.distributed as distributed
    import torch_npu  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_world_group
    from vllm.sampling_params import SamplingParams
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config, get_kv_cache_configs,
        resolve_kv_cache_block_sizes, update_kv_cache_capacity,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import (
        register_all_kvcache_specs,
    )
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm_ascend.worker.worker import NPUWorker
    from tilert.models.glm_5_2_ascend.cann_model import (
        CannModel, create_config,
    )
    from tilert.models.glm_5_2_ascend.cann_trace import CannTrace
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise ValueError("Launch exactly sixteen ranks")
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    metadata = checkpoint_fingerprint(args.checkpoint)
    prefixes, teacher, inputs = load_capture(args.capture_case, rank, metadata)
    initial = inputs["initial_position"]
    logical_capacity = args.capacity - 128
    if initial != 8192 or initial + args.steps > logical_capacity:
        raise ValueError("Require an 8K prefix and sufficient decode capacity")
    payload = torch.load(inputs["prefix_file"], map_location="cpu",
                         mmap=True, weights_only=True)
    prompt = payload["prompt_token_ids"]
    del payload
    config = create_config(args.checkpoint, logical_capacity, 0.4)
    config.model_config.skip_tokenizer_init = True
    config.cache_config.num_gpu_blocks_override = args.capacity // 128
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    with set_current_vllm_config(config), torch.inference_mode():
        worker = NPUWorker(config, local_rank=rank, rank=rank,
                           distributed_init_method="env://",
                           is_driver_worker=rank == 0)
        worker.init_device()
        worker.load_model()
        runner = worker.model_runner
        register_all_kvcache_specs(config)
        specifications = worker.get_kv_cache_spec()
        layout = resolve_kv_cache_layout(
            config, [worker.get_supported_kv_cache_layouts()],
            specifications.values(),
        )
        worker.set_kv_cache_layout(layout.name)
        kv = get_kv_cache_configs(
            config, [specifications], [2 * 1024 ** 3],
        )[0]
        kv.kv_cache_layout = layout.name
        scheduler_kv = generate_scheduler_kv_cache_config([kv])
        config.cache_config.num_gpu_blocks = scheduler_kv.num_blocks
        update_kv_cache_capacity(config, scheduler_kv)
        config.validate_block_size()
        worker.initialize_from_config(kv)
        worker.compile_or_warm_up_model()
        caches = native_caches(runner)
        native_contract = cache_contract(caches)
        block_size, hash_block_size = resolve_kv_cache_block_sizes(
            scheduler_kv, config,
        )
        scheduler = Scheduler(
            vllm_config=config, kv_cache_config=scheduler_kv,
            structured_output_manager=StructuredOutputManager(config),
            block_size=block_size, hash_block_size=hash_block_size,
        )
        request = Request(
            "native-reference", prompt,
            SamplingParams(temperature=0, max_tokens=args.steps + 1,
                           ignore_eos=True), None,
        )
        request.num_computed_tokens = initial
        request.append_output_token_ids(teacher[0])
        scheduler.add_request(request)
        original_update = runner._update_states
        blocks, seeded, checks = None, False, []

        def update(scheduled):
            nonlocal blocks, seeded
            result = original_update(scheduled)
            if not seeded:
                state = runner.requests[request.request_id]
                if (state.num_computed_tokens != initial
                        or state.output_token_ids):
                    raise ValueError(
                        "Expected a newly restored native request"
                    )
                state.output_token_ids.append(teacher[0])
                runner.input_batch.remove_request(request.request_id)
                runner.input_batch.add_request(state)
                runner.input_batch.condense()
                runner.input_batch.refresh_metadata()
                blocks = state.block_ids
                copy_native_prefix(caches, blocks, prefixes)
                check_native_prefix(caches, blocks, prefixes, checks,
                                    "native.before")
                seeded = True
            return result

        native_directory = args.output / "native" / f"rank-{rank}"
        native_directory.mkdir(parents=True, exist_ok=True)
        tokens, traces = [], {}
        with TpReductionProbe(torch.float32, capture_first=0) as reduction:
            with patch.object(runner, "_update_states", update):
                for step in range(args.steps):
                    token, position = (
                        request.output_token_ids[-1], initial + step,
                    )
                    scheduled = scheduler.schedule()
                    if scheduled.num_scheduled_tokens != {
                        request.request_id: 1,
                    }:
                        raise ValueError(
                            "Native scheduler must schedule one row"
                        )
                    actual, sampled = capture_native(
                        worker, scheduled, caches, request.request_id,
                        token, position,
                    )
                    scheduler.update_from_output(scheduled, sampled)
                    if request.num_computed_tokens != position + 1:
                        raise ValueError(
                            "Native scheduler did not advance one row"
                        )
                    path = native_directory / f"step-{position:05d}.pt"
                    torch.save(actual, path)
                    traces[str(path)] = digest(path)
                    tokens.append(token)
            reduction_calls = len(reduction.records)
        check_native_prefix(caches, blocks, prefixes, checks, "native.after")
        engine = CannModel.__new__(CannModel)
        engine.rank, engine.device = rank, device
        engine.capacity, engine.position = args.capacity, 0
        engine.attention_kernel = "sfa"
        engine.fp32_head = False
        engine.checkpoint = str(args.checkpoint.resolve())
        engine.config, engine.worker = config, worker
        engine.runner, engine.model = runner, runner.model
        engine._tp_sum_group = None
        engine._bind_cache()
        candidate_contract = cache_contract(native_caches(runner))
        if candidate_contract != native_contract:
            raise ValueError(
                "Native and TileRT must have identical KV layouts"
            )
        engine._allocate_inputs()
        first = torch.load(
            native_directory / f"step-{initial:05d}.pt", map_location="cpu",
            weights_only=True,
        )
        input_contract = {
            "input_ids": tensor_contract(engine.input_ids),
            "positions": tensor_contract(engine.positions),
        }
        if input_contract != first["input_contract"]:
            raise ValueError("Native and TileRT input tensor layouts differ")
        del first
        engine.block_table = torch.arange(
            logical_capacity // 128, dtype=torch.int32, device=device,
        ).unsqueeze(0)
        native_table = runner.input_batch.block_table[0].get_device_tensor(1)
        table_contract = tensor_contract(native_table)
        if tensor_contract(engine.block_table) != table_contract:
            raise ValueError("Native and TileRT block table layouts differ")
        engine._install_tp_sum()
        try:
            candidate_directory = args.output / "tilert" / f"rank-{rank}"
            candidate_directory.mkdir(parents=True, exist_ok=True)
            greedy_agreement = []
            for phase in ("first", "reset"):
                engine.reset()
                engine.inject(prefixes)
                for step, token in enumerate(tokens):
                    position = initial + step
                    path = native_directory / f"step-{position:05d}.pt"
                    if digest(path) != traces[str(path)]:
                        raise ValueError("Native reference trace hash changed")
                    expected = torch.load(path, map_location="cpu",
                                          weights_only=True)
                    actual = capture_forward(engine, token, position, True)
                    saved = candidate_directory / f"{phase}-{position:05d}.pt"
                    torch.save(actual, saved)
                    traces[str(saved)] = digest(saved)
                    label = f"{phase}.{step}"
                    compare_step(label, actual, expected, checks, exact=True,
                                 max_ulp=1)
                    greedy_agreement.append({
                        "phase": phase, "position": position,
                        "native": int(expected["token_ids"].item()),
                        "tilert": int(actual["token_ids"].item()),
                    })
                check_prefix(engine, prefixes, checks, phase)
            source_objects = (type(worker), type(runner), Scheduler,
                              get_kv_cache_configs, CannModel, capture_forward,
                              compare_step, check_prefix, compare_tensor,
                              TpReductionProbe, CannTrace,
                              resolve_kv_cache_layout)
            sources = {str(Path(inspect.getfile(value)).resolve()): digest(
                Path(inspect.getfile(value))) for value in source_objects}
            sources.update(engine.source_metadata())
            sources.update(actual["choice_source_sha256"])
            sources.update(expected["choice_source_sha256"])
            sources[str(Path(__file__).resolve())] = digest(Path(__file__))
            report = {
                "status": ("pass" if all(c["passed"] for c in checks)
                           else "fail"),
                "implementation": "CannModel vs native vLLM scheduler",
                "oracle": "Scheduler/NPUWorker/NPUModelRunner/sample_tokens",
                "comparison_kind": "same_configuration", "allowed_ulp": 1,
                "rank": rank, "world_size": 16, "device": str(device),
                "steps": args.steps, "teacher_tokens": tokens,
                "logical_capacity": logical_capacity,
                "physical_cache_pages": args.capacity // 128,
                "kv_cache_layout": layout.name,
                "initial_position": initial, "inputs": inputs,
                "checkpoint_sha256": metadata,
                "weights_shared": True, "cache_allocations_independent": True,
                "cache_contract": native_contract,
                "block_table_contract": table_contract,
                "input_contract": input_contract,
                "greedy_agreement": greedy_agreement,
                "native_reduction_calls": reduction_calls,
                "tp_reduction": "float32", "performance_measured": False,
                "source_sha256": sources, "trace_sha256": traces,
                "packages": {name: version(name) for name in (
                    "torch", "torch-npu", "vllm", "vllm-ascend")},
                "collective_env": {name: os.environ.get(name) for name in (
                    "HCCL_DETERMINISTIC", "HCCL_OP_EXPANSION_MODE",
                    "HCCL_BUFFSIZE", "ASCEND_CUSTOM_OPP_PATH")},
                "checks": checks,
            }
            path = args.output / f"rank-{rank}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            ranks = [None] * 16
            distributed.all_gather_object(
                ranks, {"rank": rank, "status": report["status"],
                        "teacher_tokens": tokens,
                        "predictions": greedy_agreement,
                        "report_sha256": digest(path)},
                group=get_world_group().cpu_group,
            )
            same_tokens = all(r["teacher_tokens"] == tokens
                              and r["predictions"] == greedy_agreement
                              for r in ranks)
            passed = all(r["status"] == "pass" for r in ranks) and same_tokens
            if rank == 0:
                (args.output / "summary.json").write_text(json.dumps({
                    "status": "pass" if passed else "fail", "ranks": ranks,
                    "oracle": report["oracle"], "steps": args.steps,
                    "all_ranks_agree": same_tokens,
                    "performance_measured": False,
                }, indent=2) + "\n")
            if passed and args.profile_dir is not None:
                from profile_cann_decode import capture_validated_model

                capture = capture_validated_model(
                    engine, prefixes, teacher[0], rank, args.profile_dir,
                )
                capture.update(
                    checkpoint_sha256=metadata, inputs=inputs,
                    precision_reports={
                        str(args.output / f"rank-{item['rank']}.json"):
                        item["report_sha256"] for item in ranks
                    },
                )
                (args.profile_dir / f"rank-{rank}.json").write_text(
                    json.dumps(capture, indent=2) + "\n",
                )
            return 0 if passed else 1
        finally:
            engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capture-case", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=8448)
    parser.add_argument("--profile-dir", type=Path,
                        help="Capture two M1 steps after all ranks pass")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if args.capacity % 128:
        parser.error("capacity must contain whole 128-token pages")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
