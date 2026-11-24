#!/usr/bin/env python3
"""Construct and install a real TP16 attention shard with one NPU-owning rank."""

import argparse
from datetime import timedelta
import hashlib
import importlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import traceback


GROUP_FIELDS = ("_TP", "_ETP", "_DCP", "_PCP", "_PP", "_DP", "_EP")


def identity(value):
    path = Path(inspect.getfile(value)).resolve()
    return {"class": f"{value.__module__}.{value.__name__}", "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def initialize_groups(torch, rank, args):
    from vllm.distributed import parallel_state as state

    state.init_distributed_environment(
        world_size=args.world_size, rank=rank, local_rank=0, backend="gloo",
        distributed_init_method=f"file://{args.output / 'rendezvous'}",
        timeout=timedelta(seconds=args.timeout),
    )
    if any(getattr(state, field) is not None for field in GROUP_FIELDS):
        raise RuntimeError("The isolated process already owns model-parallel groups")
    tp = state.init_model_parallel_group(
        [list(range(args.world_size))], 0, "gloo",
        use_device_communicator=False, group_name="vendor_hook_tp",
    )
    singleton = state.init_model_parallel_group(
        [[index] for index in range(args.world_size)], 0, "gloo",
        use_device_communicator=False, group_name="vendor_hook_singleton",
    )
    # Register real coordinators. The standard TP initializer also creates an
    # NPU communicator on every rank; this harness exercises only rank 0's
    # local attention shard and excludes that collective.
    for field in GROUP_FIELDS:
        setattr(state, field, tp if field in ("_TP", "_ETP") else singleton)
    witness = torch.tensor(rank, dtype=torch.int64)
    torch.distributed.all_reduce(witness, group=tp.cpu_group)
    expected = args.world_size * (args.world_size - 1) // 2
    if witness.item() != expected:
        raise RuntimeError("Real Gloo TP group failed its rank-sum witness")
    return state, (tp, singleton), {
        "tp_world_size": tp.world_size, "tp_rank": tp.rank_in_group,
        "cpu_rank_sum": witness.item(), "expected_rank_sum": expected,
        "coordinator": identity(type(tp)), "device_communicator": None,
        "registration": "actual coordinators assigned to version-scoped parallel_state fields",
        "singleton_fields": list(GROUP_FIELDS[2:]),
    }


def make_configuration(args):
    from vendor_hook_cpu import MODEL_CONFIG
    from vllm.engine.arg_utils import EngineArgs

    model = args.output / "synthetic-model"
    model.mkdir()
    config_data = dict(MODEL_CONFIG)
    config_data["indexer_rope_interleave"] = True
    (model / "config.json").write_text(json.dumps(config_data, indent=2))
    prefix = "model.layers.0.self_attn"
    description = {
        f"{prefix}.{name}.weight": "W8A8_DYNAMIC"
        for name in ("q_a_proj", "kv_a_proj_with_mqa", "fused_qkv_a_proj",
                     "q_b_proj", "o_proj", "indexer.wq_b")
    }
    if args.mode == "int8":
        description[f"{prefix}.indexer.quant_type"] = "INT8_DYNAMIC"
    (model / "quant_model_description.json").write_text(json.dumps(description, indent=2))
    config = EngineArgs(
        model=str(model), dtype="bfloat16", quantization="ascend",
        skip_tokenizer_init=True, load_format="dummy",
        tensor_parallel_size=16, pipeline_parallel_size=1, data_parallel_size=1,
        distributed_executor_backend="external_launcher",
        max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=32,
        block_size=128, kv_cache_dtype="auto",
        attention_config={"indexer_kv_dtype": "int8" if args.mode == "int8" else "auto"},
        enforce_eager=True, enable_prefix_caching=False,
        enable_chunked_prefill=True, async_scheduling=False,
        additional_config={"enable_mlapo": False, "enable_cpu_binding": False,
                           "ascend_log_path": "/tmp/vendor-hook-log"},
    ).create_engine_config()
    return config, prefix


def initialize_parameters(torch, layer):
    generator = torch.Generator(device="cpu").manual_seed(41021)
    report = []
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if parameter.dtype == torch.int8:
                value = torch.randint(-8, 9, parameter.shape, dtype=torch.int8,
                                      generator=generator)
            elif name.endswith("weight_scale"):
                value = (0.002 + torch.rand(parameter.shape, generator=generator)
                         * 0.002).to(parameter.dtype)
            elif name.endswith("weight_offset") or name.endswith("bias"):
                value = torch.zeros(parameter.shape, dtype=parameter.dtype)
            elif parameter.ndim == 1:
                value = torch.ones(parameter.shape, dtype=parameter.dtype)
            else:
                value = (torch.randn(parameter.shape, generator=generator)
                         * 0.01).to(parameter.dtype)
            parameter.copy_(value)
            report.append({"name": name, "shape": list(parameter.shape),
                           "dtype": str(parameter.dtype),
                           "bytes": parameter.numel() * parameter.element_size()})
    return report


def run_rank_zero(torch, args, stage):
    import torch_npu
    from vllm.config import set_current_vllm_config

    stage("npu_bind", lambda: (torch.npu.set_device(0), {
        "device_count": torch.npu.device_count(), "logical_device": 0,
        "name": torch.npu.get_device_name(0),
    }))
    if torch.npu.device_count() != 1:
        raise RuntimeError("Expose exactly one NPU to this bootstrap container")
    torch.npu.config.allow_internal_format = True
    config, prefix = stage("engine_config_tp16", lambda: (
        make_configuration(args), {"tp": 16, "pp": 1, "dp": 1,
                                   "synthetic_checkpoint": True,
                                   "indexer_mode": args.mode},
    ))
    with set_current_vllm_config(config):
        from vllm_ascend.utils import adapt_patch, register_ascend_customop

        stage("real_worker_registration", lambda: (
            (adapt_patch(), register_ascend_customop(config)), {},
        ))
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

        def construct():
            previous = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                with torch.device("cpu"):
                    topk = torch.full((32, 2048), -1, dtype=torch.int32)
                    layer = DeepseekV2MLAAttention(
                        config, config.model_config.hf_config,
                        6144, 64, 192, 64, 256, 2048, 512,
                        cache_config=config.cache_config, quant_config=config.quant_config,
                        prefix=prefix, reduce_results=False, topk_indices_buffer=topk,
                    )
            finally:
                torch.set_default_dtype(previous)
            attention = layer.mla_attn.mla_attn
            impl = attention.impl
            return layer, {
                "outer": identity(type(layer)), "wrapper": identity(type(layer.mla_attn)),
                "attention": identity(type(attention)), "impl": identity(type(impl)),
                "indexer": identity(type(impl.indexer)),
                "indexer_backend": identity(type(impl.indexer.impl)),
                "tp_size": impl.tp_size, "local_heads": impl.local_num_heads,
                "reduce_results": impl.o_proj.reduce_results,
                "initial_preprocess": str(impl.preprocess_type),
                "quant_config": identity(type(config.quant_config)),
            }

        layer = stage("real_attention_constructor", construct)
        parameters = stage("synthetic_parameter_fill", lambda: (
            initialize_parameters(torch, layer), {"seed": 41021},
        ))
        attention = layer.mla_attn.mla_attn
        impl = attention.impl
        topk = torch.full((32, 2048), -1, dtype=torch.int32, device="npu")
        layer.topk_indices_buffer = topk
        impl.topk_indices_buffer = topk
        stage("move_real_parameters", lambda: (layer.to("npu"), {
            "source_parameter_bytes": sum(item["bytes"] for item in parameters),
            "parameters": parameters,
        }))
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        stage("official_weight_postprocessing", lambda: (
            process_weights_after_loading(layer, config.model_config, torch.device("npu")),
            {"loader": identity(process_weights_after_loading)},
        ))
        from vllm.v1.worker.utils import bind_kv_cache

        pages = 66
        cache = tuple(torch_npu.npu_format_cast(torch.zeros(
            pages, 128, 1, dimension, dtype=torch.bfloat16, device="npu"), 2)
                      for dimension in (512, 64))
        key_dtype = torch.int8 if args.mode == "int8" else torch.bfloat16
        index_cache = (torch_npu.npu_format_cast(torch.zeros(
            pages, 128, 1, 128, dtype=key_dtype, device="npu"), 2),)
        if args.mode == "int8":
            index_cache += (torch.ones(pages, 128, 1, dtype=torch.float16, device="npu"),)
        allocated = {attention.layer_name: cache, impl.indexer.k_cache.prefix: index_cache}
        runner_caches = []
        stage("official_cache_binding", lambda: (
            bind_kv_cache(allocated, config.compilation_config.static_forward_context,
                          runner_caches),
            {"function": identity(bind_kv_cache), "cache_names": list(allocated),
             "pages": pages, "page_size": 128},
        ))
        os.environ["TILERT_INDEXED_ATTENTION_LIBRARY"] = str(args.indexed_library)
        os.environ["TILERT_SHARED_ATTENTION_LIBRARY"] = str(args.shared_library)
        sys.path.insert(0, str(args.adapter.resolve().parents[3]))
        adapter = importlib.import_module("tilert.models.glm_5_2_ascend.attention_native")
        if Path(adapter.__file__).resolve() != args.adapter.resolve():
            raise RuntimeError("The imported adapter differs from the requested source")
        stage("load_torch210_libraries", lambda: (
            adapter.load(),
            {"paths": [str(args.indexed_library), str(args.shared_library)],
             "sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (args.indexed_library, args.shared_library)]},
        ))
        pool = adapter.WorkspacePool(blocks=24)
        restore = stage("real_attention_install", lambda: (
            adapter.install(attention, pool), {
                "adapter_sha256": hashlib.sha256(args.adapter.read_bytes()).hexdigest(),
                "preprocess": str(impl.preprocess_type),
                "native_mode": adapter._validate_impl(impl),
                "tp_size": impl.tp_size, "local_heads": impl.local_num_heads,
                "installed_layer_name": impl.layer_name,
                "final_weight_shapes": {name: list(getattr(impl, name).shape)
                                        for name in ("weight_dq", "weight_dkv_kr",
                                                     "weight_uq_qr", "W_UK_T", "W_UV")},
            },
        ))
        torch.npu.synchronize()
        restore()
        return {"installed": True, "forward_executed": False,
                "tp_collective_executed": False,
                "npu_allocated_bytes": torch.npu.memory_allocated(),
                "npu_peak_allocated_bytes": torch.npu.max_memory_allocated()}


def worker(rank, args):
    log = (args.output / f"rank-{rank:02d}.log").open("w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    report = {"rank": rank, "stages": [], "scope": "real Gloo groups, rank-0 local attention shard"}
    groups = ()
    state = None

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
        import torch
        import torch_npu
        torch.set_num_threads(1)
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__,
                      npu_initialized_before=torch.npu.is_initialized())
        if rank != 0 or args.groups_only:
            def refuse_init(*unused_args, **unused_kwargs):
                raise RuntimeError("CPU Gloo participant refuses NPU initialization")
            torch.npu._lazy_init = refuse_init
            torch_npu._C._npu_init = refuse_init
        from vllm.platforms import current_platform
        stage("platform_registration", lambda: (
            current_platform.pre_register_and_update(), identity(type(current_platform)),
        ))
        state, groups, detail = initialize_groups(torch, rank, args)
        report["stages"].append({"name": "real_groups", "success": True, **detail})
        status = [None]
        if rank == 0:
            try:
                report["result"] = ({"groups_only": True} if args.groups_only
                                    else run_rank_zero(torch, args, stage))
                status[0] = {"success": True}
            except Exception:
                report["error"] = traceback.format_exc()
                status[0] = {"success": False, "error": report["error"]}
        torch.distributed.broadcast_object_list(status, src=0,
                                                group=state.get_world_group().cpu_group)
        report["rank_zero_status"] = status[0]
        torch.distributed.barrier(group=state.get_world_group().cpu_group)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        if state is not None:
            for field in GROUP_FIELDS:
                setattr(state, field, None)
            for group in groups:
                group.destroy()
            state.destroy_distributed_environment()
        if "torch" in locals():
            report["npu_initialized_after"] = torch.npu.is_initialized()
        report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["memory_snapshot"] = Path("/proc/self/smaps_rollup").read_text()
        (args.output / f"rank-{rank:02d}.json").write_text(json.dumps(report, indent=2) + "\n")
    success = report.get("rank_zero_status", {}).get("success", False)
    if "error" in report or not success:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-only", action="store_true")
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--mode", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--indexed-library", type=Path)
    parser.add_argument("--shared-library", type=Path)
    args = parser.parse_args()
    if not args.groups_only:
        if args.world_size != 16:
            parser.error("The actual attention shard requires world-size=16")
        if any(value is None or not value.is_file() for value in (
                args.adapter, args.indexed_library, args.shared_library)):
            parser.error("Actual installation requires adapter and both library files")
    if args.world_size < 1 or args.timeout < 1:
        parser.error("world-size and timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, args))
                 for rank in range(args.world_size)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + args.timeout
    while any(process.is_alive() for process in processes):
        if (time.monotonic() >= deadline
                or any(process.exitcode not in (None, 0) for process in processes)):
            for process in processes:
                if process.is_alive():
                    process.terminate()
            break
        for process in processes:
            process.join(timeout=0.1)
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    summary = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "arguments": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
               "exitcodes": [process.exitcode for process in processes],
               "success": all(process.exitcode == 0 for process in processes)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(not summary["success"])


if __name__ == "__main__":
    raise SystemExit(main())
