#!/usr/bin/env python3
"""Compare real decoder layers with explicit rank-local TP boundary inputs."""

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def tensor_descriptor(torch_npu, value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "stride": list(value.stride()),
            "format": int(torch_npu.get_npu_format(value)) if value.device.type == "npu" else None,
            "logical_bytes": value.numel() * value.element_size()}


def make_configuration(args):
    from cpu_constructor import quantization_description
    from vendor_hook_glm52_config import model_config, selection
    from vllm.engine.arg_utils import EngineArgs

    model = args.output / "synthetic-model"
    model.mkdir()
    description = {}
    for index in (6, 7):
        description.update(quantization_description(index))
    if args.mode == "bf16":
        for name in list(description):
            if name.endswith("indexer.quant_type"):
                del description[name]
    (model / "config.json").write_text(json.dumps(model_config(), indent=2))
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
    return config, {"tp": 16, "ep": 1, "dp": 1, "pcp": 1,
                    "layers": [selection(6), selection(7)],
                    "synthetic_checkpoint": True, "quantization_description": description}


def join_nonmember_mc2(torch):
    from vllm_ascend.utils import create_hccl_pg_options

    group = torch.distributed.new_group(
        [0], backend="hccl", pg_options=create_hccl_pg_options("mc2"))
    cpu = torch.distributed.new_group([0], backend="gloo")
    expected = torch.distributed.GroupMember.NON_GROUP_MEMBER
    if group != expected or cpu != expected:
        raise RuntimeError("A CPU helper unexpectedly joined the rank-0 MC2 group")
    if torch.npu.is_initialized():
        raise RuntimeError("A CPU helper initialized the NPU while creating a nonmember group")
    return None, {"ranks": [0], "device_backend": "hccl",
                  "device_nonmember": True, "cpu_nonmember": True}


@contextmanager
def rank_zero_mc2(torch, state, stage, command_sent):
    from vendor_hook_runtime212_device import identity
    from vllm_ascend.distributed import parallel_state as ascend_state
    from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch

    if ascend_state._MC2 is not None or state.GroupCoordinator is not GroupCoordinatorPatch:
        raise RuntimeError("Expected the actual worker coordinator and an uninitialized MC2 group")
    torch.distributed.broadcast_object_list(
        [{"create_mc2": True}], src=0, group=state.get_world_group().cpu_group)
    command_sent[0] = True
    group = None
    try:
        def construct():
            nonlocal group
            group = state.init_model_parallel_group(
                [[0]], 0, "hccl", group_name="mc2", use_device_communicator=False)
            if (type(group) is not GroupCoordinatorPatch or group.ranks != [0]
                    or group.world_size != 1 or group.rank_in_group != 0):
                raise RuntimeError("The MC2 metadata group does not have the declared singleton geometry")
            backend = group.device_group._get_backend(torch.device("npu"))
            name = backend.get_hccl_comm_name(0)
            if not name:
                raise RuntimeError("The actual HCCL backend returned an empty communicator name")
            ascend_state._MC2 = group
            return group, {"coordinator": identity(type(group)),
                           "backend_class": str(type(backend)), "ranks": group.ranks,
                           "world_size": group.world_size,
                           "tp_world_size": state.get_tp_group().world_size,
                           "comm_name_sha256": hashlib.sha256(str(name).encode()).hexdigest(),
                           "comm_name_nonempty": True, "tp_collective_validation": False}

        stage("actual_rank_zero_hccl_metadata_group", construct)
        yield group
    finally:
        ascend_state._MC2 = None
        if group is not None:
            group.destroy()


def run_rank_zero(torch, args, state, stage, command_sent):
    import torch_npu
    from vendor_hook_runtime212_device import identity, initialize_parameters
    from vendor_hook_glm52_config import selection
    from vllm.config import set_current_vllm_config

    if not torch.__version__.startswith("2.12.0") or torch_npu.__version__ != "2.12.0":
        raise RuntimeError("Use the recorded Torch/TorchNPU 2.12 runtime")
    stage("npu_bind", lambda: (torch.npu.set_device(0), {
        "device_count": torch.npu.device_count(), "logical_device": 0,
        "device_name": torch.npu.get_device_name(0)}))
    if torch.npu.device_count() != 1:
        raise RuntimeError("Expose exactly one NPU")
    torch.npu.config.allow_internal_format = True
    config = stage("engine_configuration_tp16", lambda: make_configuration(args))
    with set_current_vllm_config(config):
        from vllm_ascend.utils import adapt_patch, enable_custom_op, register_ascend_customop
        stage("actual_worker_registration", lambda: (
            (adapt_patch(), register_ascend_customop(config)),
            {"adapt_patch": identity(adapt_patch), "registration": identity(register_ascend_customop)}))

        def enable_extension():
            enabled = enable_custom_op()
            actual = importlib.import_module("vllm_ascend.vllm_ascend_C")
            path = Path(actual.__file__).resolve()
            if not enabled or hashlib.sha256(path.read_bytes()).hexdigest() != args.extension_sha256:
                raise RuntimeError("The required frozen official extension was not enabled")
            resolver = ctypes.CDLL(str(path)).decoder_official_api_library
            resolver.argtypes, resolver.restype = [ctypes.c_char_p], ctypes.c_char_p
            libraries = {}
            private = {
                "/oracle/packages/vendors/tilert_bridge_oracle/op_api/lib/libcust_opapi.so":
                "b24a676805eac8a528fd6534ad890c038fd68e01e28935f2cb9d22d9f6f53cda"}
            sdk = {
                "/opt/ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so":
                "347437ef3aaf109b4088af90763a2330fb66bb5278e63245366afa8fd7a2233a",
                "/opt/ascend/cann-9.2.0/aarch64-linux/lib64/libopapi_transformer.so":
                "3eebcd29e93860ea2aafaf9dfe269e98a28dc90c7cc325914e479be43bd247cf"}
            for function, expected in (("aclnnAddRmsNormBias", private),
                                       ("aclnnMoeGatingTopK", private),
                                       ("aclnnGroupedMatmulSwigluQuantWeightNzV2", sdk),
                                       ("aclnnDequantSwigluQuantV2", sdk)):
                for symbol in (function, function + "GetWorkspaceSize"):
                    address = resolver(symbol.encode())
                    if address is None:
                        raise RuntimeError(f"The actual official operator did not resolve: {symbol}")
                    library = Path(address.decode()).resolve()
                    sha = hashlib.sha256(library.read_bytes()).hexdigest()
                    if expected.get(str(library)) != sha:
                        raise RuntimeError(f"The official operator library changed: {symbol}: {library}")
                    libraries[symbol] = {"path": str(library), "sha256": sha}
            return None, {"enabled": enabled, "path": str(path),
                          "sha256": args.extension_sha256, "api_libraries": libraries}

        stage("actual_official_extension", enable_extension)
        with rank_zero_mc2(torch, state, stage, command_sent) as mc2:
            from vllm.model_executor.models.deepseek_v2 import (
                DeepseekV2DecoderLayer, GlmMoeDsaForCausalLM,
            )
            from vllm.model_executor.model_loader.utils import (
                configure_quant_config, process_weights_after_loading,
            )
            from vllm_ascend.ascend_forward_context import select_moe_comm_method
            from vllm_ascend.ops.fused_moe.moe_comm_method import (
                AllGatherCommImpl, get_moe_comm_method,
            )
            from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod

            stage("official_packed_quant_mapping", lambda: (
                configure_quant_config(config.quant_config, GlmMoeDsaForCausalLM),
                {"loader": identity(configure_quant_config),
                 "model": identity(GlmMoeDsaForCausalLM),
                 "packed_modules_mapping": config.quant_config.packed_modules_mapping}))
            if config.quant_config.packed_modules_mapping.get("gate_up_proj") != ["gate_proj", "up_proj"]:
                raise RuntimeError("The actual model loader did not configure the shared packed projections")
            from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
            from vllm_ascend.ops.fused_moe.shared_experts import SharedExpertMLPPath

            topk = torch_npu.npu_format_cast(torch.full(
                (32, 2048), -1, dtype=torch.int32, device="npu"), 2)
            layers = []
            for index in (6, 7):
                def construct():
                    previous = torch.get_default_dtype()
                    torch.set_default_dtype(torch.bfloat16)
                    try:
                        with torch.device("cpu"):
                            layer = DeepseekV2DecoderLayer(
                                config, prefix=f"model.layers.{index}", topk_indices_buffer=topk)
                    finally:
                        torch.set_default_dtype(previous)
                    impl = layer.self_attn.mla_attn.mla_attn.impl
                    runner = layer.mlp.experts
                    routed = runner.routed_experts
                    scheme = routed.quant_method.quant_method
                    if not isinstance(scheme, AscendW4A8DynamicFusedMoEMethod):
                        raise RuntimeError("The actual MoE did not construct the original W4A8 scheme")
                    if (impl.local_num_heads != 4 or impl.tp_size != 16
                            or bool(impl.skip_topk) != (index == 7)
                            or not impl.o_proj.reduce_results
                            or layer.use_sequence_parallel_moe):
                        raise RuntimeError("The true decoder does not preserve TP16 attention and layer selection")
                    if (impl.topk_indices_buffer is not topk
                            or layer.self_attn.mla_attn.topk_indices_buffer is not topk):
                        raise RuntimeError("The constructor did not retain the supplied shared NPU top-k owner")
                    cfg = runner.moe_config
                    if (cfg.tp_size, cfg.ep_size, cfg.dp_size, cfg.pcp_size) != (16, 1, 1, 1):
                        raise RuntimeError("The true MoE changed TP16/EP1/DP1/PCP1 geometry")
                    kind = select_moe_comm_method(12, config)
                    method = get_moe_comm_method(kind, cfg)
                    if type(method) is not AllGatherCommImpl:
                        raise RuntimeError("EP1 did not select the actual local AllGather implementation")
                    shared = runner.ascend_shared_experts
                    for projection in (shared.layer.gate_up_proj, shared.layer.down_proj):
                        if (not isinstance(projection.quant_method.quant_method,
                                           AscendW8A8DynamicLinearMethod)
                                or projection.weight.dtype != torch.int8):
                            raise RuntimeError("Both actual shared projections must use W8A8 INT8 weights")
                    if (tuple(shared.layer.gate_up_proj.weight.shape) != (256, 6144)
                            or tuple(shared.layer.down_proj.weight.shape) != (6144, 128)):
                        raise RuntimeError("The shared expert does not have the expected TP16 local weight geometry")
                    if shared._select_mlp_path() is not SharedExpertMLPPath.A8_INT_FUSED:
                        raise RuntimeError("The actual shared expert did not select A8_INT_FUSED")
                    expected_name = mc2.device_group._get_backend(torch.device("npu")).get_hccl_comm_name(0)
                    if scheme.moe_all_to_all_group_name != expected_name:
                        raise RuntimeError("The original W4A8 constructor did not retain the actual HCCL name")
                    return layer, {
                        "layer": identity(type(layer)), "selection": selection(index),
                        "attention": identity(type(impl)), "actual_skip_topk": impl.skip_topk,
                        "attention_reduce_results": impl.o_proj.reduce_results,
                        "input_norm": identity(type(layer.input_layernorm)),
                        "post_norm": identity(type(layer.post_attention_layernorm)),
                        "runner": identity(type(runner)), "routed": identity(type(routed)),
                        "scheme": identity(type(scheme)), "communication": identity(type(method)),
                        "shared_gate_scheme": identity(type(shared.layer.gate_up_proj.quant_method.quant_method)),
                        "shared_down_scheme": identity(type(shared.layer.down_proj.quant_method.quant_method)),
                        "shared_path": shared._select_mlp_path().name,
                        "shared_gate_before": tensor_descriptor(torch_npu, shared.layer.gate_up_proj.weight),
                        "shared_down_before": tensor_descriptor(torch_npu, shared.layer.down_proj.weight),
                        "tp": cfg.tp_size, "ep": cfg.ep_size, "dp": cfg.dp_size, "pcp": cfg.pcp_size,
                        "w13_before": tensor_descriptor(torch_npu, routed.w13_weight),
                        "w2_before": tensor_descriptor(torch_npu, routed.w2_weight),
                        "topk_owner_is_supplied_buffer": impl.topk_indices_buffer is topk,
                        "topk_storage_ptr": topk.untyped_storage().data_ptr(),
                        "topk_device": str(topk.device),
                    }

                layer = stage(f"actual_decoder_layer{index}_constructor", construct)
                parameters = stage(f"layer{index}_synthetic_parameter_fill", lambda: (
                    initialize_parameters(torch, layer), {"seed": 41021}))
                stage(f"layer{index}_move_parameters", lambda: (layer.to("npu"), {
                    "source_parameter_bytes": sum(item["bytes"] for item in parameters),
                    "parameters": parameters}))
                stage(f"layer{index}_official_weight_postprocessing", lambda: (
                    process_weights_after_loading(layer, config.model_config, torch.device("npu")),
                    {"loader": identity(process_weights_after_loading)}))
                impl = layer.self_attn.mla_attn.mla_attn.impl
                if (impl.topk_indices_buffer is not topk
                        or layer.self_attn.mla_attn.topk_indices_buffer is not topk):
                    raise RuntimeError("Moving parameters or official postprocessing changed the top-k owner")
                routed = layer.mlp.experts.routed_experts
                stage(f"layer{index}_processed_weight_contract", lambda: (None, {
                    name: tensor_descriptor(torch_npu, getattr(routed, name))
                    for name in ("w13_weight", "w2_weight", "w13_weight_scale",
                                 "w2_weight_scale", "w13_scale_bias", "w2_scale_bias")}))
                shared = layer.mlp.experts.ascend_shared_experts
                if (shared.layer.gate_up_proj.weight.dtype != torch.int8
                        or shared.layer.down_proj.weight.dtype != torch.int8
                        or shared._select_mlp_path() is not SharedExpertMLPPath.A8_INT_FUSED):
                    raise RuntimeError("Official postprocessing changed the shared W8A8 contract")
                stage(f"layer{index}_processed_shared_contract", lambda: (None, {
                    "path": shared._select_mlp_path().name,
                    "gate": tensor_descriptor(torch_npu, shared.layer.gate_up_proj.weight),
                    "down": tensor_descriptor(torch_npu, shared.layer.down_proj.weight),
                    "gate_scale": tensor_descriptor(torch_npu, shared.layer.gate_up_proj.weight_scale),
                    "down_scale": tensor_descriptor(torch_npu, shared.layer.down_proj.weight_scale),
                    "topk_owner_storage_ptr": topk.untyped_storage().data_ptr(),
                }))
                layers.append(layer)
            from decoder_forward import run_decoder_forward

            stage("actual_local_decoder_forward", lambda: (
                run_decoder_forward(torch, layers, config, state, args),
                {"report": str(args.output / "decoder-forward.json")}))
            return {"constructed_layers": [6, 7], "official_postprocessing": True,
                    "full_decoder_forward_executed": True,
                    "tp16_collectives_executed": 0,
                    "scope": "Original DecoderLayer.forward with explicit post-AR1 inputs and local pre-AR2 outputs",
                    "mc2_scope": "actual HCCL rank0 singleton used by the original W4A8 constructor"}



def worker(rank, args):
    log = (args.output / f"rank-{rank:02d}.log").open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout, sys.stderr = log, log
    report = {"rank": rank, "stages": []}
    state = None
    groups = ()

    def stage(name, action):
        item = {"name": name}
        try:
            result, details = action()
            item.update(success=True, **details)
        except Exception:
            item.update(success=False, error=traceback.format_exc())
            report["stages"].append(item)
            print(json.dumps(item), flush=True)
            raise
        report["stages"].append(item)
        print(json.dumps(item), flush=True)
        return result

    try:
        import torch
        import torch_npu
        from vendor_hook_runtime212_device import GROUP_FIELDS, identity, initialize_groups
        torch.set_num_threads(1)
        report.update(torch=torch.__version__, torch_npu=torch_npu.__version__,
                      python=sys.version, runtime_image=os.environ.get("VENDOR_HOOK_RUNTIME_IMAGE"),
                      npu_initialized_before=torch.npu.is_initialized())
        if rank != 0:
            def refuse_init(*unused_args, **unused_kwargs):
                raise RuntimeError("A CPU-only participant may not initialize the NPU")
            torch.npu._lazy_init = refuse_init
            torch_npu._C._npu_init = refuse_init
        from vllm.platforms import current_platform
        stage("platform_registration", lambda: (
            current_platform.pre_register_and_update(), identity(type(current_platform))))
        state, groups, details = initialize_groups(torch, rank, args)
        report["stages"].append({"name": "actual_tp16_gloo_groups", "success": True, **details})
        status = [None]
        if rank == 0:
            sent = [False]
            try:
                report["result"] = run_rank_zero(torch, args, state, stage, sent)
                status[0] = {"success": True}
            except Exception:
                report["error"] = traceback.format_exc()
                status[0] = {"success": False, "error": report["error"]}
            finally:
                if not sent[0]:
                    torch.distributed.broadcast_object_list(
                        [{"create_mc2": False}], src=0, group=state.get_world_group().cpu_group)
        else:
            command = [None]
            torch.distributed.broadcast_object_list(command, src=0,
                                                    group=state.get_world_group().cpu_group)
            if command[0]["create_mc2"]:
                stage("nonmember_hccl_group_participation", lambda: join_nonmember_mc2(torch))
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
        (args.output / f"rank-{rank:02d}.json").write_text(json.dumps(report, indent=2) + "\n")
    if "error" in report or not report.get("rank_zero_status", {}).get("success", False):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--mode", choices=("int8", "bf16"), default="int8")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--indexed-library", type=Path, required=True)
    parser.add_argument("--shared-library", type=Path, required=True)
    parser.add_argument("--sfa-library", type=Path, required=True)
    parser.add_argument("--forward-widths", type=int, nargs="+", default=[1])
    parser.add_argument("--forward-replays", type=int, default=2)
    parser.add_argument("--forward-sequence", type=int, default=8192)
    args = parser.parse_args()
    args.world_size = 16
    if (any(width not in (1, 2, 12, 32) for width in args.forward_widths)
            or not 1 <= args.forward_replays <= 32
            or not max(args.forward_widths) <= args.forward_sequence <= 8192):
        parser.error("Use widths 1/2/12/32, 1..32 replays and a sequence covering every width")
    for path in (args.adapter, args.indexed_library, args.shared_library, args.sfa_library):
        if not path.is_file():
            parser.error(f"Missing frozen input: {path}")
    if args.timeout < 1 or not (args.runtime / "tests/glm52/whole_layer/vendor_hook_runtime212_device.py").is_file():
        parser.error("An existing frozen runtime and a positive timeout are required")
    sys.path.insert(0, str(args.runtime / "tests/glm52/whole_layer"))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=worker, args=(rank, args)) for rank in range(16)]
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
    report = {"scope": "Two actual decoder layers; explicit local TP cuts; original FFN in both arms; no TP16 numerical reduction",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "exitcodes": [p.exitcode for p in processes],
              "success": all(p.exitcode == 0 for p in processes)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return int(not report["success"])


if __name__ == "__main__":
    raise SystemExit(main())
