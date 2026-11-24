"""Original constructor and postprocessing with explicit two-device setup."""

import ctypes
import hashlib
import importlib
import json
from pathlib import Path

from tp_device import device_mapping
from tp_groups import device_groups_context
from tp_parameters import initialize as initialize_rank_parameters

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


def run_device_rank(torch, args, state, stage, command_sent):
    import torch_npu
    from vendor_hook_runtime212_device import identity, initialize_parameters
    from vendor_hook_glm52_config import selection
    from vllm.config import set_current_vllm_config

    if not torch.__version__.startswith("2.12.0") or torch_npu.__version__ != "2.12.0":
        raise RuntimeError("Use the recorded Torch/TorchNPU 2.12 runtime")
    stage("npu_bind", lambda: (torch.npu.set_device(args.device_rank), {
        "device_count": torch.npu.device_count(), "logical_device": args.device_rank,
        "device_name": torch.npu.get_device_name(args.device_rank)}))
    if torch.npu.device_count() != 2:
        raise RuntimeError("Expose exactly two NPUs")
    stage("physical_device_mapping", lambda: device_mapping(torch, args))
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
        with device_groups_context(torch, state, stage, args.device_rank, args) as mc2:
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
                    initialize_rank_parameters(torch, layer, args.device_rank, args.oracle_cpu), {"seed": 41021, "rank_shard": args.device_rank}))
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
            from decoder_storage_tp_chain import run_decoder_forward

            stage("actual_two_rank_component_chain", lambda: (
                run_decoder_forward(torch, layers, config, state, args),
                {"report": str(args.output / "tp-chain.json")}))
            return {"constructed_layers": [6, 7], "official_postprocessing": True,
                    "full_decoder_forward_executed": False,
                    "two_rank_component_chain_executed": True,
                    "tp16_collectives_executed": 0,
                    "scope": "Two TP16-shaped shards; separate native components and HCCL2 versus in-kernel SHMEM2",
                    "mc2_scope": "actual per-device singleton used only by the original W4A8 constructor"}
