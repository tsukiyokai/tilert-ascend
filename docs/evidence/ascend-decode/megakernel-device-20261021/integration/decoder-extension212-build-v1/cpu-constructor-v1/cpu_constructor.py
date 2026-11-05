#!/usr/bin/env python3
"""Record the actual DecoderLayer constructor boundary without an exposed NPU."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import resource
import sys
import tempfile
import traceback


def identity(value):
    path = Path(inspect.getfile(value)).resolve()
    return {"symbol": f"{value.__module__}.{value.__name__}",
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def quantization_description(layer_index):
    prefix = f"model.layers.{layer_index}"
    description = {
        f"{prefix}.self_attn.{name}.weight": "W8A8_DYNAMIC"
        for name in ("q_a_proj", "kv_a_proj_with_mqa", "fused_qkv_a_proj",
                     "q_b_proj", "o_proj", "indexer.wq_b")
    }
    description[f"{prefix}.self_attn.indexer.quant_type"] = "INT8_DYNAMIC"
    for name in ("gate_proj", "up_proj", "down_proj"):
        description[f"{prefix}.mlp.shared_experts.{name}.weight"] = "W8A8_DYNAMIC"
        description[f"{prefix}.mlp.experts.0.{name}.weight"] = "W4A8_DYNAMIC"
    return description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(6, 7), default=6)
    args = parser.parse_args()
    import torch
    import torch_npu

    if list(Path("/dev").glob("davinci*")) or torch.npu.is_initialized():
        raise RuntimeError("The constructor audit requires no exposed or initialized NPU")
    torch.set_num_threads(1)

    def refuse_init(*unused_args, **unused_kwargs):
        raise RuntimeError("The CPU constructor audit refuses NPU initialization")

    torch.npu._lazy_init = refuse_init
    torch_npu._C._npu_init = refuse_init
    report = {
        "scope": "Actual worker, official extension and DecoderLayer constructor; CPU TP1 only",
        "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "stages": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "constructor_completed": False,
    }
    state = None

    def stage(name, action):
        item = {"name": name}
        try:
            value, details = action()
            item.update(success=True, **details)
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
            current_platform.pre_register_and_update(), identity(type(current_platform))))
        from vendor_hook_glm52_config import model_config, selection
        from vllm.config import set_current_vllm_config
        from vllm.engine.arg_utils import EngineArgs

        report["selection"] = selection(args.layer)
        report["quantization_description"] = quantization_description(args.layer)
        with tempfile.TemporaryDirectory(prefix="decoder-cpu-") as temporary:
            model = Path(temporary)
            (model / "config.json").write_text(json.dumps(model_config()))
            (model / "quant_model_description.json").write_text(
                json.dumps(report["quantization_description"]))

            def configure():
                config = EngineArgs(
                    model=str(model), dtype="bfloat16", quantization="ascend",
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
                return config, {"config": identity(type(config)), "tp_size": 1,
                                "quantization": identity(type(config.quant_config))}

            config = stage("engine_configuration", configure)
            with set_current_vllm_config(config):
                from vllm.distributed import parallel_state
                state = parallel_state

                def groups():
                    state.init_distributed_environment(
                        world_size=1, rank=0, local_rank=0, backend="gloo",
                        distributed_init_method=f"file://{model / 'rendezvous'}")
                    state.initialize_model_parallel(backend="gloo")
                    tp = state.get_tp_group()
                    return None, {"coordinator": identity(type(tp)),
                                  "world_size": tp.world_size,
                                  "backend": torch.distributed.get_backend(tp.device_group)}

                stage("real_tp1_cpu_groups", groups)
                from vllm_ascend.utils import (
                    adapt_patch, enable_custom_op, register_ascend_customop,
                )
                stage("real_worker_registration", lambda: (
                    (adapt_patch(), register_ascend_customop(config)),
                    {"adapt_patch": identity(adapt_patch),
                     "registration": identity(register_ascend_customop)}))

                def extension():
                    enabled = enable_custom_op()
                    import vllm_ascend.vllm_ascend_C as actual
                    path = Path(actual.__file__).resolve()
                    if not enabled:
                        raise RuntimeError("The actual worker did not enable the official extension")
                    return actual, {"enabled": enabled, "path": str(path),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "methods": [name for name in dir(actual)
                                                if name.startswith("python_") or name == "init_module"]}

                stage("real_enable_custom_op", extension)

                def norm_contract():
                    from vllm.model_executor.layers.layernorm import RMSNorm
                    with torch.device("meta"):
                        norm = RMSNorm(6144, eps=1e-5, dtype=torch.bfloat16)
                        hidden = torch.empty(12, 6144, dtype=torch.bfloat16)
                        residual = torch.empty_like(hidden)
                        output = norm(hidden, residual)
                    if type(norm).__name__ != "AscendRMSNorm":
                        raise RuntimeError("The real custom-op class registry did not select AscendRMSNorm")
                    if len(output) != 2 or any(list(tensor.shape) != [12, 6144] for tensor in output):
                        raise RuntimeError("The actual norm class changed its two-output Meta contract")
                    return None, {"class": identity(type(norm)),
                                  "forward": identity(type(norm).forward_oot),
                                  "device": "meta", "outputs": [list(x.shape) for x in output],
                                  "numerical_validation": False}

                stage("actual_norm_class_meta_contract", norm_contract)

                def construct():
                    from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
                    from vllm_ascend.distributed import parallel_state as ascend_state
                    report["decoder_constructor"] = identity(DeepseekV2DecoderLayer)
                    report["mc2_registered_before_constructor"] = ascend_state._MC2 is not None
                    previous = torch.get_default_dtype()
                    torch.set_default_dtype(torch.bfloat16)
                    try:
                        with torch.device("cpu"):
                            layer = DeepseekV2DecoderLayer(
                                config, prefix=f"model.layers.{args.layer}",
                                topk_indices_buffer=torch.zeros(32, 2048, dtype=torch.int32))
                    finally:
                        torch.set_default_dtype(previous)
                    return layer, {"class": identity(type(layer)),
                                   "mlp": identity(type(layer.mlp)),
                                   "parameter_bytes": sum(p.numel() * p.element_size()
                                                          for p in layer.parameters())}

                stage("actual_decoder_constructor", construct)
                report["constructor_completed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        report["stopped_at"] = report["stages"][-1]["name"] if report["stages"] else "imports"
    finally:
        if state is not None:
            state.destroy_model_parallel()
            state.destroy_distributed_environment()
        report["npu_initialized"] = torch.npu.is_initialized()
        report["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return int(not report["constructor_completed"] or report["npu_initialized"])


if __name__ == "__main__":
    raise SystemExit(main())
