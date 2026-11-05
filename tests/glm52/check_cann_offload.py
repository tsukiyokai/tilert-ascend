"""Check one TP16 W4 layer through the installed Ascend prefetch offloader.

CPU mode reads real weights and checks packed representations. Device mode
compares identical GMM inputs before and after offloading processed weights;
it does not load a model, initialize collectives, or measure performance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import traceback


EXPERTS = 256
HIDDEN = 6144
INTERMEDIATE = 128
ALLOCATOR_LIMIT = 1536 * 1024 ** 2
SELECTED = (0, 1, 7, 64, 128, 129, 254, 255)
METADATA = ("config.json", "quant_model_description.json",
            "quant_model_weights.safetensors.index.json")
UPSTREAM = Path("/workspace/src/vllm-unified")
ASCEND = Path("/workspace/src/vllm-ascend-unified")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(value):
    import torch

    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def record(report, name, passed, **details):
    report["checks"].append({"name": name, "passed": bool(passed), **details})


def compare(report, name, actual, expected):
    import torch

    actual = actual.detach().cpu()
    same_layout = (actual.shape == expected.shape
                   and actual.dtype == expected.dtype)
    equal = same_layout and torch.equal(actual, expected)
    record(report, name, equal, shape=list(actual.shape),
           dtype=str(actual.dtype), sha256=tensor_digest(actual),
           different_elements=(int((actual != expected).sum())
                               if same_layout else None))


def sources():
    local = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), local / "tilert/models/glm_5_2_ascend/weights.py",
             local / "tilert/models/glm_5_2_ascend/ops/expert_bank.py"]
    paths += [UPSTREAM / name for name in (
        "vllm/model_executor/offloader/prefetch.py",
        "vllm/model_executor/offloader/prefetch_ops.py",
        "vllm/model_executor/model_loader/utils.py",
    )]
    paths += [ASCEND / ("vllm_ascend/" + name) for name in (
        "model_executor/offloader/base.py",
        "model_executor/offloader/prefetch.py",
        "worker/model_runner_v1.py",
        "quantization/methods/w4a8/w4a8.py",
    )]
    return {str(path): digest(path) for path in paths}


def load_weights(args, report):
    import torch

    from tilert.models.glm_5_2_ascend.ops.expert_bank import (
        checkpoint_contract,
    )
    from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint

    report["checkpoint_contract"] = checkpoint_contract(
        args.checkpoint, args.layer, args.rank,
    )
    checkpoint = ModelSlimCheckpoint(args.checkpoint)
    banks = {}
    for name, inputs, outputs in (
        ("w13", HIDDEN, 2 * INTERMEDIATE),
        ("w2", INTERMEDIATE, HIDDEN),
    ):
        banks[name] = {
            "packed": torch.empty(EXPERTS, inputs, outputs // 2,
                                  dtype=torch.int8),
            "scale": torch.empty(EXPERTS, 1, outputs, dtype=torch.int64),
            "bias": torch.empty(EXPERTS, outputs, dtype=torch.float32),
        }
    for expert in range(EXPERTS):
        weights = checkpoint.expert(args.layer, expert, args.rank)
        for name, projections in (("w13", (weights.gate, weights.up)),
                                  ("w2", (weights.down,))):
            bank = banks[name]
            packed = torch.cat([item.values for item in projections])
            scale = torch.cat([item.scale for item in projections])
            bias = torch.cat([item.scale_bias for item in projections])
            bank["packed"][expert].copy_(packed.T.contiguous())
            bank["scale"][expert, 0].copy_(
                scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF,
            )
            bank["bias"][expert].copy_(bias)
    for name, bank in banks.items():
        packed = bank["packed"]
        compare(report, name + ".cpu_int32_roundtrip",
                packed.view(torch.int32).view(torch.int8), packed)
        report["weights"][name] = {
            field: {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": tensor_digest(value)}
            for field, value in bank.items()
        }
    record(report, "packed_weight_bytes",
           sum(bank["packed"].numel() for bank in banks.values())
           == 301989888)
    return banks


def fixed_inputs():
    import torch

    rows = len(SELECTED)
    counts = torch.zeros(EXPERTS, dtype=torch.int64)
    counts[list(SELECTED)] = 1
    inputs = {}
    for name, width in (("w13", HIDDEN), ("w2", INTERMEDIATE)):
        values = torch.arange(rows * width).reshape(rows, width)
        inputs[name] = ((values * 37 + 11) % 255 - 127).to(torch.int8)
        inputs[name + "_scale"] = torch.tensor(
            [2 ** (-7 + row % 4) for row in range(rows)],
            dtype=torch.float32,
        )
    inputs["counts"] = counts
    return inputs


def make_module(banks, device):
    import torch
    import torch_npu

    from tilert.models.glm_5_2_ascend.ops.expert_bank import ProjectionBank
    from vllm_ascend.model_executor.offloader.prefetch import (
        mark_prefetch_offload_nz_weight,
    )

    class PackedLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reads = []
            for name, bank in banks.items():
                packed = bank["packed"].to(device)
                packed = torch_npu.npu_format_cast(packed, 29)
                parameter = torch.nn.Parameter(
                    packed.view(torch.int32).contiguous(),
                    requires_grad=False,
                )
                if torch_npu.get_npu_format(parameter) != 29:
                    raise ValueError("The processed W4 parameter is not NZ")
                mark_prefetch_offload_nz_weight(parameter)
                self.register_parameter(name + "_weight", parameter)
                self.register_buffer(name + "_scale", bank["scale"].to(device))
                self.register_buffer(name + "_bias", bank["bias"].to(device))

        def forward(self, gate_codes, gate_scale, down_codes, down_scale,
                    counts):
            outputs = []
            read = {}
            for name, codes, scale in (("w13", gate_codes, gate_scale),
                                        ("w2", down_codes, down_scale)):
                weight = getattr(self, name + "_weight")
                read[name] = {
                    "device": str(weight.device),
                    "format": int(torch_npu.get_npu_format(weight)),
                    "data_ptr": weight.data_ptr(),
                    "storage_bytes": weight.untyped_storage().nbytes(),
                }
                projection = ProjectionBank(
                    weight, getattr(self, name + "_scale"),
                    getattr(self, name + "_bias"),
                )
                outputs.append(projection.apply_quantized(codes, scale,
                                                           counts))
            self.reads.append(read)
            return tuple(outputs)

    return PackedLayer()


def eager_config(checkpoint):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    current_platform.pre_register_and_update()
    return EngineArgs(
        model=str(checkpoint), trust_remote_code=True, dtype="bfloat16",
        quantization="ascend", tensor_parallel_size=16,
        distributed_executor_backend="external_launcher",
        enforce_eager=True, max_model_len=128, max_num_seqs=1,
        max_num_batched_tokens=128, cpu_offload_gb=0,
        offload_backend="prefetch", offload_group_size=1,
        offload_num_in_group=1, offload_prefetch_step=1,
        offload_params={"w13_weight", "w2_weight"},
        speculative_config=None,
        additional_config={"enable_fused_mc2": 0},
    ).create_engine_config()


def device_check(args, banks, inputs, report):
    import torch
    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    torch.npu.set_device(device)
    total = torch.npu.get_device_properties(device).total_memory
    torch.npu.set_per_process_memory_fraction(ALLOCATOR_LIMIT / total, device)
    torch.npu.config.allow_internal_format = True
    report["device_execution"] = True

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.offloader.base import get_offloader, set_offloader
    from vllm_ascend.model_executor.offloader import create_offloader
    from vllm_ascend.worker.model_runner_v1 import _torch_cuda_wrapper

    config = eager_config(args.checkpoint)
    with (torch.inference_mode(), set_current_vllm_config(config),
          _torch_cuda_wrapper()):
        module = make_module(banks, device)
        packed_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        }
        for name, bank in banks.items():
            compare(report, name + ".processed_packed",
                    packed_before[name + "_weight"],
                    bank["packed"].view(torch.int32))
        values = {name: value.to(device) for name, value in inputs.items()}
        forward_args = (values["w13"], values["w13_scale"], values["w2"],
                        values["w2_scale"], values["counts"])
        baseline = tuple(value.cpu() for value in module(*forward_args))
        old_offloader = get_offloader()
        offloader = create_offloader(config.offload_config)
        report["offloader"] = type(offloader).__name__
        if report["offloader"] != "AscendPrefetchOffloader":
            raise ValueError("The configured factory did not select prefetch")
        set_offloader(offloader)
        try:
            wrapped = offloader.wrap_modules(iter([module]))
            if len(wrapped) != 1 or wrapped[0] is not module:
                raise ValueError("The offloader replaced the module")
            for name, parameter in module.named_parameters():
                record(report, name + ".offloaded_to_cpu",
                       parameter.device.type == "cpu")
                compare(report, name + ".cpu_storage", parameter,
                        packed_before[name])
            offloader.post_init()
            saved = {"inputs": inputs, "baseline": baseline}
            for iteration in range(2):
                actual = tuple(value.cpu() for value in module(*forward_args))
                saved[f"offloaded_{iteration}"] = actual
                for name, left, right in zip(("gate_up", "down"),
                                             actual, baseline):
                    compare(report, f"round{iteration}.{name}", left, right)
                for name, parameter in module.named_parameters():
                    compare(report, f"round{iteration}.{name}.packed",
                            parameter, packed_before[name])
            report["forward_reads"] = module.reads
            for iteration, read in enumerate(module.reads[1:]):
                for name, details in read.items():
                    pool = offloader.module_offloaders[0]._param_offloaders[
                        name + "_weight"
                    ]._gpu_buffer
                    record(report, f"round{iteration}.{name}.staging_read",
                           details["data_ptr"] == pool.data_ptr()
                           and details["device"] == str(device)
                           and details["format"] == 29)
            for name, value in values.items():
                compare(report, name + ".input_preserved", value,
                        inputs[name])
            report["offloaded_bytes"] = offloader.total_offloaded_bytes
            report["static_buffer_logical_bytes"] = (
                offloader.buffer_pool.total_bytes
            )
            output = args.output.with_suffix(".pt")
            torch.save(saved, output)
            report["artifact_sha256"] = {output.name: digest(output)}
        finally:
            torch.npu.synchronize(device)
            set_offloader(old_offloader)
        report["allocator_peak_allocated"] = torch.npu.max_memory_allocated(
            device,
        )
        report["allocator_peak_reserved"] = torch.npu.max_memory_reserved(
            device,
        )
        record(report, "no_process_group",
               not torch.distributed.is_initialized())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-only", action="store_true")
    modes.add_argument("--device")
    args = parser.parse_args()
    if args.device:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "checks": [], "weights": {},
        "device_execution": False, "precision_verified": False,
        "performance_measured": False, "collectives_requested": False,
        "allocator_limit_bytes": ALLOCATOR_LIMIT,
        "scope": "processed W4 parameter offload; not full model loading",
        "selected_experts": SELECTED,
    }
    try:
        import torch

        torch.set_num_threads(2)
        report["source_sha256"] = sources()
        report["checkpoint_sha256"] = {
            name: digest(args.checkpoint / name) for name in METADATA
        }
        report["versions"] = {name: version(name) for name in
                              ("torch", "torch-npu", "vllm", "vllm-ascend")}
        banks = load_weights(args, report)
        inputs = fixed_inputs()
        record(report, "group_counts", int(inputs["counts"].sum()) == 8)
        if args.cpu_only:
            backend = sys.modules.get("torch_npu")
            initialized = bool(backend and backend.npu._initialized)
            report["npu_initialized"] = initialized
            record(report, "cpu_only", not initialized)
        else:
            device_check(args, banks, inputs, report)
        report["status"] = ("pass" if all(check["passed"]
                                        for check in report["checks"])
                            else "fail")
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"],
                      "checks": len(report["checks"]),
                      "device_execution": report["device_execution"]}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
