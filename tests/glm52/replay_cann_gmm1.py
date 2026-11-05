"""Replay frozen W4A8 GMM1 inputs against independent CPU arithmetic."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from check_cann_router import device_config  # noqa: E402
from run_native_decode import digest  # noqa: E402
from tilert.models.glm_5_2_ascend.reference_weights import (  # noqa: E402
    ReferenceWeights,
)


def compare(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    return {"exact": torch.equal(actual, expected),
            "unequal": int((actual != expected).sum()),
            "max_abs": float((actual.float() - expected.float()).abs().max())}


def prepare(weights, stages, layer, rank):
    shapes = {"w13_weight": (256, 128, 6144),
              "w2_weight": (256, 3072, 128),
              "w13_weight_scale": (256, 256, 1),
              "w2_weight_scale": (256, 6144, 1),
              "w13_scale_bias": (256, 256, 1),
              "w2_scale_bias": (256, 6144, 1)}
    bank = torch.nn.Module()
    for name, shape in shapes.items():
        dtype = torch.int8 if name.endswith("_weight") else torch.float32
        bank.register_parameter(name, torch.nn.Parameter(
            torch.zeros(shape, dtype=dtype), requires_grad=False))
    active = stages["group_counts"].nonzero().flatten().tolist()
    assert len(active) == 8 and int(stages["group_counts"].sum()) == 8
    expected, payloads = [], {}
    channels = rank * 128, (rank + 1) * 128
    for row, expert in enumerate(active):
        outputs, values, scales, biases = [], [], [], []
        for stem in ("gate_proj", "up_proj"):
            weight = weights._integer_weight(
                f"model.layers.{layer}.mlp.experts.{expert}.{stem}")
            a, b = channels
            values.append(weight.values[a // 2:b // 2])
            scales.append(weight.scales[a:b])
            biases.append(weight.correction[a:b].sum(1))
            outputs.append(weights._integer_product(
                weight, stages["dispatch_a8"][row:row + 1],
                stages["dispatch_scale"][row:row + 1], channels,
                (0, 6144), torch.float32))
        for suffix, parts in (("weight", values), ("weight_scale", scales),
                              ("scale_bias", biases)):
            target = getattr(bank, "w13_" + suffix)[expert]
            target.copy_(torch.cat(parts).reshape_as(target))
            raw = target.contiguous().view(torch.uint8).numpy().tobytes()
            payloads[f"{expert}.{suffix}"] = hashlib.sha256(raw).hexdigest()
        activated = torch.nn.functional.silu(outputs[0]) * outputs[1]
        maximum = activated.abs().amax(-1)
        scale = torch.div(maximum, torch.full_like(maximum, 127))
        assert bool(torch.isfinite(activated).all())
        assert bool((scale > 0).all())
        factor = torch.div(torch.ones_like(scale), scale)
        codes = (activated * factor[:, None]).half().round().to(torch.int8)
        expected.append((codes, scale))
    return bank, tuple(torch.cat(v) for v in zip(*expected)), payloads


@torch.inference_mode()
def replay(args, bank, stages, expected, report):
    import torch_npu
    from vllm.config import set_current_vllm_config

    config = device_config(args.checkpoint)
    from vllm_ascend.quantization.methods.w4a8 import w4a8
    from vllm_ascend.utils import enable_custom_op

    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(args.device)
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(1536 * 2**20 / capacity)
    torch.npu.reset_peak_memory_stats()
    assert enable_custom_op()
    op = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2
    report["operator_schema"] = str(op.default._schema)
    report["source_sha256"][w4a8.__file__] = digest(Path(w4a8.__file__))
    report["metadata_patches"] = {"tp_world_size": 16, "mc2_group": None}
    captures, inputs = [], []
    with set_current_vllm_config(config):
        with patch.object(w4a8, "get_tensor_model_parallel_world_size",
                          return_value=16), patch.object(
                              w4a8, "get_mc2_group", return_value=None):
            method = w4a8.AscendW4A8DynamicFusedMoEMethod()
        assert not method.use_expert_weight_list and method.tp_size == 16
        # Postprocessing requires w2; its unused weights stay zero.
        bank.to(args.device)
        method.process_weights_after_loading(bank)
        w, scales, _, _ = method._get_mlp_weights(bank)
        bias, _ = method._get_mlp_scale_bias(bank)
        report["bank"] = {name: {
            "shape": list(t.shape), "dtype": str(t.dtype),
            "format": torch_npu.get_npu_format(t),
            "storage_bytes": t.untyped_storage().nbytes()}
            for name, t in bank.named_parameters()}
        assert w[0].shape == (256, 6144, 32) and w[0].dtype == torch.int32
        assert torch_npu.get_npu_format(w[0]) == 29
        assert scales[0].shape == bias[0].shape == (256, 256)
        assert scales[0].dtype == torch.int64
        assert bias[0].dtype == torch.float32
        x = stages["dispatch_a8"].to(args.device)
        x_scale = stages["dispatch_scale"].to(args.device)
        groups = stages["group_counts"].to(args.device)
        for _ in range(args.repeats):
            inputs.append(x.clone())
            result = op(x=inputs[-1], weight=w, weight_scale=scales,
                        x_scale=x_scale, group_list=groups,
                        weight_assist_matrix=bias, dequant_mode=0,
                        group_list_type=1, swiglu_limit=0.0)
            captures.append(tuple(value.clone() for value in result))
        torch.npu.synchronize()
    captures = [tuple(value.cpu() for value in pair) for pair in captures]
    torch.save({"outputs": captures, "reference": expected},
               args.output.with_suffix(".pt"))
    report["capture_sha256"] = digest(args.output.with_suffix(".pt"))
    report["reference"] = [[compare(a, b) for a, b in zip(pair, expected)]
                           for pair in captures]
    report["repeat"] = [[compare(a, b) for a, b in zip(pair, captures[0])]
                        for pair in captures[1:]]
    report["max_abs_code"] = [pair[0].to(torch.int16).abs().amax(-1).tolist()
                              for pair in captures]
    report["amax_invariant"] = all(value == 127 for row in
                                    report["max_abs_code"] for value in row)
    report["peak_reserved_bytes"] = torch.npu.max_memory_reserved()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(16), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    torch.set_num_threads(1)
    record_path = args.trace.parent.parent / f"rank-{args.rank}.json"
    record = json.loads(record_path.read_text())
    hashes = record["trace_sha256"] | record["reset_trace_sha256"]
    assert record["rank"] == args.rank and record["world_size"] == 16
    assert digest(args.trace) == hashes[args.trace.name]
    weights = ReferenceWeights(args.checkpoint, expert_cache_size=2)
    assert weights.contract["metadata_sha256"] == record["checkpoint_sha256"]
    trace = torch.load(args.trace, map_location="cpu", weights_only=True,
                       mmap=True)
    stages = trace["mlp_stages"]
    for key, shape, dtype in (("dispatch_a8", (8, 6144), torch.int8),
                              ("dispatch_scale", (8,), torch.float32),
                              ("group_counts", (256,), torch.int64)):
        assert stages[key].shape == shape and stages[key].dtype == dtype
    assert bool(((stages["group_counts"] == 0)
                 | (stages["group_counts"] == 1)).all())
    bank, expected, payloads = prepare(
        weights, stages, trace["mlp_layer"], args.rank)
    sources = [Path(__file__), Path(inspect.getfile(ReferenceWeights)),
               Path(inspect.getfile(device_config))]
    report = {"scope": "Fixed-input M1 GMM1, one TP16 shard on one die",
              "rank": args.rank, "device": args.device,
              "layer": trace["mlp_layer"], "position": trace["position"],
              "repeats": args.repeats, "status": "error",
              "trace": str(args.trace), "trace_sha256": digest(args.trace),
              "input_report_sha256": digest(record_path),
              "checkpoint_sha256": record["checkpoint_sha256"],
              "active_payload_sha256": payloads,
              "source_sha256": {str(p): digest(p) for p in sources},
              "environment": {key: os.getenv(key) for key in (
                  "ASCEND_CUSTOM_OPP_PATH", "TASK_QUEUE_ENABLE")}}
    try:
        replay(args, bank, stages, expected, report)
        exact = all(v["exact"] for phase in ("reference", "repeat")
                    for row in report[phase] for v in row)
        passed = exact and report["amax_invariant"]
        report["status"] = "pass" if passed else "fail"
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output),
                          "status": report["status"]}))
    raise SystemExit(report["status"] != "pass")


if __name__ == "__main__":
    main()
