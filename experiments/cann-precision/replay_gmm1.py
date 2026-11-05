"""Replay one captured M32 GMM1 shard on one A3 die against CPU arithmetic."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests/glm52")]

import torch  # noqa: E402
from replay_cann_gmm1 import compare, replay  # noqa: E402
from run_cann_model import digest  # noqa: E402
from tilert.models.glm_5_2_ascend.reference_weights import ReferenceWeights  # noqa: E402
from tilert.pd_vllm.decode_timing import atomic_json  # noqa: E402


def prepare(weights, stages, layer, rank):
    """Use the existing M1 weight/rounding contract for arbitrary group counts."""
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
    expected, payloads = [], {}
    channels = rank * 128, (rank + 1) * 128
    offset = 0
    for expert, count in enumerate(stages["group_counts"].tolist()):
        if count == 0:
            continue
        outputs, values, scales, biases = [], [], [], []
        for stem in ("gate_proj", "up_proj"):
            weight = weights._integer_weight(
                f"model.layers.{layer}.mlp.experts.{expert}.{stem}")
            a, b = channels
            values.append(weight.values[a // 2:b // 2])
            scales.append(weight.scales[a:b])
            biases.append(weight.correction[a:b].sum(1))
            outputs.append(weights._integer_product(
                weight, stages["dispatch_a8"][offset:offset + count],
                stages["dispatch_scale"][offset:offset + count], channels,
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
        assert bool(torch.isfinite(activated).all()) and bool((scale > 0).all())
        factor = torch.div(torch.ones_like(scale), scale)
        codes = (activated * factor[:, None]).half().round().to(torch.int8)
        expected.append((codes, scale))
        offset += count
    assert offset == stages["dispatch_a8"].shape[0] == 256
    return bank, tuple(torch.cat(value) for value in zip(*expected)), payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "trace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(16), required=True)
    parser.add_argument("--phase", choices=("wide", "reset", "clear"), default="wide")
    parser.add_argument("--device", required=True)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".pt").exists():
        parser.error("Preserve existing artifacts; use a new output path")
    if args.repeats < 2:
        parser.error("At least two calls are needed for repeat comparison")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "One frozen M32 layer3 TP16 weight shard on one A3 die",
              "status": "error", "timed": False, "performance_measured": False,
              "precision_verified": False, "formal_precision_gate_passed": False,
              "rank": args.rank, "phase": args.phase, "repeats": args.repeats,
              "device": args.device, "source_sha256": {
                  str(Path(__file__)): digest(Path(__file__)),
                  str(ROOT / "tests/glm52/replay_cann_gmm1.py"):
                      digest(ROOT / "tests/glm52/replay_cann_gmm1.py"),
                  str(ROOT / "tilert/models/glm_5_2_ascend/reference_weights.py"):
                      digest(ROOT / "tilert/models/glm_5_2_ascend/reference_weights.py")}}
    try:
        torch.set_num_threads(1)
        record_path = args.trace.parent / f"rank-{args.rank}.json"
        record = json.loads(record_path.read_text())
        assert record["status"] == "captured" and record["rank"] == args.rank
        phase = next(v for v in record["phases"] if v["phase"] == args.phase)
        assert args.trace.name == f"rank-{args.rank}-{args.phase}.pt"
        assert digest(args.trace) == phase["artifact_sha256"]
        trace = torch.load(args.trace, map_location="cpu", weights_only=True)
        prefix = "layer.3.mlp_stages."
        stages = {key[len(prefix):]: value for key, value in trace.items()
                  if key.startswith(prefix)}
        for key, shape, dtype in (("dispatch_a8", (256, 6144), torch.int8),
                                  ("dispatch_scale", (256,), torch.float32),
                                  ("group_counts", (256,), torch.int64)):
            assert stages[key].shape == shape and stages[key].dtype == dtype
        assert bool((stages["group_counts"] >= 0).all())
        assert int(stages["group_counts"].sum()) == 256
        weights = ReferenceWeights(args.checkpoint, expert_cache_size=2)
        assert weights.contract["metadata_sha256"] == record["checkpoint_sha256"]
        bank, expected, payloads = prepare(weights, stages, 3, args.rank)
        report.update(trace_sha256=digest(args.trace),
                      input_report_sha256=digest(record_path),
                      checkpoint_sha256=record["checkpoint_sha256"],
                      active_payload_sha256=payloads,
                      captured_vs_cpu=[compare(a, b) for a, b in zip(
                          (stages["fused_a8"], stages["fused_scale"]), expected)],
                      environment={key: os.getenv(key) for key in (
                          "ASCEND_CUSTOM_OPP_PATH", "TASK_QUEUE_ENABLE")})
        import torch_npu  # noqa: F401

        properties = torch.npu.get_device_properties(args.device)
        assert properties.name.upper() in ("ASCEND910_9392", "ASCEND910_9382")
        report["device_properties"] = str(properties)
        replay(args, bank, stages, expected, report)
        passed = all(item["exact"] for phase in ("reference", "repeat")
                     for row in report[phase] for item in row)
        report["status"] = "pass" if passed else "fail"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    report["effective_opp_path"] = os.getenv("ASCEND_CUSTOM_OPP_PATH")
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
