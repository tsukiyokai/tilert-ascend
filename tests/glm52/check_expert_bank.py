"""Inspect bank metadata on CPU, or explicitly validate it on one NPU.

The NPU mode compares TP16 rank-local partitions, without HCCL or a full
model. --bench times the candidate only after every numerical check passes.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

from tilert.models.glm_5_2_ascend.ops.expert_bank import checkpoint_contract


CASES = {
    "M1_edges": [[255, 0, 129, 7, 64, 1, 128, 254]],
    "M2_shared": [[3, 90, 20, 8, 127, 201, 45, 250]] * 2,
    "M2_disjoint": [list(range(8, 16)), list(range(32, 40))],
    "M4_shared": [list(range(8))[shift:] + list(range(8))[:shift]
                  for shift in range(4)],
    "M4_disjoint": [list(range(64 * row, 64 * row + 8))
                    for row in range(4)],
}

KERNEL_ANCHORS = {
    "grouped_matmul.cpp": ("op1.Init(x, x, groupList, user1,",),
    "grouped_matmul_antiquant_a8w4_msd.h": (
        "int32_t m = static_cast<int32_t>(groupTokensGm.GetValue(groupIdx));",
        "if (m <= 0)", "quantGroupSize = tiling->k / tiling->quantGroupNum",
        "mnConfig.offsetM += mnConfig.m;",
    ),
    "grouped_matmul_antiquant_a8w4_msd_pre.h": (
        "ReduceSum(groupListFTensor, groupListFTensor, workTensor, groupNum);",
        "totalGroup = groupListTensor.GetValue(0);",
        "xloop < totalGroup",
    ),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(checks, name, passed, **details):
    checks.append({"name": name, "passed": bool(passed), **details})


def scalar_routes(choices):
    entries = sorted((expert, token, slot)
                     for token, row in enumerate(choices)
                     for slot, expert in enumerate(row))
    order = [token * 8 + slot for _, token, slot in entries]
    inverse = [order.index(route) for route in range(len(order))]
    counts = [sum(expert == group for expert, _, _ in entries)
              for group in range(256)]
    return order, inverse, counts


def metadata(args, report):
    report["contracts"] = [
        checkpoint_contract(args.checkpoint, args.layer, rank)
        for rank in (0, 15)
    ]
    record(report["checks"], "real_checkpoint_headers", True,
           headers_per_rank=3072)
    report["installed_kernel_source"] = {}
    for filename, fragments in KERNEL_ANCHORS.items():
        path = args.cann_source / filename
        lines = path.read_text().splitlines()
        matches = {fragment: [number for number, line in enumerate(lines, 1)
                              if fragment in line] for fragment in fragments}
        report["installed_kernel_source"][filename] = {
            "sha256": digest(path), "anchors": matches,
        }
        record(report["checks"], filename, all(matches.values()))
    provenance = args.cann_source / "provenance.json"
    if provenance.exists():
        report["kernel_provenance"] = json.loads(provenance.read_text())
        expected = report["kernel_provenance"]["sha256"]
        copied = report["installed_kernel_source"]
        record(report["checks"], "copied_kernel_hashes",
               all(details["sha256"] == expected.get(name)
                   for name, details in copied.items()))
    for name, choices in CASES.items():
        order, inverse, counts = scalar_routes(choices)
        sorted_ids = [choices[route // 8][route % 8] for route in order]
        record(report["checks"], name + ".scalar_route_oracle",
               sum(counts) == len(choices) * 8
               and sorted_ids == sorted(sorted_ids)
               and all(order[inverse[route]] == route
                       for route in range(len(order))))
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {}
    for name in (
        "tilert/models/glm_5_2_ascend/ops/expert_bank.py",
        "tests/glm52/check_expert_bank.py",
    ):
        path = root / name
        source = path.read_text()
        ast.parse(source, feature_version=(3, 11))
        report["source_sha256"][name] = digest(path)
        record(report["checks"], name + ".line_length",
               all(len(line) <= 79 for line in source.splitlines()))
    for filename in ("weights.py", "moe_plan.py", "ops/expert.py",
                     "ops/quantized.py"):
        name = "tilert/models/glm_5_2_ascend/" + filename
        report["source_sha256"][name] = digest(root / name)
    record(report["checks"], "metadata_without_torch",
           "torch" not in sys.modules)
    report["checkpoint_sha256"] = {
        name: digest(args.checkpoint / name) for name in (
            "config.json", "quant_model_description.json",
            "quant_model_weights.safetensors.index.json",
        )
    }


def compare(checks, name, actual, expected, exact=False):
    import torch
    import torch.nn.functional as functional

    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        record(checks, name, False, error="shape or dtype mismatch")
        return
    if exact:
        record(checks, name, torch.equal(actual, expected))
        return
    a, b = actual.float(), expected.float()
    nrmse = (a - b).norm().item() / max(b.norm().item(), 1e-12)
    cosine = (1.0 if torch.equal(a, b) else functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0
    ).item())
    passed = (bool(torch.isfinite(a).all() and torch.isfinite(b).all())
              and nrmse <= 0.01 and cosine >= 0.999)
    record(checks, name, passed, nrmse=nrmse, cosine=cosine,
           max_abs=(a - b).abs().max().item())


def integer_projection(weight, activation, scale):
    import torch

    packed = weight.values.to(torch.int32) & 255
    nibbles = torch.stack((packed & 15, packed >> 4), dim=1)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    matrix = signed.reshape(weight.out_features, weight.in_features)
    product = activation.to(torch.int32) @ matrix.T
    return (product.float() * weight.scale * scale[:, None]).to(torch.bfloat16)


def single_projection(bank, expert):
    from tilert.models.glm_5_2_ascend.ops.quantized import NpuQuantizedLinear

    return NpuQuantizedLinear(
        bank.weight[expert:expert + 1], bank.scale[expert:expert + 1], 4,
        bank.scale_bias[expert:expert + 1],
    )


def serial_weights(bank, choices):
    return {
        expert: (single_projection(bank.gate_up, expert),
                 single_projection(bank.down, expert))
        for expert in sorted({expert for row in choices for expert in row})
    }


def serial_reference(weights, x, choices, probabilities, capture=True):
    """Run the main engine's host packing and per-expert native CANN chain."""
    import torch
    import torch_npu

    from tilert.models.glm_5_2_ascend.moe_plan import build_plan

    plan = build_plan(choices)
    rows = torch.tensor(plan.token_ids, device=x.device)
    packed = x.index_select(0, rows)
    projected, intermediates, outputs = [], [], []
    for group, expert in enumerate(plan.expert_ids):
        start, end = plan.group_offsets[group:group + 2]
        gate_projection, down_projection = weights[expert]
        gate_up = gate_projection(packed[start:end])
        intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
        outputs.append(down_projection(intermediate))
        if capture:
            projected.append(gate_up)
            intermediates.append(intermediate)
    expert_output = torch.cat(outputs)
    inverse = torch.tensor(plan.inverse, dtype=torch.int32, device=x.device)
    output = torch_npu.npu_moe_token_unpermute(
        expert_output, inverse, probs=probabilities.to(x.dtype)
    )
    if not capture:
        return output
    return {
        "gate_up": torch.cat(projected),
        "intermediate": torch.cat(intermediates),
        "expert_output": expert_output, "output": output,
    }


def check_case(bank, checkpoint, x, choices, probabilities, label, checks):
    import torch

    from tilert.models.glm_5_2_ascend.ops import quantized

    expected_input = x.detach().cpu().clone()
    expected_choices = torch.tensor(choices, dtype=torch.int64)
    device_choices = expected_choices.to(x.device)
    original_probabilities = probabilities.detach().cpu().clone()
    expected_quant, expected_scale = quantized.quantize_a8_reference(
        expected_input
    )
    order, inverse, counts = scalar_routes(choices)
    token_rows = torch.tensor([route // 8 for route in order])
    actual = bank.trace(x, device_choices, probabilities)
    for name, left, right in (
        ("input_preserved", x, expected_input),
        ("choices_preserved", device_choices, expected_choices),
        ("probabilities_preserved", probabilities, original_probabilities),
        ("order", actual.routes.order, torch.tensor(order)),
        ("token_rows", actual.routes.token_ids, token_rows),
        ("inverse", actual.routes.inverse,
         torch.tensor(inverse, dtype=torch.int32)),
        ("counts", actual.routes.counts, torch.tensor(counts)),
        ("input_a8", actual.input_quantized, expected_quant),
        ("packed_a8_preserved", actual.packed_quantized,
         expected_quant.index_select(0, token_rows)),
    ):
        compare(checks, label + "." + name, left, right, exact=True)
    compare(checks, label + ".input_scale", actual.input_scale, expected_scale)
    compare(checks, label + ".packed_scale", actual.packed_scale,
            expected_scale.index_select(0, token_rows))
    down_quant, down_scale = quantized.quantize_a8_reference(
        actual.intermediate.cpu()
    )
    compare(checks, label + ".down_a8_preserved", actual.down_quantized,
            down_quant, exact=True)
    compare(checks, label + ".down_scale", actual.down_scale, down_scale)
    expected = serial_reference(
        serial_weights(bank, choices), x, choices, probabilities
    )
    for name, value in expected.items():
        compare(checks, label + ".serial." + name,
                getattr(actual, name), value)
    packed = actual.packed_quantized.cpu()
    packed_scale = actual.packed_scale.cpu()
    down_input = actual.down_quantized.cpu()
    down_scale = actual.down_scale.cpu()
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        source = checkpoint.expert(bank.layer, expert, bank.rank)
        rows = slice(start, start + count)
        gate = integer_projection(
            source.gate, packed[rows], packed_scale[rows]
        )
        up = integer_projection(source.up, packed[rows], packed_scale[rows])
        down = integer_projection(
            source.down, down_input[rows], down_scale[rows]
        )
        compare(checks, label + f".integer.e{expert}.gate_up",
                actual.gate_up[rows], torch.cat((gate, up), dim=-1))
        compare(checks, label + f".integer.e{expert}.down",
                actual.expert_output[rows], down)
        start += count
        del source
    return device_choices


def benchmark(call, iterations, batches, scope):
    import torch

    for _ in range(20):
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(batches):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append({
            "device_us": start.elapsed_time(end) * 1000 / iterations,
            "host_wall_us": (time.perf_counter() - wall) * 1e6 / iterations,
        })
    return {"warmup": 20, "iterations": iterations, "samples": samples,
            "scope": scope,
            "device_clock": "stream events, including host enqueue gaps",
            "host_clock": "enqueue plus end-event synchronization"}


def case_inputs(choices, generator, device):
    import torch

    x = torch.randn(len(choices), 6144, generator=generator)
    if len(choices) == 4:
        x[-1].zero_()
    probabilities = torch.arange(1, 9).float().repeat(len(choices), 1)
    return (x.to(device=device, dtype=torch.bfloat16),
            (probabilities / 36 * 2.5).to(device))


def run_npu(args, report):
    import torch
    import torch_npu

    from tilert.models.glm_5_2_ascend.ops.expert_bank import NpuExpertBank
    from tilert.models.glm_5_2_ascend.weights import ModelSlimCheckpoint

    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    report["versions"] = {"torch": torch.__version__,
                          "torch_npu": torch_npu.__version__}
    checkpoint = ModelSlimCheckpoint(args.checkpoint)
    generator = torch.Generator().manual_seed(1234)
    report["timings"] = {}
    with torch.inference_mode():
        for rank in (0, 15):
            bank = NpuExpertBank.from_checkpoint(
                checkpoint, args.layer, rank, args.device
            )
            byte_count = sum(
                value.numel() * value.element_size()
                for projection in (bank.gate_up, bank.down)
                for value in (projection.weight, projection.scale,
                              projection.scale_bias)
            )
            record(report["checks"], f"rank{rank}.resident_bytes",
                   byte_count == 321650688, actual=byte_count)
            for name, choices in CASES.items():
                x, probabilities = case_inputs(choices, generator, args.device)
                label = f"rank{rank}.{name}"
                check_case(
                    bank, checkpoint, x, choices, probabilities, label,
                    report["checks"],
                )
            del bank
        if args.bench and all(item["passed"] for item in report["checks"]):
            generator.manual_seed(1234)
            for rank in (0, 15):
                bank = NpuExpertBank.from_checkpoint(
                    checkpoint, args.layer, rank, args.device
                )
                for name, choices in CASES.items():
                    x, probabilities = case_inputs(
                        choices, generator, args.device
                    )
                    device_choices = torch.tensor(choices, device=args.device)
                    weights = serial_weights(bank, choices)
                    report["timings"][f"rank{rank}.{name}"] = {
                        "candidate": benchmark(
                            lambda: bank(x, device_choices, probabilities),
                            args.iterations, args.batches,
                            "device routing/pack/quant/GMM/SwiGLU/combine",
                        ),
                        "host_per_expert": benchmark(
                            lambda: serial_reference(
                                weights, x, device_choices.cpu().tolist(),
                                probabilities, capture=False,
                            ),
                            args.iterations, args.batches,
                            "D2H routes/host plan/pack/CANN experts/combine",
                        ),
                        "weights": "shared bank views, cached before timing",
                    }
                    del weights
                del bank
    report["numerical_execution_completed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cann-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--mode", choices=("metadata", "npu"),
                        default="metadata")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()
    if args.bench and args.mode != "npu":
        parser.error("--bench requires explicit --mode npu")
    if args.iterations < 1 or args.batches < 1:
        parser.error("Timing iterations and batches must be positive")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": sys.version,
        "command": [sys.executable, *sys.argv], "status": "error",
        "source_commit": args.source_commit, "mode": args.mode,
        "numerical_execution_completed": False,
        "device_execution_attempted": False, "hccl_execution": False,
        "scope": "one TP16 partition, no shared FFN or TP sum",
        "checks": [], "cases": CASES,
    }
    try:
        metadata(args, report)
        if not all(item["passed"] for item in report["checks"]):
            raise ValueError("Metadata contract failed")
        if args.mode == "npu":
            report["device_execution_attempted"] = True
            run_npu(args, report)
        report["status"] = ("pass" if all(
            item["passed"] for item in report["checks"]
        ) else "fail")
    except Exception:
        report["error"] = traceback.format_exc()
    report["performance_measured"] = bool(report.get("timings"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "mode": args.mode,
                      "checks": len(report["checks"])}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
