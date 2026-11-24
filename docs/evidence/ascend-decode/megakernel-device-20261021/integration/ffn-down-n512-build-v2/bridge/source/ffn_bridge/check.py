"""Compare the bridge with the production A3 norm and top-k APIs."""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import traceback

from fixtures import TOPK_PARAMETERS, bridge_case, epilogue_reference, logits_cases


BUFFER_NAMES = (
    "normalized", "retained", "rstd", "logits", "probabilities", "ids",
    "shared_codes", "shared_scales", "tiling", "system", "combined", "scaled_routed",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(torch, value):
    return value.detach().cpu().clone(memory_format=torch.contiguous_format)


def compare(torch, actual, expected):
    actual, expected = snapshot(torch, actual), snapshot(torch, expected)
    compatible = actual.dtype == expected.dtype and actual.shape == expected.shape
    different = None
    if compatible:
        a = actual.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
        b = expected.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
        different = int((a != b).sum())
    return {"bitwise_equal": compatible and different == 0,
            "bitwise_different_bytes": different,
            "numerical_equal": compatible and torch.equal(actual, expected),
            "shape": list(actual.shape), "dtype": str(actual.dtype),
            "finite": bool(torch.isfinite(actual).all())}


def production_oracles(torch):
    from vllm_ascend.device.device_op import BaseDeviceAdaptor, DeviceOperator
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("The production custom operator library is unavailable")
    if DeviceOperator is not BaseDeviceAdaptor:
        raise RuntimeError("This source-scoped gate requires the A3 BaseDeviceAdaptor")

    def topk(logits, bias):
        return DeviceOperator.moe_gating_top_k(
            logits, bias_opt=bias, **TOPK_PARAMETERS)[:2]

    def norm(hidden, residual, gamma):
        return torch.ops._C_ascend.npu_add_rms_norm_bias(
            hidden, residual, gamma, None, 1e-5)

    identity = {
        "adaptor": str(inspect.getsourcefile(BaseDeviceAdaptor)),
        "adaptor_sha256": sha(inspect.getsourcefile(BaseDeviceAdaptor)),
        "topk_schema": str(torch.ops._C_ascend.moe_gating_top_k.default._schema),
        "norm_schema": str(torch.ops._C_ascend.npu_add_rms_norm_bias.default._schema),
        "topk_parameters": TOPK_PARAMETERS,
        "order": "Descending corrected sigmoid; exact ordered IDs compared to production",
        "probability": "Uncorrected sigmoid selected scores; ReduceSum+eps, Div, factor1",
    }
    return topk, norm, identity


def run(args):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    topk_oracle, norm_oracle, oracle_identity = production_oracles(torch)
    report = {
        "scope": "Rank-local bridge only; TP and model latency are not exercised",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "library": str(args.library.resolve()), "library_sha256": sha(args.library),
        "script_sha256": sha(__file__), "fixtures_sha256": sha(Path(__file__).with_name("fixtures.py")),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": str(torch.npu.get_device_properties(args.device)),
        "warmup": 2, "replays": args.replays, "timing": False,
        "oracle": oracle_identity, "cases": [], "pass": False,
    }
    manifest = args.library.parent / "manifest.json"
    if manifest.is_file():
        report["build_manifest"] = json.loads(manifest.read_text())
    try:
        report["cann"] = str(torch_npu.utils.get_cann_version())
    except (AttributeError, TypeError, RuntimeError, UnicodeError) as error:
        report["cann_version_query_error"] = str(error)
    args.output.mkdir(parents=True, exist_ok=True)

    def persist():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    def check_case(name, actual, expected, inputs):
        torch.npu.synchronize()
        checks = {key: compare(torch, actual[key], value) for key, value in expected.items()}
        case = {"name": name, "checks": checks,
                "pass": all(item["bitwise_equal"] for item in checks.values())}
        report["cases"].append(case)
        if not case["pass"]:
            destination = args.output / (name + ".pt")
            torch.save({"inputs": {key: snapshot(torch, value) for key, value in inputs.items()},
                        "actual": {key: snapshot(torch, actual[key]) for key in expected},
                        "expected": {key: snapshot(torch, value) for key, value in expected.items()}},
                       destination)
            case["failure_dump"] = str(destination)
        persist()
        if not case["pass"]:
            raise RuntimeError(f"Production byte comparison failed: {name}")

    try:
        with torch.inference_mode():
            for blocks in args.blocks:
                for rows in args.rows:
                    sample = nd(torch.zeros(rows, 6144, dtype=torch.bfloat16))
                    scratch, sync = torch.ops.tilert_ffn_bridge.prepare(sample, blocks)
                    outputs = dict(zip(BUFFER_NAMES, scratch))
                    fixed = logits_cases(torch, rows)
                    logits, bias = nd(fixed[0][1]), nd(fixed[0][2])

                    def native_topk():
                        torch.ops.tilert_ffn_bridge.topk_out(
                            logits.clone(), bias.clone(), scratch, blocks, sync)

                    for _ in range(2):
                        native_topk()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        native_topk()
                    for iteration in range(args.replays):
                        name, x, correction = fixed[iteration % len(fixed)]
                        logits.copy_(x)
                        bias.copy_(correction)
                        expected_p, expected_ids = topk_oracle(logits, bias)
                        graph.replay()
                        check_case(f"topk-b{blocks}-m{rows}-r{iteration}-{name}", outputs,
                                   {"probabilities": expected_p, "ids": expected_ids},
                                   {"logits": logits, "bias": bias})
                    graph.reset()
                    if args.topk_only:
                        continue

                    hidden, residual, gamma, weight, correction = [
                        nd(value) for value in bridge_case(torch, rows, 0)]

                    def native_full():
                        torch.ops.tilert_ffn_bridge.forward_out(
                            hidden.clone(), residual.clone(), gamma, weight, correction,
                            scratch, 1e-5, blocks, sync)

                    for _ in range(2):
                        native_full()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        native_full()
                    for iteration in range(args.replays):
                        live = (hidden, residual, gamma, weight, correction)
                        for destination, value in zip(live, bridge_case(torch, rows, iteration)):
                            destination.copy_(value)
                        normalized, rstd, retained = norm_oracle(hidden, residual, gamma)
                        expected_logits = torch.mm(normalized, weight, out_dtype=torch.float32)
                        probabilities, ids = topk_oracle(expected_logits, correction)
                        codes, scales = torch_npu.npu_dynamic_quant(normalized)
                        expected = {
                            "normalized": normalized, "retained": retained,
                            "rstd": rstd.reshape(rows, 1), "logits": expected_logits,
                            "probabilities": probabilities, "ids": ids,
                            "shared_codes": codes, "shared_scales": scales.reshape(rows, 1),
                        }
                        graph.replay()
                        check_case(f"bridge-b{blocks}-m{rows}-r{iteration}", outputs, expected,
                                   dict(zip(("hidden", "residual", "gamma", "weight", "bias"), live)))
                    graph.reset()

                    routed = nd(torch.zeros_like(sample.cpu()))
                    shared = nd(torch.zeros_like(sample.cpu()))

                    def native_final():
                        torch.ops.tilert_ffn_bridge.finalize_out(
                            routed.clone(), shared.clone(), scratch, blocks, sync)

                    for _ in range(2):
                        native_final()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        native_final()
                    for iteration in range(args.replays):
                        x, y, *_ = bridge_case(torch, rows, iteration)
                        if iteration % 8 == 4:
                            x.fill_(-0.0)
                            y.fill_(-0.0)
                        routed.copy_(x)
                        shared.copy_(y)
                        scaled, combined = epilogue_reference(routed, shared)
                        graph.replay()
                        check_case(f"epilogue-b{blocks}-m{rows}-r{iteration}", outputs,
                                   {"scaled_routed": scaled, "combined": combined},
                                   {"routed": routed, "shared": shared})
                    graph.reset()
        report["pass"] = True
        persist()
        print(json.dumps({"pass": True, "cases": len(report["cases"])}))
    except Exception:
        report["error"] = traceback.format_exc()
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 12, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[16, 24])
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--topk-only", action="store_true")
    args = parser.parse_args()
    if args.replays < 8 or any(not 1 <= rows <= 32 for rows in args.rows):
        parser.error("Require M in [1,32] and at least eight changing replays")
    run(args)


if __name__ == "__main__":
    main()
