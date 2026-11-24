"""Check FFN composition separately from its official bridge arithmetic."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


BRIDGE_NAMES = (
    "normalized", "retained", "rstd", "logits", "probabilities", "ids",
    "shared_codes", "shared_scales", "tiling", "system", "combined", "scaled_routed",
)
BODY_NAMES = {
    0: "stacked", 1: "gate_int32", 2: "middle_packed", 3: "middle_scale",
    4: "down_int32", 5: "routes_bf16", 9: "routed_output", 10: "routed_codes",
    11: "routed_scales", 12: "counts", 13: "starts", 14: "source_row",
    15: "source_slot", 16: "row_weight", 19: "shared_output",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_layout(capacity):
    def align(value):
        return (value + 511) // 512 * 512

    partial = align(capacity * 6144)
    middle = align(partial + 6 * 256 * capacity * 4)
    scales = align(middle + capacity * 128)
    down = align(scales + capacity * 8 * 4)
    return partial, middle, scales, down


def body_values(torch, scratch, rows):
    """Expose initialized logical values; omit capacity padding and workspaces."""
    result = {"body." + name: scratch[index] for index, name in BODY_NAMES.items()}
    capacity = 16 if rows <= 16 else 32
    partial, middle, scale, down = shared_layout(capacity)
    payload = scratch[17]
    result["body.shared_staged"] = payload[:6 * capacity * 1024].view(torch.int8).reshape(
        6, capacity, 1024)[:, :rows]
    result["body.shared_gate_int32"] = payload[
        partial:partial + 6 * 256 * capacity * 4].view(torch.int32).reshape(
            6, 4, capacity, 64)[:, :, :rows]
    result["body.shared_middle"] = payload[middle:middle + capacity * 128].view(
        torch.int8).reshape(capacity, 128)[:rows]
    result["body.shared_middle_scale"] = payload[scale:scale + capacity * 32].view(
        torch.float32).reshape(capacity, 8)[:rows, 0]
    result["body.shared_down_int32"] = payload[
        down:down + capacity * 6144 * 4].view(torch.int32).reshape(24, capacity, 256)[:, :rows]
    return result


def all_values(torch, bridge, body, rows):
    values = {"bridge." + name: bridge[index]
              for index, name in enumerate(BRIDGE_NAMES) if index not in (8, 9)}
    values.update(body_values(torch, body, rows))
    return values


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bridge-library", type=Path, required=True)
    parser.add_argument("--body-library", type=Path, required=True)
    parser.add_argument("--bridge-helper", type=Path, required=True,
                        help="Frozen runtime212-v1/check.py and its sibling helpers")
    parser.add_argument("--body-helper", type=Path, required=True,
                        help="Frozen routed-body build-v2/source/routed_int4/check.py")
    parser.add_argument("--oracle-mode", choices=("orchestration", "both"),
                        default="orchestration")
    parser.add_argument("--oracle-library", type=Path)
    parser.add_argument("--oracle-prefix", type=Path)
    parser.add_argument("--gate-weight-layout", choices=("nd", "nz"), default="nd")
    parser.add_argument("--weight-layout", choices=("nd", "nz"), default="nz")
    parser.add_argument("--shared-scale-dtype", choices=("float32", "bfloat16"),
                        default="bfloat16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 12, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[16, 24])
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.replays < 8 or any(row < 1 or row > 32 for row in args.rows)
            or any(blocks not in (16, 24) for blocks in args.blocks)):
        parser.error("Require M1..32, 16/24 groups and at least eight changing replays")
    if args.oracle_mode == "both" and (args.oracle_library is None or args.oracle_prefix is None):
        parser.error("The production mode requires the pinned private oracle library/prefix")
    return args


def run(args):
    if args.oracle_prefix is not None:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.oracle_prefix.resolve())
    import torch
    import torch_npu

    torch.set_num_threads(1)
    sys.path.insert(0, str(args.bridge_helper.resolve().parent))
    bridge_check = load_module("ffn_bridge_reference", args.bridge_helper)
    fixture = load_module("ffn_routed_fixture", args.body_helper)
    previous = sys.modules.get("check")
    sys.modules["check"] = fixture
    try:
        pipeline = load_module("ffn_pipeline_fixture", args.body_helper.with_name("check_pipeline.py"))
    finally:
        if previous is None:
            del sys.modules["check"]
        else:
            sys.modules["check"] = previous
    libraries = (args.library, args.bridge_library, args.body_library)
    for path in libraries:
        torch.ops.load_library(str(path.resolve()))
    topk, norm, oracle_identity = None, None, None
    if args.oracle_mode == "both":
        topk, norm, oracle_identity = bridge_check.production_oracles(
            torch, args.oracle_library, args.oracle_prefix)
    report = {
        "scope": "One rank-local FFN launch; TP and model latency excluded",
        "oracle_mode": args.oracle_mode,
        "orchestration_reference": "frozen native bridge, accepted callable body, native BF16 epilogue",
        "production_reference": "official norm/gate/top-k/DynamicQuant, accepted body, separate BF16 multiply/add",
        "official_bridge": oracle_identity,
        "utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "libraries": {str(path): digest(path) for path in libraries},
        "sources": {str(path): digest(path) for path in (
            Path(__file__), args.bridge_helper, args.bridge_helper.with_name("fixtures.py"),
            args.bridge_helper.with_name("gate_probe.py"), args.body_helper,
            args.body_helper.with_name("check_pipeline.py"))},
        "build_manifest": json.loads((args.library.parent / "manifest.json").read_text()),
        "queued_clone_producers": True, "warmup": 2, "replays": args.replays,
        "timing": False, "cases": [], "pass": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)

    def persist():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.preflight_only:
        report.update(npu_initialized=torch.npu.is_initialized(), cpu_preflight=True)
        if report["npu_initialized"]:
            raise RuntimeError("CPU preflight initialized NPU")
        persist()
        return
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    report.update(device=str(torch.npu.get_device_properties(args.device)),
                  cann=fixture.cann_versions(torch_npu))

    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    source = fixture.load_weights(torch, SimpleNamespace(
        fixture=None, checkpoint=None, seed=args.seed))
    weights = {name: nd(value) for name, value in source.items()}
    layout = 1 if args.weight_layout == "nz" else 0
    if layout:
        for name in ("gate_up", "down"):
            weights[name] = torch_npu.npu_format_cast(weights[name], 29).view(torch.int32)
    routed_weights = [weights[name] for name in
                      ("gate_up", "gu_bias", "gu_scale", "down", "down_bias", "down_scale")]
    shared = pipeline.shared_weights(torch, nd, torch.Generator().manual_seed(args.seed + 17),
                                     scale_dtype=args.shared_scale_dtype)
    shared_weights = [shared[name] for name in
                      ("gate_tiles", "gate_scale", "down_tiles", "down_scale")]
    report["weight_descriptors"] = {
        name: bridge_check.descriptor(torch_npu, value) for name, value in weights.items()}

    def snapshot(values):
        return {name: bridge_check.snapshot(torch, value) for name, value in values.items()}

    def check_values(name, actual, expected):
        checks = {key: bridge_check.compare(torch, actual[key], value)
                  for key, value in expected.items()}
        failed = [key for key, value in checks.items() if not value["bitwise_equal"]]
        result = {"name": name, "checks": checks, "pass": not failed,
                  "first_mismatch": failed[0] if failed else None}
        report["cases"].append(result)
        return result

    try:
        with torch.inference_mode():
            for blocks in args.blocks:
                for rows in args.rows:
                    hidden, residual, gamma, gate, bias = [
                        nd(value) for value in bridge_check.bridge_case(torch, rows, 0)]
                    original = torch_npu.npu_format_cast(
                        gate.t().contiguous(), 2 if args.gate_weight_layout == "nd" else 29)
                    report["gate_weight_descriptors"] = {
                        "original": bridge_check.descriptor(torch_npu, original),
                        "reference_operand": bridge_check.descriptor(torch_npu, original.t()),
                        "candidate": bridge_check.descriptor(torch_npu, gate)}
                    cb, cr, cs = torch.ops.tilert_ffn_composed.prepare(hidden, blocks, layout)
                    sb, ss = torch.ops.tilert_ffn_bridge.prepare(hidden, blocks)
                    sr, rs = torch.ops.tilert_pipeline_body.routed_shared_prepare(
                        sb[6], 256, blocks, layout)
                    if cs != ss or cs != rs:
                        raise RuntimeError("Same-device component preparations returned different sync bases")
                    official = {}
                    if topk is not None:
                        pr, ps = torch.ops.tilert_pipeline_body.routed_shared_prepare(
                            sb[6], 256, blocks, layout)

                    def composed():
                        torch.ops.tilert_ffn_composed.forward_out(
                            hidden.clone(), residual.clone(), gamma, gate, bias,
                            routed_weights, shared_weights, cb, cr, 1e-5, blocks, layout, cs)

                    def separate():
                        torch.ops.tilert_ffn_bridge.forward_out(
                            hidden.clone(), residual.clone(), gamma, gate, bias,
                            sb, 1e-5, blocks, ss)
                        torch.ops.tilert_pipeline_body.routed_shared_from_bf16(
                            sb[0], *routed_weights, sb[5], sb[4],
                            [sb[6], sb[7].reshape(-1), *shared_weights], sr, rs, 256, blocks, layout)
                        torch.ops.tilert_ffn_bridge.finalize_out(sr[9], sr[19], sb, blocks, ss)

                    def production():
                        normalized, rstd, retained = norm(hidden.clone(), residual.clone(), gamma)
                        logits = torch.mm(normalized, original.t(), out_dtype=torch.float32)
                        probabilities, ids = topk(logits, bias)
                        codes, scales = torch_npu.npu_dynamic_quant(normalized)
                        torch.ops.tilert_pipeline_body.routed_shared_from_bf16(
                            normalized, *routed_weights, ids, probabilities,
                            [codes, scales.reshape(-1), *shared_weights], pr, ps, 256, blocks, layout)
                        scaled, combined = bridge_check.epilogue_reference(pr[9], pr[19])
                        official.update({"bridge.normalized": normalized,
                            "bridge.retained": retained, "bridge.rstd": rstd.reshape(rows, 1),
                            "bridge.logits": logits, "bridge.probabilities": probabilities,
                            "bridge.ids": ids, "bridge.shared_codes": codes,
                            "bridge.shared_scales": scales.reshape(rows, 1),
                            "bridge.combined": combined, "bridge.scaled_routed": scaled})

                    operations = {"composed": composed, "separate": separate}
                    if topk is not None:
                        operations["production"] = production
                    captures = {}
                    for name, operation in operations.items():
                        for _ in range(2):
                            operation()
                        torch.npu.synchronize()
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            operation()
                        captures[name] = graph
                    for iteration in range(args.replays):
                        for target, value in zip((hidden, residual, gamma, gate, bias),
                                bridge_check.bridge_case(torch, rows, iteration)):
                            target.copy_(value)
                        original.copy_(gate.t())
                        # Updates finish outside the capture stream. Each captured
                        # graph still includes the queued hidden/residual producers.
                        torch.npu.synchronize()
                        for graph in captures.values():
                            graph.replay()
                        torch.npu.synchronize()
                        actual = all_values(torch, cb, cr, rows)
                        expected = all_values(torch, sb, sr, rows)
                        case_name = f"b{blocks}-m{rows}-r{iteration}"
                        result = check_values(case_name + "-orchestration", actual, expected)
                        production_result = None
                        if topk is not None:
                            production_values = {**official, **body_values(torch, pr, rows)}
                            production_result = check_values(
                                case_name + "-production", actual, production_values)
                        if not result["pass"] or (production_result and not production_result["pass"]):
                            destination = args.output / (case_name + ".pt")
                            torch.save({"inputs": snapshot({"hidden": hidden, "residual": residual,
                                "gamma": gamma, "gate_kn": gate, "original_gate_nk": original,
                                "bias": bias}), "actual": snapshot(actual),
                                "separate": snapshot(expected),
                                "production": snapshot(production_values) if topk else None}, destination)
                            report["failure_dump"] = str(destination)
                            persist()
                            raise RuntimeError(f"FFN byte comparison failed: {case_name}")
                        persist()
                    for graph in captures.values():
                        graph.reset()
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        persist()
    print(json.dumps({"pass": report["pass"], "cases": len(report["cases"]),
                      "oracle_mode": args.oracle_mode}))


if __name__ == "__main__":
    run(arguments())
