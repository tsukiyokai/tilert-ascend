"""Compare the FFN BF16 multiply/add epilogue with separate Torch operations."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback

from fixtures import bridge_case, epilogue_reference


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


def run(args):
    import torch
    import torch_npu

    torch.ops.load_library(str(args.library.resolve()))
    torch.npu.set_device(args.device)
    report = {
        "scope": "Rank-local FFN epilogue only; separate BF16 multiply and add",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "library": str(args.library.resolve()), "library_sha256": sha(args.library),
        "script_sha256": sha(__file__), "fixtures_sha256": sha(Path(__file__).with_name("fixtures.py")),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "device": str(torch.npu.get_device_properties(args.device)),
        "warmup": 2, "replays": args.replays, "timing": False,
        "oracle": "BF16 scaled = routed * 2.5; combined = scaled + shared",
        "cases": [], "pass": False,
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
    args = parser.parse_args()
    if args.replays < 8 or any(not 1 <= rows <= 32 for rows in args.rows):
        parser.error("Require M in [1,32] and at least eight changing replays")
    run(args)


if __name__ == "__main__":
    main()
