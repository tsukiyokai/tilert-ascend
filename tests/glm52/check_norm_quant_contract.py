"""Check CANN norm/quant against its FP32 formula and an FP64 reference.

Without --device, audit saved full-model norm snapshots on CPU. With it,
replay those inputs and seeded cases on one NPU. No model weights are loaded.
"""

import argparse
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
import numpy as np


EPSILON = 1e-5
INPUT_NAMES = ("x", "residual", "gamma", "beta")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def quantize(value):
    scale = value.abs().amax(-1, keepdim=True) / 127
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = (value / scale).round().clamp(-128, 127).to(torch.int8)
    return codes, scale.squeeze(-1).float()


def reference(inputs, dtype):
    x, residual, gamma, beta = [inputs[name].to(dtype)
                                for name in INPUT_NAMES]
    total = x + residual
    norm = total / (total.square().mean(-1, keepdim=True) + EPSILON).sqrt()
    norm = norm * gamma + beta
    codes, scale = quantize(norm)
    return {"q": codes, "scale": scale,
            "retained": total.to(inputs["x"].dtype), "norm": norm}


def reference_fp64(inputs):
    x, residual, gamma, beta = [inputs[name].double().numpy()
                                for name in INPUT_NAMES]
    total = x + residual
    norm = total / np.sqrt(np.mean(total * total, axis=-1, keepdims=True)
                           + EPSILON)
    norm = norm * gamma + beta
    scale = np.max(np.abs(norm), axis=-1, keepdims=True) / 127
    scale = np.where(scale > 0, scale, 1)
    q = np.clip(np.rint(norm / scale), -128, 127).astype(np.int8)
    return {"q": torch.from_numpy(q),
            "scale": torch.from_numpy(scale[:, 0].astype(np.float32)),
            "retained": torch.from_numpy(total).to(inputs["x"].dtype),
            "norm": torch.from_numpy(norm)}


def rounding_disagreements(inputs, actual, ref):
    norm = ref["norm"]
    scaled = norm / (norm.abs().amax(-1, keepdim=True) / 127)
    points = []
    zero_beta = bool((inputs["beta"] == 0).all())
    for row, col in (actual["q"] != ref["q"]).nonzero().tolist():
        value = float(scaled[row, col])
        point = {"row": row, "col": col, "before_rounding": value,
                 "half_distance": abs(value - (np.floor(value) + 0.5)),
                 "actual": int(actual["q"][row, col]),
                 "fp64": int(ref["q"][row, col])}
        if zero_beta:
            # RMS normalization cancels from the scale ratio when beta=0.
            maximum = int(norm[row].abs().argmax())

            def product(index):
                total = (Fraction(float(inputs["x"][row, index]))
                         + Fraction(float(inputs["residual"][row, index])))
                return total * Fraction(float(inputs["gamma"][index]))

            exact = 127 * product(col) / abs(product(maximum))
            point["rational_ratio"] = str(exact)
            point["rational_rne"] = round(exact)
        points.append(point)
    return points


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Compared tensor shapes or dtypes differ")
    delta = actual.double() - expected.double()
    relative = delta.abs() / expected.double().abs().clamp_min(1e-300)
    return {"different": int((actual != expected).sum()),
            "elements": actual.numel(),
            "max_abs": float(delta.abs().max()),
            "max_relative": float(relative.max())}


def assess(name, inputs, actual):
    fp32 = reference(inputs, torch.float32)
    fp64 = reference_fp64(inputs)
    rounded_q, rounded_scale = quantize(fp32["norm"].to(
        inputs["x"].dtype).float())
    comparisons = {
        label: {key: compare(actual[key], ref[key])
                for key in ("q", "scale", "retained")}
        for label, ref in (("fp32", fp32), ("fp64", fp64))
    }
    # Match op-plugin's published test: <=1 INT8 code; mixed absolute /
    # relative tolerance 2^-7 for scales and the residual output.
    passed = True
    for ref in (fp32, fp64):
        passed &= bool((actual["q"].int() - ref["q"].int()).abs().max() <= 1)
        for key in ("scale", "retained"):
            error = (actual[key].double() - ref[key].double()).abs()
            bound = ref[key].double().abs().clamp_min(1) * 2 ** -7
            passed &= bool(torch.isfinite(actual[key]).all())
            passed &= bool((error <= bound).all())
    return {"name": name, "shape": list(inputs["x"].shape),
            "dtype": str(inputs["x"].dtype), "published_test_pass": passed,
            "references": comparisons,
            "rounding_disagreements": rounding_disagreements(
                inputs, actual, fp64),
            "scale_fp32_ulp_max": int((actual["scale"].view(torch.int32).long()
                                       - fp64["scale"].view(torch.int32).long()
                                       ).abs().max()),
            "fp32_vs_fp64_q": compare(fp32["q"], fp64["q"]),
            "final_output_rounding_only": {
                "q": compare(actual["q"], rounded_q),
                "scale": compare(actual["scale"], rounded_scale)},
            "output_contract": {
                key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                for key, value in actual.items()}}


def load_cases(args):
    cases = []
    for path in sorted(args.snapshots.glob("norm-layer-*.pt")):
        snapshots = torch.load(path, map_location="cpu", weights_only=True)
        for step, snapshot in enumerate(snapshots):
            inputs = {key: snapshot[key] for key in INPUT_NAMES}
            saved = {key: snapshot[key] for key in ("q", "scale", "retained")}
            cases.append((f"{path.stem}-round-{step}", inputs, saved))
    if not cases:
        raise ValueError("No saved full-model norm snapshots")
    expected = {f"norm-layer-{layer:02d}-round-{step}"
                for layer in range(3) for step in range(2)}
    if {name for name, _, _ in cases} != expected:
        raise ValueError("Expected three layers and two calls per layer")
    if args.device is None:
        return cases
    frozen = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if frozen["x"].shape != (1, 6144):
        raise ValueError("Expected one frozen 6144-element row")
    for width in (1, 8):
        inputs = {key: value.repeat(width, 1) if key in ("x", "residual")
                  else value for key, value in frozen.items()}
        cases.append((f"frozen-m{width}", inputs, None))
    generator = torch.Generator().manual_seed(20261020)
    for dtype in (torch.bfloat16, torch.float16):
        for width, columns in ((1, 128), (1, 6144), (8, 6144),
                               (32, 6144), (8, 6145)):
            inputs = {
                key: torch.randn((width, columns) if key in ("x", "residual")
                                 else (columns,), generator=generator).to(dtype)
                for key in INPUT_NAMES
            }
            cases.append((f"random-{dtype}-m{width}-n{columns}", inputs, None))
    inputs = {key: value.clone() for key, value in frozen.items()}
    inputs["residual"] = -inputs["x"]
    inputs["beta"] = torch.linspace(-2, 2, 6144).bfloat16()
    cases.append(("cancellation-nonzero-beta", inputs, None))
    return cases


def run_device(inputs, device):
    import torch_npu

    x, residual, gamma, beta = [inputs[name].to(device)
                                for name in INPUT_NAMES]
    q, _, retained, scale, _ = torch_npu.npu_add_rms_norm_dynamic_quant(
        x, residual, gamma, beta=beta, epsilon=EPSILON,
        output_mask=[True, False],
    )
    torch.npu.synchronize(device)
    return {"q": q.cpu(), "scale": scale.cpu(), "retained": retained.cpu()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    args.output.mkdir(parents=True, exist_ok=False)
    device = None
    if args.device:
        import torch_npu

        device = torch.device(args.device)
        torch.npu.set_device(device)
        capacity = torch.npu.get_device_properties(device).total_memory
        torch.npu.set_per_process_memory_fraction(256 * 1024 ** 2 / capacity,
                                                  device)
    records, raw = [], {}
    with torch.inference_mode():
        for name, inputs, saved in load_cases(args):
            actual = run_device(inputs, device) if device else saved
            record = assess(name, inputs, actual)
            if device:
                swapped = dict(inputs, x=inputs["residual"],
                               residual=inputs["x"])
                repeated = run_device(inputs, device)
                symmetric = run_device(swapped, device)
                record["repeat_bitexact"] = all(
                    torch.equal(actual[key], repeated[key]) for key in actual)
                record["swap_bitexact"] = all(
                    torch.equal(actual[key], symmetric[key]) for key in actual)
                record["replay_checks_pass"] = (record["repeat_bitexact"]
                                                 and record["swap_bitexact"])
                if saved:
                    record["saved_vs_replay"] = {
                        key: compare(actual[key], saved[key]) for key in actual}
                    record["saved_replay_bitexact"] = all(
                        torch.equal(actual[key], saved[key]) for key in actual)
            raw[name] = {"inputs": inputs, "outputs": actual}
            records.append(record)
            print(json.dumps(record), flush=True)
    raw_path = args.output / "tensors.pt"
    torch.save(raw, raw_path)
    report = {"published_tests_pass": all(item["published_test_pass"]
                                          for item in records),
              "device": args.device, "cases": records,
              "hostname": platform.node(), "script_sha256": sha256(__file__),
              "torch": version("torch"), "timing_measured": False,
              "raw_sha256": sha256(raw_path),
              "snapshot_sha256": {str(path): sha256(path) for path in
                                  sorted(args.snapshots.glob("norm-layer-*.pt"))}}
    if device:
        report["torch_npu"] = version("torch-npu")
        report["device_name"] = torch.npu.get_device_name(device)
        report["peak_allocated"] = torch.npu.max_memory_allocated(device)
        report["replay_checks_pass"] = all(
            item["replay_checks_pass"]
            and item.get("saved_replay_bitexact", True) for item in records)
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    passed = (report["published_tests_pass"]
              and report.get("replay_checks_pass", True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
