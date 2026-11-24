"""Compare residual/norm/A8 fusion with the selected residual-norm baseline."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics

from attention_check import capture, metric, time_graph


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent /
                        "norm_build/whole_layer_norm.so")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 12, 32])
    parser.add_argument("--blocks", type=int, nargs="+", default=[8])
    parser.add_argument("--replays", type=int, default=8)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--no-timing", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timing-pairs", type=int, default=3)
    parser.add_argument("--baseline", choices=("vllm-ascend", "source-replay"),
                        default="vllm-ascend")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(width < 1 or width > 32 for width in args.widths):
        parser.error("widths must be in [1,32]")
    if any(blocks < 1 or blocks > 24 for blocks in args.blocks):
        parser.error("blocks must be in [1,24]")
    if min(args.replays, args.iterations, args.timing_pairs) < 1:
        parser.error("replays and timing counts must be positive")
    return args


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(torch, torch_npu, args, rows, blocks, with_bias):
    def nd(value):
        return torch_npu.npu_format_cast(value.to(args.device).contiguous(), 2)

    def empty(shape, dtype):
        return nd(torch.empty(shape, dtype=dtype))

    generator = torch.Generator().manual_seed(20261021 + rows)
    initial = {
        "input": torch.randn(rows, 6144, generator=generator).bfloat16(),
        "residual": torch.randn(rows, 6144, generator=generator).bfloat16(),
        "gamma": (1 + torch.randn(6144, generator=generator) / 8).bfloat16(),
    }
    if with_bias:
        initial["beta"] = (torch.randn(6144, generator=generator) / 4).bfloat16()
    live = {name: nd(value) for name, value in initial.items()}

    def outputs():
        return {
            "normalized": empty((rows, 6144), torch.bfloat16),
            "retained": empty((rows, 6144), torch.bfloat16),
            "rstd": empty((rows, 1), torch.float32),
            "codes": empty((rows, 6144), torch.int8),
            "scale": empty((rows, 1), torch.float32),
        }

    native_out, source_out = outputs(), outputs()
    baseline_out = {}
    blocks, sync_base = torch.ops.tilert.norm_prepare(live["input"], blocks)
    native_args = [native_out[name] for name in (
        "normalized", "retained", "rstd", "codes", "scale")]
    source_args = [source_out[name] for name in ("normalized", "retained", "rstd")]
    epsilon = 1e-5

    def native():
        # The queued producer exercises host task-queue submission ordering.
        produced = live["input"].clone()
        return torch.ops.tilert.norm_quant_out(
            produced, live["residual"], live["gamma"], live.get("beta"),
            *native_args, epsilon, blocks, sync_base,
        )

    def baseline():
        produced = live["input"].clone()
        if args.baseline == "vllm-ascend":
            normalized, rstd, retained = (
                torch.ops._C_ascend.npu_add_rms_norm_bias(
                    produced, live["residual"], live["gamma"],
                    live.get("beta"), epsilon,
                )
            )
        else:
            torch.ops.tilert.norm_bias_out(
                produced, live["residual"], live["gamma"], live.get("beta"),
                *source_args, epsilon, blocks, sync_base,
            )
            normalized, retained, rstd = (source_out[name] for name in (
                "normalized", "retained", "rstd"))
        codes, scale = torch_npu.npu_dynamic_quant(normalized)
        baseline_out.update(normalized=normalized, retained=retained,
                            rstd=rstd.reshape(rows, 1), codes=codes,
                            scale=scale.reshape(rows, 1))
        return codes

    def update(replay):
        hidden = initial["input"].roll(replay, dims=1)
        residual = initial["residual"].roll(-replay, dims=1)
        if replay % 4 == 1:
            residual = -hidden
        elif replay % 4 == 2:
            hidden = -hidden
        elif replay % 4 == 3:
            hidden = (hidden.float() / 16).bfloat16()
        live["input"].copy_(hidden)
        live["residual"].copy_(residual)
        live["gamma"].copy_(initial["gamma"].roll(replay))
        if with_bias:
            live["beta"].copy_(initial["beta"].roll(replay * 3))

    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))
    with torch.npu.stream(stream):
        update(0)
    stream.synchronize()
    if args.capture:
        native_graph, _ = capture(torch, native, stream)
        baseline_graph, _ = capture(torch, baseline, stream)
    checks = []
    for replay in range(args.replays):
        with torch.npu.stream(stream):
            update(replay)
            calls = ((native_graph.replay, baseline_graph.replay)
                     if args.capture else (native, baseline))
            for call in calls[::1 if replay % 2 == 0 else -1]:
                call()
        stream.synchronize()
        result = {name: metric(torch, native_out[name], baseline_out[name])
                  for name in native_out}
        result.update(replay=replay, passed=all(item["pass"]
                      for item in result.values()))
        checks.append(result)
        if not result["passed"]:
            break
    report = {"rows": rows, "blocks": blocks, "with_bias": with_bias,
              "checks": checks, "pass": all(item["passed"] for item in checks)}
    if report["pass"] and args.capture and not args.no_timing:
        pairs = []
        for pair in range(args.timing_pairs):
            graphs = [("native", native_graph), ("baseline", baseline_graph)]
            values = {name: time_graph(torch, graph, stream, args.iterations)
                      for name, graph in graphs[::1 if pair % 2 == 0 else -1]}
            pairs.append(values)
        report["timing_pairs_us"] = pairs
        report["median_us"] = {
            name: statistics.median(pair[name] for pair in pairs)
            for name in ("native", "baseline")
        }
    return report


def main():
    args = arguments()
    import torch
    import torch_npu

    if args.baseline == "vllm-ascend":
        import vllm_ascend.vllm_ascend_C  # noqa: F401
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    torch.npu.set_device(args.device)
    torch.ops.load_library(str(args.library.resolve()))
    versions = {}
    for name in ("CANN", "RUNTIME", "DRIVER"):
        try:
            versions[name] = torch_npu.utils.get_cann_version(name)
        except (RuntimeError, UnicodeError) as error:
            versions[name] = f"unavailable: {error}"
    manifest = args.library.parent / "provenance.json"
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "versions": versions,
        "device": str(torch.npu.get_device_properties(args.device)),
        "library_sha256": sha256(args.library), "harness_sha256": sha256(Path(__file__)),
        "build_manifest": json.loads(manifest.read_text()),
        "baseline": args.baseline,
        "scope": ("deployed residual/norm followed by CANN DynamicQuant"
                  if args.baseline == "vllm-ascend" else
                  "same norm source in a separate launch, followed by CANN DynamicQuant"),
        "capture": args.capture, "queued_producer": True,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for rows in args.widths:
            for blocks in args.blocks:
                for with_bias in (False, True):
                    result = run_case(torch, torch_npu, args, rows, blocks, with_bias)
                    report["cases"].append(result)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(result), flush=True)
                    if not result["pass"]:
                        raise SystemExit(1)


if __name__ == "__main__":
    main()
