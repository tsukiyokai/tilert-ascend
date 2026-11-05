"""Measure prewarmed one-time BF16 shadow packing into an existing ND tensor."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--pages", type=int, nargs="+", default=[68])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(pages <= 0 for pages in args.pages):
        parser.error("pages must be positive")
    if args.samples < 30:
        parser.error("samples must be at least 30 independent events per arm")
    if args.warmup < 1:
        parser.error("warmup must be positive")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_bytes(torch, tensor):
    return tensor.detach().cpu().reshape(-1).clone(
        memory_format=torch.contiguous_format).view(torch.uint8)


def compare_bytes(torch, actual, expected):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return {"pass": False, "bitwise_equal": False,
                "bitwise_different_bytes": None,
                "actual_dtype": str(actual.dtype),
                "expected_dtype": str(expected.dtype),
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape)}
    different = int((raw_bytes(torch, actual) !=
                     raw_bytes(torch, expected)).sum())
    return {"pass": different == 0, "bitwise_equal": different == 0,
            "bitwise_different_bytes": different, "bytes": actual.nbytes}


def fixture(torch, pages, seed):
    generator = torch.Generator().manual_seed(seed)
    values = []
    for channels in (512, 64):
        value = torch.randn(pages, 128, 1, channels,
                            generator=generator).bfloat16()
        # Packing copies all BF16 encodings, including signed zero and NaN payloads.
        value.view(torch.int16).reshape(-1)[:5] = torch.tensor(
            [0, -32768, 32640, -128, 32705], dtype=torch.int16)
        values.append(value)
    return values


def pack_into(torch, latent, rope, packed):
    return torch.cat((latent.squeeze(2), rope.squeeze(2)), dim=-1, out=packed)


def summary(values):
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return {"samples": len(values), "p50_us": statistics.median(values),
            "p90_us": quantiles[89], "p99_us": quantiles[98],
            "mean_us": statistics.mean(values),
            "std_us": statistics.pstdev(values),
            "min_us": min(values), "max_us": max(values)}


def provenance(torch, torch_npu, args):
    versions = {}
    for name in ("CANN", "RUNTIME", "DRIVER"):
        try:
            versions[name] = torch_npu.utils.get_cann_version(name)
        except Exception as error:
            versions[name] = f"unavailable: {type(error).__name__}: {error}"
    module_root = Path(torch_npu.__file__).resolve().parent
    library = module_root / "lib" / "libtorch_npu.so"
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "versions": versions,
        "device": args.device,
        "device_properties": str(torch.npu.get_device_properties(args.device)),
        "harness": str(Path(__file__).resolve()),
        "harness_sha256": sha256(Path(__file__)),
        "operator_library": str(library),
        "operator_library_sha256": sha256(library) if library.is_file() else None,
    }


def run_case(torch, torch_npu, args, pages):
    originals = fixture(torch, pages, args.seed)
    latent, rope = [torch_npu.npu_format_cast(value.to(args.device), 2)
                    for value in originals]
    packed = torch_npu.npu_format_cast(torch.empty(
        (pages, 128, 576), dtype=torch.bfloat16, device=args.device), 2)
    tensors = (latent, rope, packed)
    formats = [torch_npu.get_npu_format(value) for value in tensors]
    if formats != [2, 2, 2] or not all(value.is_contiguous() for value in tensors):
        raise ValueError("packing requires three contiguous ND tensors")
    pointers = [value.data_ptr() for value in tensors]
    if len(set(pointers)) != 3:
        raise ValueError("source caches and shadow must have independent storage")
    expected = torch.cat((originals[0].squeeze(2), originals[1].squeeze(2)), dim=-1)
    stream = torch.npu.Stream(device=args.device)
    stream.wait_stream(torch.npu.current_stream(args.device))

    def direct():
        return pack_into(torch, latent, rope, packed)

    def inspect(reference):
        checks = {"packed": compare_bytes(torch, packed, reference),
                  "latent_unchanged": compare_bytes(torch, latent, originals[0]),
                  "rope_unchanged": compare_bytes(torch, rope, originals[1])}
        if [value.data_ptr() for value in tensors] != pointers:
            raise RuntimeError("packing changed preallocated tensor storage")
        return {"pass": all(value["pass"] for value in checks.values()),
                "boundaries": checks}

    with torch.npu.stream(stream):
        direct()
    stream.synchronize()
    initial = inspect(expected)
    report = {
        "physical_pages": pages, "page_size": 128,
        "shapes": [list(value.shape) for value in tensors],
        "dtypes": [str(value.dtype) for value in tensors],
        "formats": formats, "addresses": pointers,
        "persistent_shadow_bytes": packed.nbytes,
        "logical_read_plus_write_bytes": 2 * packed.nbytes,
        "warmup_per_arm_before_each_sample": args.warmup,
        "pack_operations_per_event": 1, "samples_per_arm": args.samples,
        "initial": initial, "pass": initial["pass"],
        "scope": (
            "One direct torch.cat(out=packed) or one graph replay per event. "
            "Inputs and output use fixed, preallocated ND addresses. Compilation, "
            "allocation, graph capture, warmup and byte checks are outside events. "
            "This is prewarmed initialization-operation cost, not cold first use "
            "or per-decode shadow update. The 2*S payload is logical traffic, "
            "not measured HBM traffic. Direct events can include host enqueue gaps."),
    }
    if not report["pass"]:
        return report

    with torch.npu.stream(stream):
        for _ in range(args.warmup):
            direct()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        direct()
    stream.synchronize()
    # A different payload checks graph input reuse before the timed fixture is restored.
    changed = fixture(torch, pages, args.seed + 1)
    with torch.npu.stream(stream):
        latent.copy_(changed[0])
        rope.copy_(changed[1])
        graph.replay()
    stream.synchronize()
    changed_expected = torch.cat((changed[0].squeeze(2), changed[1].squeeze(2)), dim=-1)
    changed_metric = compare_bytes(torch, packed, changed_expected)
    report["changing_graph_input"] = changed_metric
    report["pass"] = report["pass"] and changed_metric["pass"]
    with torch.npu.stream(stream):
        latent.copy_(originals[0])
        rope.copy_(originals[1])
        graph.replay()
    stream.synchronize()
    report["captured"] = inspect(expected)
    report["pass"] = report["pass"] and report["captured"]["pass"]
    if not report["pass"]:
        return report

    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    with torch.npu.stream(stream):
        begin.record()
        end.record()
    end.synchronize()
    arms = {"direct_event": direct, "graph_replay": graph.replay}
    samples = []
    for sample in range(args.samples):
        order = list(arms)[::1 if sample % 2 == 0 else -1]
        measured = {"order": order}
        for name in order:
            with torch.npu.stream(stream):
                for _ in range(args.warmup):
                    arms[name]()
            stream.synchronize()
            with torch.npu.stream(stream):
                begin.record()
                arms[name]()
                end.record()
            end.synchronize()
            measured[name] = float(begin.elapsed_time(end)) * 1000
        samples.append(measured)
    report["event_samples_us"] = samples
    report["summary"] = {name: summary([sample[name] for sample in samples])
                         for name in arms}
    report["post_timing"] = inspect(expected)
    report["pass"] = report["pass"] and report["post_timing"]["pass"]
    return report


def main():
    args = arguments()
    import torch
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    result = {
        "identity": provenance(torch, torch_npu, args),
        "fixture": {"seed": args.seed,
                    "type": "BF16 random normal with signed-zero/Inf/NaN bit witnesses"},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for pages in args.pages:
        result["cases"].append(run_case(torch, torch_npu, args, pages))
        result["pass"] = all(case["pass"] for case in result["cases"])
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not result["pass"]:
            break
    print(json.dumps({"pass": result["pass"], "output": str(args.output)}))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
