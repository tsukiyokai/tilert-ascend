"""Compare two input quantizers directly, including nonfinite BF16 payloads."""

import argparse
import hashlib
import json
from pathlib import Path


HIDDEN = 6144
OUTPUTS = ("shared_codes", "shared_scale", "routed_codes", "routed_scale")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_bytes(torch, value):
    return value.detach().cpu().reshape(-1).clone(
        memory_format=torch.contiguous_format).view(torch.uint8)


def compare(torch, actual, expected):
    same = actual.dtype == expected.dtype and actual.shape == expected.shape
    different = None
    if same:
        different = int((raw_bytes(torch, actual) != raw_bytes(torch, expected)).sum())
    return {"bitwise_equal": same and different == 0,
            "different_bytes": different}


def fixtures(torch, rows):
    column = torch.arange(HIDDEN, dtype=torch.int32).expand(rows, -1)
    row = torch.arange(rows, dtype=torch.int32)[:, None]
    finite = (0x3B00 + (column + row * 17) % 384) | ((column % 2) * 0x8000)
    cases = []

    def append(name, bits):
        cases.append((name, bits.to(torch.int16).view(torch.bfloat16).clone()))

    append("positive_zero", torch.zeros_like(finite))
    append("negative_zero", torch.full_like(finite, 0x8000))
    append("mixed_signed_zero", (column % 2) * 0x8000)
    append("distinct_scale_witness", torch.full_like(finite, 0x3C2F))
    append("finite_large", torch.where(column % 2 == 0, 0x7F7F, 0xFF7F))
    tiny = torch.tensor([0x0001, 0x007F, 0x0080, 0x8001, 0x807F, 0x8080],
                        dtype=torch.int32)
    append("finite_tiny", tiny[column % len(tiny)])
    boundaries = (0, 63, 64, 191, 192, 4095, 6143)
    for name, payloads in (
            ("positive_infinity", (0x7F80,)),
            ("signed_infinity", (0x7F80, 0xFF80)),
            ("quiet_nan_payloads", (0x7FC1, 0x7FD3, 0xFFC7)),
            ("signaling_nan_payloads", (0x7F81, 0x7F93, 0xFF87)),
            ("mixed_nonfinite", (0x7FC1, 0x7F80, 0x7F81, 0xFF80))):
        bits = finite.clone()
        for index, position in enumerate(boundaries):
            bits[:, position] = payloads[index % len(payloads)]
        append(name, bits)
    append("finite_mixed", finite)
    return cases


def views(bridge, body):
    return dict(zip(OUTPUTS, (bridge[6], bridge[7], body[10], body[11])))


def poison(torch, tensors, seed):
    for value in tensors.values():
        if value.dtype == torch.int8:
            value.fill_(17 + seed)
        else:
            value.view(torch.int32).fill_(0x7FC01000 + seed)


def verify_build(path, library):
    manifest = json.loads((path / "manifest.json").read_text())
    for kind in ("sources", "artifacts"):
        for name, expected in manifest[kind].items():
            if digest(path / name) != expected:
                raise RuntimeError(f"Frozen build differs: {path / name}")
    return {"path": str(path), "library_sha256": digest(path / library),
            "manifest_sha256": digest(path / "manifest.json")}


def run_case(torch, ops, prepare, rows, blocks, report):
    sample = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device="npu")
    arenas = [prepare(sample, blocks, 1) for _ in range(2)]
    graphs = []
    consumers = []
    for arm, (bridge, body, sync) in enumerate(arenas):
        bridge[0].copy_(sample)
        bridge[4].fill_(0.125)
        bridge[5].copy_(torch.arange(8, device="npu", dtype=torch.int32).expand(rows, -1))
        output = views(bridge, body)
        consumer = {name: torch.empty_like(value) for name, value in output.items()}
        ops.forward_out(bridge, body, blocks, sync, bool(arm))
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            ops.forward_out(bridge, body, blocks, sync, bool(arm))
            for name, value in output.items():
                consumer[name].copy_(value)
        graphs.append(graph)
        consumers.append(consumer)
    report["captures"] += 2
    try:
        for index, (name, input_cpu) in enumerate(fixtures(torch, rows)):
            choices = (torch.arange(rows, dtype=torch.int32)[:, None] * 19
                       + torch.arange(8, dtype=torch.int32)[None, :] * 23
                       + index * 7) % 256
            first = None
            for repeat in range(2):
                snapshots = []
                checks = {}
                for arm, ((bridge, body, _), graph, consumer) in enumerate(
                        zip(arenas, graphs, consumers)):
                    bridge[0].copy_(input_cpu)
                    bridge[5].copy_(choices)
                    bridge[4].fill_(0.125)
                    output = views(bridge, body)
                    poison(torch, output, 31 * arm + repeat)
                    poison(torch, consumer, 61 + 17 * arm + repeat)
                    graph.replay()
                    torch.npu.synchronize()
                    report["replays"] += 1
                    snapshot = {key: value.detach().cpu().clone()
                                for key, value in output.items()}
                    snapshots.append(snapshot)
                    checks[f"input.{arm}"] = compare(torch, bridge[0], input_cpu)
                    for key, value in consumer.items():
                        checks[f"consumer.{arm}.{key}"] = compare(
                            torch, value, snapshot[key])
                for key in OUTPUTS:
                    checks[key] = compare(torch, snapshots[1][key], snapshots[0][key])
                if first is not None:
                    for arm in range(2):
                        for key in OUTPUTS:
                            checks[f"fresh_write.{arm}.{key}"] = compare(
                                torch, snapshots[arm][key], first[arm][key])
                first = snapshots
                passed = all(item["bitwise_equal"] for item in checks.values())
                record = {"rows": rows, "blocks": blocks, "fixture": name,
                          "repeat": repeat, "checks": checks, "pass": passed}
                report["cases"].append(record)
                if not passed:
                    return {"input": input_cpu, "choices": choices,
                            "outputs": snapshots, "record": record}
    finally:
        for graph in graphs:
            graph.reset()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--probe-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="1,2,12,32")
    parser.add_argument("--blocks", default="16,24")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    report = {
        "scope": "Direct input quantization; fixed legal metadata; no norm, TopK or FFN",
        "candidate": verify_build(args.candidate_build, "ffn_dual_quant.so"),
        "probe": verify_build(args.probe_build, "quant_probe.so"),
        "checker_sha256": digest(Path(__file__)),
        "captures": 0, "replays": 0, "cases": [], "pass": False,
        "timing": "not measured", "fresh_write": "two different poisons per fixture",
    }
    import torch
    import torch_npu

    torch.ops.load_library(str(args.candidate_build / "ffn_dual_quant.so"))
    torch.ops.load_library(str(args.probe_build / "quant_probe.so"))
    report.update({"torch": torch.__version__, "torch_npu": torch_npu.__version__})
    torch.npu.set_device(args.device)
    report["device"] = {"index": args.device, "name": torch.npu.get_device_name()}
    try:
        report["cann"] = str(torch_npu.utils.get_cann_version())
    except (AttributeError, TypeError, RuntimeError, UnicodeError) as error:
        report["cann_query"] = type(error).__name__ + ": " + str(error)
    prepare = torch.ops.tilert_ffn_dual_quant.prepare
    ops = torch.ops.tilert_ffn_dual_quant_probe
    failure = None
    for blocks in map(int, args.blocks.split(",")):
        for rows in map(int, args.rows.split(",")):
            failure = run_case(torch, ops, prepare, rows, blocks, report)
            if failure is not None:
                break
        if failure is not None:
            break
    report["pass"] = failure is None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if failure is not None:
        dump = args.output.with_suffix(".pt")
        torch.save(failure, dump)
        report["failure_dump"] = str(dump)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "cases": len(report["cases"]),
                      "captures": report["captures"], "replays": report["replays"]}))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
