"""Paired graph screen of the GMM2 traversal; each call restores its input."""

import argparse
import json
import os
from pathlib import Path
import statistics
import time

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
import torch_npu
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validate import build_bank, digest

CALLS = 10
BANKS = 8


def stats(values):
    ordered = sorted(values)
    return {"p50": statistics.median(values),
            "p90": ordered[round((len(values) - 1) * .9)],
            "p99": ordered[round((len(values) - 1) * .99)],
            "std": statistics.stdev(values)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:15")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(2048 * 2**20 / capacity)
    torch.npu.reset_peak_memory_stats()
    banks = [build_bank(args.device)[:3] for _ in range(BANKS)]
    assert all(torch_npu.get_npu_format(b[0][0]) == 29 for b in banks)
    generator = torch.Generator().manual_seed(20261020)
    report = {"scope": "one-die shared-load screen, not request TPOT",
              "device": args.device, "calls_per_graph": CALLS,
              "banks": BANKS, "warmup_pairs": 10, "abba_samples": 30,
              "outliers": "none discarded", "clocks_locked": False,
              "timing": "NPU events, profiler off; includes x restore",
              "test_switch": "group-list pointer bit5; both use compact expert table",
              "quota_bytes": 2048 * 2**20, "cases": [],
              "counts_sha256": hashlib.sha256(Path(__file__).with_name("counts.json").read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "validate_sha256": hashlib.sha256((Path(__file__).resolve().parent.parent / "validate.py").read_bytes()).hexdigest()}
    fixture = json.loads(Path(__file__).with_name("counts.json").read_text())
    report["counts_scope"] = fixture["scope"]
    report["data_scope"] = "synthetic concentration boundary; seeded synthetic x/weights/scales"
    for case in fixture["cases"]:
        width, distribution = 32, case["name"]
        rows = 256
        frozen = torch.randint(-127, 128, (rows, 128), dtype=torch.int8,
                               generator=generator).to(args.device)
        token_scale = (torch.rand(rows, generator=generator) * .016 + .004).to(args.device)
        routes = [torch.tensor(case["counts"], dtype=torch.int64)] * CALLS
        graphs, outputs, buffers = {}, {}, []
        for arm, offset in (("base512", 4), ("wide1024", 0)):
            x = frozen.clone()
            groups = []
            for counts in routes:
                allocation = torch.zeros(260, dtype=torch.int64,
                                         device=args.device)
                view = allocation[offset:offset + 256]
                view.copy_(counts)
                assert view.data_ptr() % 64 == offset * 8
                groups.append(view)
            buffers.append((x, groups))

            def call(index):
                w, s, b = banks[index % BANKS]
                x.copy_(frozen)
                return torch_npu.npu_grouped_matmul(
                    x=[x], weight=w, scale=s, bias=b,
                    per_token_scale=[token_scale], group_list=groups[index],
                    split_item=2, group_type=0, group_list_type=1,
                    output_dtype=torch.bfloat16)[0]

            call(0)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = [call(i) for i in range(CALLS)]
            graph.replay()
            torch.npu.synchronize()
            graphs[arm], outputs[arm] = graph, result
        cpu_outputs = {arm: [value.cpu() for value in values]
                       for arm, values in outputs.items()}
        exact = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                    for a, b in zip(cpu_outputs["base512"],
                                    cpu_outputs["wide1024"]))
        assert exact, (width, distribution)
        assert all(torch.isfinite(v).all().item()
                   for values in cpu_outputs.values() for v in values)
        output_hashes = {arm: [digest(v) for v in values]
                         for arm, values in cpu_outputs.items()}

        def measure(arm):
            torch.npu.synchronize()
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            host_start = time.perf_counter()
            graphs[arm].replay()
            host = (time.perf_counter() - host_start) * 1e6 / CALLS
            end.record()
            torch.npu.synchronize()
            return start.elapsed_time(end) * 1000 / CALLS, host

        for _ in range(10):
            measure("base512")
            measure("wide1024")
        raw, host_times = [], []
        for _ in range(30):
            samples = {"base512": [], "wide1024": []}
            for arm in ("base512", "wide1024", "wide1024", "base512"):
                value, host = measure(arm)
                samples[arm].append(value)
                host_times.append(host)
            raw.append(samples)
        deltas = [statistics.fmean(r["wide1024"]) - statistics.fmean(r["base512"])
                  for r in raw]
        record = {"width": width, "distribution": distribution,
                  "outputs_exact": exact, "output_sha256": output_hashes,
                  "x_sha256": digest(frozen),
                  "token_scale_sha256": digest(token_scale),
                  "group_counts": case["counts"],
                  "active": case["active"], "hot": case["hot"],
                  "maximum": case["maximum"],
                  "compact_tasks": case["compact"]["tasks"],
                  "wide_tasks": case["wide"]["tasks"],
                  "base512_us": stats([v for r in raw for v in r["base512"]]),
                  "wide1024_us": stats([v for r in raw for v in r["wide1024"]]),
                  "paired_delta_us": stats(deltas),
                  "wide1024_wins": sum(d < 0 for d in deltas),
                  "raw_us": raw, "host_us": stats(host_times)}
        report["cases"].append(record)
        report["peak_reserved_bytes"] = torch.npu.max_memory_reserved()
        report["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("group_counts", "raw_us")}), flush=True)
        del graphs, outputs, buffers, result, graph, x, groups
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
