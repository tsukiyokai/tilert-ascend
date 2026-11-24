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
from validate import build_bank

CALLS = 128
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
    torch.npu.set_per_process_memory_fraction(3584 * 2**20 / capacity)
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
              "cases": []}
    for width, distribution in ((1, "random"), (8, "random"),
                                (32, "random"), (32, "shared8"),
                                (32, "full256")):
        rows = width * 8
        frozen = torch.randint(-127, 128, (rows, 128), dtype=torch.int8,
                               generator=generator).to(args.device)
        token_scale = (torch.rand(rows, generator=generator) * .016 + .004).to(args.device)
        routes = []
        for _ in range(CALLS):
            selected = (torch.randperm(256, generator=generator)[:8].repeat(width)
                        if distribution == "shared8" else
                        torch.cat([torch.randperm(256, generator=generator)[:8]
                                   for _ in range(width)]))
            routes.append(torch.ones(256, dtype=torch.int64)
                          if distribution == "full256" else
                          torch.bincount(selected, minlength=256))
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
        exact = all(torch.equal(a.cpu(), b.cpu())
                    for a, b in zip(outputs["base512"], outputs["wide1024"]))
        assert exact, (width, distribution)

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
                  "outputs_exact": exact,
                  "group_counts": [c.tolist() for c in routes],
                  "base512_us": stats([v for r in raw for v in r["base512"]]),
                  "wide1024_us": stats([v for r in raw for v in r["wide1024"]]),
                  "paired_delta_us": stats(deltas),
                  "wide1024_wins": sum(d < 0 for d in deltas),
                  "raw_us": raw, "host_us": stats(host_times)}
        report["cases"].append(record)
        report["peak_reserved_bytes"] = torch.npu.max_memory_reserved()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("group_counts", "raw_us")}), flush=True)
        del graphs, outputs, buffers, result, graph, x, groups
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
