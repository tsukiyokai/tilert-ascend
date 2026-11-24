"""Paired graph replay of dense versus compact GMM1 expert traversal."""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
import torch_npu

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gmm1_inputs
from vllm_ascend.utils import enable_custom_op


def stats(values):
    ordered = sorted(values)
    return {"p50": statistics.median(values),
            "p90": ordered[round((len(ordered) - 1) * .90)],
            "p99": ordered[round((len(ordered) - 1) * .99)],
            "std": statistics.stdev(values)}


ARMS = ("stock128", "stream64", "full64")
TAGS = {"stock128": 0, "stream64": 32, "full64": 64}


def tagged_counts(counts, device, tag):
    backing = torch.empty(272, dtype=torch.int64, device=device)
    offset = ((tag - (backing.data_ptr() & 127)) & 127) // 8
    result = backing[offset:offset + 256]
    assert result.data_ptr() & 96 == tag
    result.copy_(counts)
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:15")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-union", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    capacity = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(2048 * 2**20 / capacity)
    torch.npu.reset_peak_memory_stats()
    assert enable_custom_op()
    inputs = gmm1_inputs.build(args.device, active=256, banks=2)
    op = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2
    report = {"scope": "single-die shared-load screen; not decoder TPOT",
              "hostname": platform.node(), "device": args.device,
              "warmup": 5, "samples": 10, "calls_per_graph": 32,
              "order": "ABBA", "clocks_locked": False,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
              "log_level": os.environ.get("ASCEND_GLOBAL_LOG_LEVEL"),
              "tag_contract": TAGS, "cases": []}
    generator = torch.Generator().manual_seed(20261020)
    cases = ([(32, "full256")] if args.full_union else
             [(1, "random"), (8, "random"), (32, "random"), (32, "shared8")])
    for width, distribution in cases:
        rows = width * 8
        arguments = []
        counts_list = []
        for call in range(32):
            selected = (torch.randperm(256, generator=generator)[:8].repeat(width)
                        if distribution == "shared8" else
                        torch.cat([torch.randperm(256, generator=generator)[:8]
                                   for _ in range(width)]))
            counts = (torch.ones(256, dtype=torch.int64) if distribution == "full256"
                      else torch.bincount(selected, minlength=256))
            counts_list.append(counts.tolist())
            arguments.append(inputs.kwargs(call % 2) | {
                "x": inputs.x[:rows].clone(),
                "x_scale": inputs.x_scale[:rows],
                "group_list": counts.to(args.device)})
        graphs, outputs = {}, {}
        tagged_arguments = {arm: [kwargs | {"group_list": tagged_counts(kwargs["group_list"], args.device, TAGS[arm])}
                                  for kwargs in arguments] for arm in ARMS}
        for arm in ARMS:
            limit = 0.0
            arm_arguments = tagged_arguments[arm]
            for kwargs in arm_arguments:
                op(**(kwargs | {"swiglu_limit": limit}))
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                pairs = [op(**(kwargs | {"swiglu_limit": limit}))
                         for kwargs in arm_arguments]
            graphs[arm], outputs[arm] = graph, pairs
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
        exact = all(torch.equal(a.cpu(), b.cpu())
                    for arm in ("stream64", "full64")
                    for pair_a, pair_b in zip(outputs["stock128"], outputs[arm])
                    for a, b in zip(pair_a, pair_b))
        if not exact:
            raise RuntimeError("Captured outputs differ")

        def measure(arm):
            torch.npu.synchronize()
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin.record()
            wall = time.perf_counter()
            graphs[arm].replay()
            host = (time.perf_counter() - wall) * 1e6 / 32
            end.record()
            torch.npu.synchronize()
            return begin.elapsed_time(end) * 1000 / 32, host

        for _ in range(5):
            for arm in ARMS:
                measure(arm)
        raw, host_cost = [], []
        for _ in range(10):
            samples = {"residency": {"stream64": [], "full64": []},
                       "package": {"stock128": [], "full64": []}}
            for comparison, control in (("residency", "stream64"), ("package", "stock128")):
                for arm in (control, "full64", "full64", control):
                    value, host = measure(arm)
                    samples[comparison][arm].append(value)
                    host_cost.append(host)
            raw.append(samples)
        comparisons = {}
        for comparison, control in (("residency", "stream64"), ("package", "stock128")):
            differences = [statistics.fmean(r[comparison]["full64"])
                           - statistics.fmean(r[comparison][control]) for r in raw]
            comparisons[comparison] = {
                "control": control,
                "control_us": stats([v for r in raw for v in r[comparison][control]]),
                "full64_us": stats([v for r in raw for v in r[comparison]["full64"]]),
                "paired_delta_us": stats(differences),
                "full64_wins": sum(v < 0 for v in differences),
            }
        record = {"width": width, "rows": rows, "distribution": distribution,
                  "outputs_exact": exact, "group_counts": counts_list,
                  "group_pointer_low7": {arm: [kw["group_list"].data_ptr() & 127 for kw in tagged_arguments[arm]] for arm in ARMS},
                  "active_experts": [sum(v > 0 for v in c) for c in counts_list],
                  "max_expert_count": max(max(c) for c in counts_list),
                  "comparisons": comparisons, "host_us": stats(host_cost), "raw_us": raw}
        report["cases"].append(record)
        report["peak_reserved_bytes"] = torch.npu.max_memory_reserved()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items()
                          if k not in ("raw_us", "group_counts", "active_experts")}),
              flush=True)
        del graphs, outputs, arguments, tagged_arguments, arm_arguments, pairs, graph
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
