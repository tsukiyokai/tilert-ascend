"""Measure one production GMM2 ELF in the four-process ABBA protocol."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validate import build_bank, digest

CALLS, BANKS, WARMUP, SAMPLES = 10, 8, 10, 30
QUOTA = 2 * 1024**3


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    ordered = sorted(values)
    return {"p50": statistics.median(values),
            "p90": ordered[round((len(values) - 1) * .9)],
            "p99": ordered[round((len(values) - 1) * .99)],
            "std": statistics.stdev(values)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A1", "B1", "B2", "A2"), required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    fixture = json.loads((root / "counts.json").read_text())
    identity = json.loads((root / "identity.json").read_text())[args.arm[0]]
    selection_field = "new_expected_wide" if args.arm[0] == "B" else "old_expected_wide"
    binary = Path(identity["binary"])
    assert file_hash(binary) == identity["elf_sha256"]
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    total = torch.npu.get_device_properties(args.device).total_memory
    torch.npu.set_per_process_memory_fraction(QUOTA / total)
    torch.npu.reset_peak_memory_stats()
    banks = [build_bank(args.device)[:3] for _ in range(BANKS)]
    pointers = [bank[0][0].data_ptr() for bank in banks]
    assert len(set(pointers)) == BANKS
    assert all(torch_npu.get_npu_format(bank[0][0]) == 29 for bank in banks)
    generator = torch.Generator().manual_seed(20261020)
    report = {"scope": "Four-process ABBA, one shared die; event time includes input restoration",
              "arm": args.arm, "device": args.device, "calls_per_graph": CALLS,
              "banks": BANKS, "weight_pointers": pointers, "warmup": WARMUP,
              "samples": SAMPLES, "quota_bytes": QUOTA, "clocks_locked": False,
              "profiler": False, "outliers": "none discarded",
              "data_scope": fixture["scope"], "opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
              "elf_path": str(binary), "elf_sha256": identity["elf_sha256"],
              "descriptor_sha256": file_hash(binary.with_suffix(".json")),
              "probe_sha256": file_hash(Path(__file__)),
              "fixture_sha256": file_hash(root / "counts.json"),
              "bank_builder_sha256": file_hash(Path(build_bank.__code__.co_filename)),
              "cases": []}
    for case in fixture["cases"]:
        frozen = torch.randint(-127, 128, (256, 128), dtype=torch.int8,
                               generator=generator).to(args.device)
        x = frozen.clone()
        scale = (torch.rand(256, generator=generator) * .016 + .004).to(args.device)
        counts_cpu = torch.tensor(case["counts"], dtype=torch.int64)
        groups = counts_cpu.to(args.device)

        def call(index):
            weight, packed_scale, bias = banks[index % BANKS]
            x.copy_(frozen)
            return torch_npu.npu_grouped_matmul(
                x=[x], weight=weight, scale=packed_scale, bias=bias,
                per_token_scale=[scale], group_list=groups,
                split_item=2, group_type=0, group_list_type=1,
                output_dtype=torch.bfloat16)[0]

        eager = call(0).cpu()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = [call(i) for i in range(CALLS)]
        for _ in range(WARMUP):
            graph.replay()
        torch.npu.synchronize()
        samples, host_samples = [], []
        for _ in range(SAMPLES):
            begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            begin.record()
            start = time.perf_counter()
            graph.replay()
            host_samples.append((time.perf_counter() - start) * 1e6 / CALLS)
            end.record()
            torch.npu.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / CALLS)
        cpu_outputs = [value.cpu() for value in outputs]
        assert torch.equal(eager.view(torch.uint8), cpu_outputs[0].view(torch.uint8))
        assert all(torch.isfinite(value).all().item() for value in cpu_outputs)
        combined = hashlib.sha256()
        for value in cpu_outputs:
            combined.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        result = {"name": case["name"], "active": case["active"], "hot": case["hot"],
                  "maximum": case["maximum"],
                  "source_predicted_wide": case[selection_field],
                  "us": stats(samples), "raw_us": samples,
                  "host_us": stats(host_samples), "raw_host_us": host_samples,
                  "output_sha256": [digest(value) for value in cpu_outputs],
                  "output_concat_sha256": combined.hexdigest(),
                  "route_sha256": digest(counts_cpu.repeat(CALLS, 1)),
                  "counts_sha256": digest(counts_cpu),
                  "x_sha256": digest(frozen), "token_scale_sha256": digest(scale),
                  "eager_graph_first_exact": True}
        report["cases"].append(result)
        report["peak_allocated_bytes"] = torch.npu.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.npu.max_memory_reserved()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("raw_us", "raw_host_us", "output_sha256")}), flush=True)
        del graph, outputs, cpu_outputs, groups, x, eager
    report["status"] = "pass"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
