"""Shared-device hot-input attention pipeline screening with ACL graphs."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import torch

from compare_cann_sparse_attention import (
    difference, gathered_fia, partitioned_case, prepare, sfa, split_kv,
)


HELPER_SHA = (
    "57c4b56771cf3ec2f75880e223227207694a664c4c46cdc67237eda8f8c28bc0"
)
ROOT = Path(__file__).parent
CALLS, WARMUP, BATCHES = 10, 10, 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(name):
    print(json.dumps({"stage": name}), flush=True)


def split_pipeline(case, fixed):
    width = case["query"].shape[0]
    split = dict(case)
    split.update(
        query=case["query"].repeat(4, 1, 1),
        query_rope=case["query_rope"].repeat(4, 1, 1),
        sparse_indices=case["sparse_indices"].reshape(
            width, 4, 512).permute(1, 0, 2).contiguous().reshape(
                4 * width, 1, 512),
        block_table=case["block_table"].repeat(4 * width, 1),
        **fixed,
    )
    return split_kv(split)


def summarize(values):
    ordered = sorted(values)

    def quantile(fraction):
        index = (len(ordered) - 1) * fraction
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (index - low) * (ordered[high] - ordered[low])

    return {"count": len(values), "min": min(values), "max": max(values),
            "mean": statistics.mean(values), "std": statistics.pstdev(values),
            **{f"p{int(p * 100)}": quantile(p)
               for p in (0.5, 0.9, 0.95, 0.99)}}


def write_report(directory, report, outputs):
    torch.save(outputs, directory / "outputs.pt")
    report["outputs_sha256"] = sha(directory / "outputs.pt")
    (directory / "result.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input-label", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    helper = ROOT / "compare_cann_sparse_attention.py"
    assert sha(helper) == HELPER_SHA
    torch.set_num_threads(1)
    stage("prepare_cpu")
    case = prepare(args.input)
    width = case["query"].shape[0]
    split_cpu = partitioned_case(case)
    length_keys = ("actual_seq_lengths_query", "actual_seq_lengths_kv")
    fixed_cpu = {key: split_cpu[key] for key in length_keys}
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    report = {
        "status": "prepared", "scope": "shared-device hot-input screening",
        "precision_verified": False, "decoder_performance_measured": False,
        "query_count": width,
        "input_kind": args.input_label,
        "input": str(args.input), "input_sha256": sha(args.input),
        "source_sha256": sha(__file__), "helper_sha256": HELPER_SHA,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "graph_api_sha256": sha(inspect.getfile(torch_npu.npu.NPUGraph)),
        "event_api_sha256": sha(inspect.getfile(torch_npu.npu.Event)),
        "environment": {key: os.environ.get(key) for key in (
            "TASK_QUEUE_ENABLE", "PYTORCH_NPU_ALLOC_CONF",
            "ASCEND_CUSTOM_OPP_PATH", "ASCEND_GLOBAL_LOG_LEVEL",
            "ASCEND_SLOG_PRINT_TO_STDOUT")},
        "calls_per_graph": CALLS, "warmup_graphs_per_pipeline": WARMUP,
        "sample_batches": BATCHES,
        "timing_scope": "Same-stream events around one replay of ten calls; "
                        "event_ms_per_call is graph duration / 10, including "
                        "graph scheduling and submission gaps between events. "
                        "Not bare kernel latency or decoder TPOT.",
        "split_included": "Q/RoPE repeat, indices permutation/contiguous, "
                          "block_table repeat, SFA statistics, log/LSE, "
                          "attention_update and BF16 output cast",
        "split_fixed_npu_lengths": {key: value.tolist()
                                    for key, value in fixed_cpu.items()},
    }
    stage("device_initialize")
    torch_npu.npu.set_device(args.device)
    total = torch_npu.npu.get_device_properties(args.device).total_memory
    torch_npu.npu.set_per_process_memory_fraction(1.5 * 2**30 / total)
    torch.npu.config.allow_internal_format = True
    assert enable_custom_op()
    report["fia_schema"] = str(
        torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema)
    report["sfa_schema"] = str(
        torch.ops._C_ascend.npu_sparse_flash_attention.default._schema)
    device = torch.device("npu", args.device)
    stream = torch_npu.npu.Stream()
    graphs, outputs, verification = {}, {}, {}
    with torch.inference_mode(), torch_npu.npu.stream(stream):
        case = {key: value.to(device) for key, value in case.items()}
        fixed = {key: value.to(device) for key, value in fixed_cpu.items()}
        operations = {
            "sfa": lambda: sfa(case)[0],
            "gathered_fia": lambda: gathered_fia(case),
            "split_kv": lambda: split_pipeline(case, fixed),
        }
        for name, operation in operations.items():
            stage(f"eager_and_capture_{name}")
            value = operation()
            stream.synchronize()
            outputs[f"eager_{name}"] = value.cpu().clone()
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph, stream=stream):
                captured = [operation() for _ in range(CALLS)]
            graph.replay()
            stream.synchronize()
            outputs[f"graph_{name}"] = [value.cpu() for value in captured]
            verification[name] = [difference(value, outputs[f"eager_{name}"])
                                  for value in outputs[f"graph_{name}"]]
            graphs[name] = (graph, captured)
        report["graph_vs_eager"] = verification
        if not all(item["bit_equal"] for checks in verification.values()
                   for item in checks):
            report["status"] = "graph_eager_mismatch"
            write_report(args.output, report, outputs)
            return 1
        names = tuple(graphs)
        stage("warmup")
        for batch in range(WARMUP):
            for offset in range(3):
                graphs[names[(batch + offset) % 3]][0].replay()
        stream.synchronize()
        start, end = (torch_npu.npu.Event(enable_timing=True)
                      for _ in range(2))
        samples = []
        stage("timed_interleaved_batches")
        for batch in range(BATCHES):
            for offset in range(3):
                name = names[(batch + offset) % 3]
                stream.synchronize()
                begin = time.perf_counter()
                start.record(stream)
                graphs[name][0].replay()
                end.record(stream)
                end.synchronize()
                wall_ms = (time.perf_counter() - begin) * 1000
                event_ms = start.elapsed_time(end)
                samples.append({"batch": batch, "order": offset,
                                "pipeline": name, "graph_event_ms": event_ms,
                                "event_ms_per_call": event_ms / CALLS,
                                "wall_ms_per_graph": wall_ms})
        report.update(status="completed", raw_samples=samples,
                      summaries_ms_per_call={name: summarize([
                          item["event_ms_per_call"] for item in samples
                          if item["pipeline"] == name]) for name in names},
                      graph_pools={name: list(pair[0].pool())
                                   for name, pair in graphs.items()},
                      peak_reserved_bytes=torch_npu.npu.max_memory_reserved())
        report["after_timing_checks"] = {}
        for name, (_, captured) in graphs.items():
            values = [value.cpu() for value in captured]
            outputs[f"after_timing_{name}"] = values
            report["after_timing_checks"][name] = [
                difference(value, outputs[f"eager_{name}"]) for value in values
            ]
        valid = all(item["bit_equal"] for checks in
                    report["after_timing_checks"].values() for item in checks)
        report["timing_outputs_verified"] = valid
        if not valid:
            report["status"] = "post_timing_output_mismatch"
        for graph, _ in graphs.values():
            graph.reset()
    write_report(args.output, report, outputs)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
