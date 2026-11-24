"""Screen SFA and direct INT32 gather/FIA on one identity cache."""

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch

from bench_cann_sparse_attention import (
    BATCHES, CALLS, WARMUP, sha, stage, summarize, write_report,
)
from compare_cann_sparse_attention import difference, prepare, sfa


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from tilert.models.glm_5_2_ascend.cann_attention import (  # noqa: E402
    gathered_fia, install_gather_fia,
)


def make_cases(path):
    original = prepare(path)
    pages = original["block_table"][0].long()
    for key in ("key", "key_rope"):
        original[key] = original[key].index_select(0, pages)
    original["block_table"] = torch.arange(
        pages.numel(), dtype=torch.int32).unsqueeze(0)
    changed = {key: value.clone() for key, value in original.items()}
    changed["query"] = (-original["query"].float() + 0.125).bfloat16()
    changed["query_rope"] = (
        original["query_rope"].float().flip(-1) - 0.25).bfloat16()
    changed["key"] = (original["key"].float() + 0.125).bfloat16()
    changed["key_rope"] = (original["key_rope"].float() - 0.25).bfloat16()
    for row in range(original["query"].shape[0]):
        changed["sparse_indices"][row] = (
            original["sparse_indices"][row] + 17).remainder(8193 + row)
        assert changed["sparse_indices"][row].unique().numel() == 2048
    for key in ("query", "query_rope", "key", "key_rope", "sparse_indices"):
        assert not torch.equal(original[key], changed[key])
    return original, changed


def copy_case(destination, source):
    for key, value in source.items():
        destination[key].copy_(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-label", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    case_a, case_b = make_cases(args.input)
    width = case_a["query"].shape[0]

    import torch_npu
    from vllm_ascend.utils import enable_custom_op

    report = {
        "status": "prepared", "scope": "shared-device hot-input screening",
        "precision_verified": False, "decoder_performance_measured": False,
        "query_count": width, "input_kind": args.input_label,
        "input": str(args.input), "input_sha256": sha(args.input),
        "source_sha256": sha(__file__),
        "helper_sha256": {name: sha(ROOT / name) for name in (
            "bench_cann_sparse_attention.py",
            "compare_cann_sparse_attention.py")},
        "attention_source_sha256": sha(inspect.getfile(gathered_fia)),
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "graph_api_sha256": sha(inspect.getfile(torch_npu.npu.NPUGraph)),
        "event_api_sha256": sha(inspect.getfile(torch_npu.npu.Event)),
        "environment": {key: os.environ.get(key) for key in (
            "TASK_QUEUE_ENABLE", "PYTORCH_NPU_ALLOC_CONF",
            "ASCEND_CUSTOM_OPP_PATH", "ASCEND_GLOBAL_LOG_LEVEL",
            "ASCEND_SLOG_PRINT_TO_STDOUT")},
        "calls_per_graph": CALLS, "warmup_graphs_per_pipeline": WARMUP,
        "sample_batches": BATCHES,
        "cache_contract": "Single request, physical token = logical token",
        "timing_scope": "Same-stream event duration for ten graph calls /10; "
                        "includes gather/FIA and graph scheduling. Shared "
                        "device screening, not bare FIA latency or TPOT.",
        "direct_included": "Two INT32 index_select allocations and gathers "
                           "inside capture, output views and FIA. No div, "
                           "table lookup, remainder or dtype conversion.",
        "allocation_scope": "Direct outputs use the actual allocation API "
                            "during capture; graph replay reuses its pool. "
                            "No separate eager allocator latency is inferred.",
        "fixed_metadata": {"sfa_query_lengths": [width],
                           "sfa_kv_lengths": [8192 + width],
                           "fia_query_lengths": [1] * width,
                           "fia_kv_lengths": [2048] * width},
    }
    stage("device_initialize")
    torch_npu.npu.set_device(args.device)
    total = torch_npu.npu.get_device_properties(args.device).total_memory
    torch_npu.npu.set_per_process_memory_fraction(1.5 * 2**30 / total)
    torch.npu.config.allow_internal_format = True
    assert enable_custom_op()
    report["schemas"] = {
        "sfa": str(
            torch.ops._C_ascend.npu_sparse_flash_attention.default._schema),
        "fia": str(
            torch.ops.npu.npu_fused_infer_attention_score_v2.default._schema),
    }
    report["environment"]["ASCEND_CUSTOM_OPP_PATH"] = os.environ.get(
        "ASCEND_CUSTOM_OPP_PATH")
    device = torch.device("npu", args.device)
    stream = torch_npu.npu.Stream()
    graphs, outputs, checks = {}, {}, {}
    with torch.inference_mode(), torch_npu.npu.stream(stream):
        case = {key: value.to(device) for key, value in case_a.items()}
        addresses = {key: value.data_ptr() for key, value in case.items()}
        impl = SimpleNamespace(scale=1 / 16)
        install_gather_fia(impl)
        operations = {
            "sfa": lambda: sfa(case)[0],
            "direct_fia": lambda: impl._execute_sparse_flash_attention_process(
                case["query"], case["query_rope"],
                (case["key"], case["key_rope"]), case["sparse_indices"],
                None, case["actual_seq_lengths_query"],
                case["actual_seq_lengths_kv"],
            ),
        }
        for label, source in (("a", case_a), ("b", case_b)):
            copy_case(case, source)
            for name, operation in operations.items():
                value = operation()
                stream.synchronize()
                outputs[f"eager_{label}_{name}"] = value.cpu().clone()
        report["a_b_distinguishable"] = {
            name: not torch.equal(outputs[f"eager_a_{name}"],
                                  outputs[f"eager_b_{name}"])
            for name in operations}
        assert all(report["a_b_distinguishable"].values())
        report["cross_kernel_diagnostics"] = {label: difference(
            outputs[f"eager_{label}_direct_fia"],
            outputs[f"eager_{label}_sfa"])
            for label in ("a", "b")}
        copy_case(case, case_a)
        for name, operation in operations.items():
            stage(f"capture_{name}")
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph, stream=stream):
                captured = [operation() for _ in range(CALLS)]
            graphs[name] = (graph, captured)
        for label, source in (("a", case_a), ("b", case_b),
                              ("reset_a", case_a)):
            stage(f"dynamic_replay_{label}")
            copy_case(case, source)
            for name, (graph, captured) in graphs.items():
                if label == "reset_a":
                    for value in captured:
                        value.fill_(float("nan"))
                graph.replay()
                stream.synchronize()
                values = [value.cpu() for value in captured]
                outputs[f"graph_{label}_{name}"] = values
                eager_label = "b" if label == "b" else "a"
                reference = outputs[f"eager_{eager_label}_{name}"]
                checks[f"{label}_{name}"] = [
                    difference(value, reference) for value in values]
        report["dynamic_checks"] = checks
        report["input_addresses_stable"] = all(
            value.data_ptr() == addresses[key] for key, value in case.items())
        valid = (report["input_addresses_stable"] and all(
            item["bit_equal"] for values in checks.values()
            for item in values))
        if not valid:
            report["status"] = "dynamic_replay_mismatch"
            write_report(args.output, report, outputs)
            return 1
        report["graph_pools"] = {name: list(pair[0].pool())
                                 for name, pair in graphs.items()}
        names = tuple(graphs)
        stage("warmup")
        for batch in range(WARMUP):
            for offset in range(2):
                graphs[names[(batch + offset) % 2]][0].replay()
        stream.synchronize()
        start, end = (torch_npu.npu.Event(enable_timing=True)
                      for _ in range(2))
        samples = []
        stage("timed_interleaved_batches")
        for batch in range(BATCHES):
            for offset in range(2):
                name = names[(batch + offset) % 2]
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
        report["after_timing_checks"] = {}
        for name, (graph, captured) in graphs.items():
            values = [value.cpu() for value in captured]
            outputs[f"after_timing_{name}"] = values
            report["after_timing_checks"][name] = [difference(
                value, outputs[f"eager_a_{name}"]) for value in values]
            graph.reset()
        report["input_unchanged"] = all(torch.equal(value.cpu(), case_a[key])
                                        for key, value in case.items())
        valid = report["input_unchanged"] and all(
            item["bit_equal"] for values in
            report["after_timing_checks"].values() for item in values)
        report.update(
            status="completed" if valid else "post_timing_output_mismatch",
            timing_outputs_verified=valid, raw_samples=samples,
            summaries_ms_per_call={name: summarize([
                item["event_ms_per_call"] for item in samples
                if item["pipeline"] == name]) for name in names},
            peak_reserved_bytes=torch_npu.npu.max_memory_reserved(),
        )
    write_report(args.output, report, outputs)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
