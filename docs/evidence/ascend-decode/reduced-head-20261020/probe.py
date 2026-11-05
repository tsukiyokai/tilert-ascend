"""TP2 head selection probe with the production 9680-column shard shape."""

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parent
SHARD = 9680
HIDDEN = 6144
ARMS = ("full", "reduced")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def prepare(checkpoint):
    import torch
    import torch_npu
    from safetensors import safe_open
    from vllm.distributed import (
        destroy_distributed_environment, destroy_model_parallel,
        init_distributed_environment, initialize_model_parallel,
    )
    from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor

    torch.set_num_threads(1)
    assert not torch.npu.is_initialized()
    index = checkpoint / "quant_model_weights.safetensors.index.json"
    filename = json.loads(index.read_text())["weight_map"]["lm_head.weight"]
    source = checkpoint / filename
    spec = importlib.util.find_spec(
        "vllm.model_executor.layers.logits_processor")
    hashes = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        view = handle.get_slice("lm_head.weight")
        assert view.get_shape() == [154880, HIDDEN]
        for rank in range(2):
            weight = view[rank * SHARD:(rank + 1) * SHARD].to(torch.bfloat16)
            hashes[str(rank)] = hashlib.sha256(
                memoryview(weight.view(torch.uint8).numpy())).hexdigest()
            torch.save(weight, ROOT / f"weight-{rank}.pt")
    for name, seed in (("A", 20261020), ("B", 20261021)):
        generator = torch.Generator().manual_seed(seed)
        value = torch.randn(32, HIDDEN, generator=generator).to(torch.bfloat16)
        torch.save(value, ROOT / f"hidden-{name}.pt")
    assert not torch.npu.is_initialized()
    report = {
        "scope": "CPU preparation and real Ascend processor import",
        "npu_initialized": torch.npu.is_initialized(),
        "checkpoint_file": str(source), "index_sha256": hashlib.sha256(
            index.read_bytes()).hexdigest(), "weight_bf16_sha256": hashes,
        "source_sha256": {spec.origin: hashlib.sha256(
            Path(spec.origin).read_bytes()).hexdigest()},
        "processor": AscendLogitsProcessor.__module__,
        "device_entry_imports": [function.__name__ for function in (
            destroy_distributed_environment, destroy_model_parallel,
            init_distributed_environment, initialize_model_parallel)],
        "weight_shape": [SHARD, HIDDEN], "logical_vocab": 2 * SHARD,
        "padding_size": 1, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
    }
    save(ROOT / "cpu-check.json", report)
    print(json.dumps(report), flush=True)


def summary(values):
    ordered = sorted(values)
    return {"p50": statistics.median(values),
            "p90": ordered[round((len(ordered) - 1) * .9)],
            "p99": ordered[round((len(ordered) - 1) * .99)],
            "std": statistics.pstdev(values)}


def run():
    global torch
    import torch
    import torch.distributed as dist
    import torch_npu
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment, destroy_model_parallel, get_tp_group,
        init_distributed_environment, initialize_model_parallel,
    )
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ops.vocab_parallel_embedding import (
        AscendLogitsProcessor, AscendParallelLMHead, VocabParallelMode,
    )
    from head_frozen import install_fp32_head

    rank = int(os.environ["RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 2 and rank in (0, 1)
    device = 14 + rank
    torch.set_num_threads(1)
    torch.npu.set_device(device)
    capacity = torch.npu.get_device_properties(device).total_memory
    torch.npu.set_per_process_memory_fraction(512 * 2**20 / capacity)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = 2
    init_ascend_config(config)
    report = {"rank": rank, "device": device, "world_size": 2,
              "scope": "TP2 head probe; not TP16 or decoder TPOT",
              "weight_scope": "checkpoint BF16-converted shards 0 and 1",
              "torch_quota_bytes": 512 * 2**20, "cases": [],
              "environment": {k: os.environ.get(k) for k in (
                  "HCCL_BUFFSIZE", "HCCL_OP_EXPANSION_MODE", "HCCL_ALGO",
                  "HCCL_DETERMINISTIC", "TASK_QUEUE_ENABLE",
                  "ASCEND_GLOBAL_LOG_LEVEL")}}
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2, rank=rank, local_rank=device, backend="hccl",
            timeout=timedelta(seconds=120))
        initialize_model_parallel(tensor_model_parallel_size=2, backend="hccl")
        control = get_tp_group().cpu_group
        import vllm.model_executor.layers.logits_processor as processor_module

        implementations = {
            "LogitsProcessor": processor_module.LogitsProcessor,
            "AscendLogitsProcessor": AscendLogitsProcessor,
            "get_top_tokens": AscendLogitsProcessor.get_top_tokens,
            "tp_gather": processor_module.tensor_model_parallel_all_gather,
            "group_all_gather": get_tp_group().all_gather,
            "device_all_gather": get_tp_group().device_communicator.all_gather,
        }
        report["loaded_implementations"] = {}
        for name, implementation in implementations.items():
            path = Path(inspect.getsourcefile(implementation))
            report["loaded_implementations"][name] = {
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report["tp_backend"] = dist.get_backend(get_tp_group().device_group)
        try:
            with torch.device(f"npu:{device}"), torch.inference_mode():
                processor = AscendLogitsProcessor(2 * SHARD)
                assert processor.use_all_gather and processor.head_dtype is None
                weight = torch.load(ROOT / f"weight-{rank}.pt",
                                    map_location="cpu", weights_only=True)
                heads = []
                for _ in range(2):
                    head = AscendParallelLMHead(
                        2 * SHARD, HIDDEN, params_dtype=torch.bfloat16,
                        padding_size=1, prefix="lm_head")
                    assert head.parallel_mode is VocabParallelMode.STANDARD
                    assert head.weight.shape == (SHARD, HIDDEN)
                    assert head.shard_indices.num_org_vocab_padding == 0
                    head.weight.copy_(weight)
                    install_fp32_head(head, processor)
                    heads.append(head)
                report["loaded_weight_sha256"] = [digest(h.weight) for h in heads]
                expected_hash = json.loads((ROOT / "cpu-check.json").read_text())[
                    "weight_bf16_sha256"][str(rank)]
                assert report["loaded_weight_sha256"] == [expected_hash] * 2
                report["weight_pointers"] = [h.weight.data_ptr() for h in heads]
                assert len(set(report["weight_pointers"])) == 2
                report["weight_banks"] = 2
                report["head_metadata"] = {
                    "parallel_mode": head.parallel_mode.name,
                    "tp_size": head.tp_size, "tp_rank": head.tp_rank,
                    "weight_shape": list(head.weight.shape),
                    "shard_indices": vars(head.shard_indices),
                    "scale": processor.scale, "soft_cap": processor.soft_cap,
                    "head_dtype": str(processor.head_dtype),
                    "use_all_gather": processor.use_all_gather}
                del weight
                payloads = {name: torch.load(
                    ROOT / f"hidden-{name}.pt", map_location="cpu",
                    weights_only=True) for name in ("A", "B")}
                stream = torch.npu.Stream(device=device)
                torch.npu.set_stream(stream)
                for width in (1, 8, 32):
                    hidden = torch.empty((width, HIDDEN), dtype=torch.bfloat16,
                                         device=f"npu:{device}")
                    hidden.copy_(payloads["A"][:width])

                    def call(arm, bank=0):
                        head = heads[bank]
                        if arm == "full":
                            return processor(head, hidden).argmax(dim=-1)
                        return processor.get_top_tokens(head, hidden)

                    correctness = []
                    for payload in ("A", "B", "A", "zero"):
                        if payload == "zero":
                            hidden.zero_()
                        else:
                            hidden.copy_(payloads[payload][:width])
                        full, reduced = call("full"), call("reduced")
                        torch.npu.synchronize()
                        assert torch.equal(full, reduced)
                        if payload == "zero":
                            assert bool((full == 0).all())
                        correctness.append({"payload": payload, "exact": True,
                                            "tokens": full.cpu().tolist()})
                    hidden.copy_(payloads["A"][:width])
                    for _ in range(10):
                        for arm in ARMS:
                            call(arm, _ % 2)
                    torch.npu.synchronize()
                    graphs, outputs = {}, {}
                    for arm in ARMS:
                        dist.barrier(group=control)
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph, stream=stream):
                            for index in range(10):
                                output = call(arm, index % 2)
                        graphs[arm], outputs[arm] = graph, output
                        torch.npu.synchronize()
                    for payload in ("A", "B", "A", "zero"):
                        if payload == "zero":
                            hidden.zero_()
                        else:
                            hidden.copy_(payloads[payload][:width])
                        for arm in ARMS:
                            graphs[arm].replay()
                        torch.npu.synchronize()
                        expected = call("full")
                        torch.npu.synchronize()
                        assert all(torch.equal(expected, outputs[a]) for a in ARMS)
                    save(ROOT / f"correctness-{rank}-M{width}.json", {
                        "rank": rank, "width": width,
                        "eager": correctness, "graph_A_B_reset_exact": True})
                    print(json.dumps({"rank": rank, "width": width,
                                      "correctness": "pass"}), flush=True)
                    hidden.copy_(payloads["A"][:width])

                    eager_next = {arm: 0 for arm in ARMS}

                    def measure(arm, mode):
                        torch.npu.synchronize()
                        dist.barrier(group=control)
                        begin = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        wall = time.perf_counter_ns()
                        begin.record(stream)
                        if mode == "graph":
                            graphs[arm].replay()
                        else:
                            call(arm, eager_next[arm] % 2)
                            eager_next[arm] += 1
                        submitted = time.perf_counter_ns()
                        end.record(stream)
                        torch.npu.synchronize()
                        stopped = time.perf_counter_ns()
                        calls = 10 if mode == "graph" else 1
                        return {"event_us": begin.elapsed_time(end) * 1000 / calls,
                                "wall_us": (stopped - wall) / 1000 / calls,
                                "host_submit_us": (submitted - wall) / 1000 / calls}

                    modes = {}
                    for mode, pairs in (("eager", 10), ("graph", 30)):
                        for _ in range(10):
                            for arm in ARMS:
                                measure(arm, mode)
                        raw = [{arm: [] for arm in ARMS} for _ in range(pairs)]
                        for pair in raw:
                            for arm in ("full", "reduced", "reduced", "full"):
                                pair[arm].append(measure(arm, mode))
                        modes[mode] = {"raw": raw, "pairs": pairs}
                    record = {"width": width, "correctness": correctness,
                              "graph_A_B_reset_exact": True, "modes": modes,
                              "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                              "peak_reserved_bytes": torch.npu.max_memory_reserved()}
                    report["cases"].append(record)
                    save(ROOT / f"rank-{rank}.json", report)
                    print(json.dumps({"rank": rank, "width": width,
                                      "status": "pass"}), flush=True)
                    for graph in graphs.values():
                        graph.reset()
                    del graphs, outputs, graph, output, hidden
                report["status"] = "pass"
                report["source_sha256"] = {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__), ROOT / "head_frozen.py")}
                save(ROOT / f"rank-{rank}.json", report)
        finally:
            torch.npu.synchronize()
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/workspace/models/GLM-5.2-w4a8c8"))
    args = parser.parse_args()
    if args.cpu_check:
        prepare(args.checkpoint)
    else:
        run()
