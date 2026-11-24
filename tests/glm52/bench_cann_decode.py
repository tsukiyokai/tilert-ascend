"""Measure M1 internal token delivery through the real TP16 session.

Run with torchrun --nproc-per-node=16. Each request injects the same frozen
8K prefix and emits the prefill token plus ``steps`` decode tokens. Prefix
readiness and reset are outside the first-to-last callback timing window.
This short measurement excludes HTTP/HIXL and is not a 1024-token result.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns
import traceback

from run_cann_model import digest, load_capture


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class ReadyPrefix:
    """Complete all sixteen prefix copies before the first callback."""

    def __init__(self, model, channel):
        self.model = model
        self.channel = channel

    def __getattr__(self, name):
        return getattr(self.model, name)

    def inject(self, prefixes):
        import torch
        import torch.distributed as distributed

        self.model.inject(prefixes)
        torch.npu.synchronize(self.model.device)
        distributed.barrier(group=self.channel.group)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=9344)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--cpu-affinity", action="store_true",
                        help="Bind this rank's threads; leave IRQs unchanged")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.requests < 1:
        parser.error("Require positive steps/requests and nonnegative warmup")
    if args.capacity < 8192 + args.steps:
        parser.error("Capacity must hold the prefix and consumed tokens")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0,1]")
    return args


def bind_cpu_threads(rank, device, report):
    from vllm.platforms import current_platform
    from vllm_ascend.cpu_binding import CpuAlloc
    import vllm_ascend.cpu_binding as cpu_binding

    physical = current_platform.device_id_to_physical_device_id(device.index)
    binder = CpuAlloc(rank, npu_id=physical)
    before = sorted(os.sched_getaffinity(0))
    if not binder.build_cpu_pools():
        raise RuntimeError("The CPU allocator did not build a binding plan")
    binder.allocate()
    binder.bind_threads()
    threads = []
    for task in sorted(Path("/proc/self/task").iterdir()):
        threads.append({
            "tid": int(task.name), "name": (task / "comm").read_text().strip(),
            "cpus": sorted(os.sched_getaffinity(int(task.name))),
        })
    report["cpu_affinity"] = {
        "physical_device": physical, "before": before,
        "main": binder.assign_main[physical],
        "acl": binder.assign_acl[physical],
        "release": binder.assign_rel[physical], "threads": threads,
        "method": "CpuAlloc.build_cpu_pools/allocate/bind_threads",
        "irq_modified": False, "numa_pages_migrated": False,
    }
    source = Path(cpu_binding.__file__)
    report["source_sha256"][str(source)] = digest(source)


def drive_requests(backend, first_token, args):
    from tilert.models.glm_5_2_ascend.session import PrefixRequest
    from tilert.pd_vllm.ascend_server import create_session

    session = create_session(backend, 1, ())
    records = []
    try:
        for number in range(args.warmup + args.requests):
            request = PrefixRequest(f"bench-{number}", 8192)
            tokens, timestamps = [], []

            def deliver(token):
                tokens.append(int(token))
                timestamps.append(perf_counter_ns())

            try:
                session.inject(request)
                output = session.decode(
                    first_token, max_tokens=args.steps + 1,
                    sampling={"temperature": 0.0, "ignore_eos": True},
                    on_token=deliver,
                )
                stats = dict(session.last_stats)
                if (output != tokens or len(tokens) != args.steps + 1
                        or tokens[0] != first_token
                        or stats["step_calls"] != args.steps
                        or stats["initial_position"] != 8192
                        or stats["committed_position"] != 8192 + args.steps
                        or stats["finish_reason"] != "length"):
                    raise RuntimeError("Session delivery/commit counts differ")
                elapsed = timestamps[-1] - timestamps[0]
                records.append({
                    "request_id": request.request_id,
                    "warmup": number < args.warmup,
                    "tokens": tokens, "timestamps_ns": timestamps,
                    "elapsed_ns": elapsed,
                    "tpot_ms": elapsed / args.steps / 1e6,
                    "pending_token": tokens[-1], **stats,
                })
            finally:
                if not backend.failed:
                    session.reset()
    finally:
        if not backend.failed:
            backend.close()
    return records


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu  # noqa: F401
    from vllm.distributed import get_world_group

    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.models.glm_5_2_ascend.tp_runtime import (
        CoordinatedBackend, RankExecutor, TorchControlChannel,
    )
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = report["rank"]
    if (int(os.environ["WORLD_SIZE"]) != 16
            or int(os.environ["LOCAL_RANK"]) != rank):
        raise ValueError("Launch one local sixteen-rank torchrun group")
    if any(os.environ.get(name) for name in (
        "PROFILER_SAMPLECONFIG", "PROFILING_MODE",
    )):
        raise ValueError("Run the latency measurement without profiling")
    torch.set_num_threads(1)
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    prefixes, teacher, inputs = load_capture(
        args.capture_case, rank, checkpoint,
    )
    if inputs["initial_position"] != 8192:
        raise ValueError("The benchmark requires a captured 8K prefix")
    device = torch.device(f"npu:{rank}")
    model = CannModel(
        args.checkpoint, device, rank, capacity=args.capacity,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    try:
        if model.tp_reduction != "float32":
            raise ValueError("Keep the CANN model's default FP32 TP reduction")
        channel = TorchControlChannel(get_world_group().cpu_group)
        engine = ReadyPrefix(model, channel)
        executor = RankExecutor(engine, rank, lambda request, _rank: prefixes)
        sources = [Path(__file__),
                   Path(__file__).with_name("run_cann_model.py"),
                   Path(__file__).with_name("prefix_io.py")]
        sources.extend(ROOT / "tilert" / name for name in (
            "models/glm_5_2_ascend/tp_runtime.py",
            "models/glm_5_2_ascend/session.py", "pd_vllm/ascend_server.py",
            "pd_vllm/ascend_cache.py",
        ))
        report.update(
            checkpoint_sha256=checkpoint, inputs=inputs,
            packages={name: version(name) for name in (
                "torch", "torch-npu", "vllm", "vllm-ascend")},
            source_sha256={**model.source_metadata(), **{
                str(path): digest(path) for path in sources}},
            device=str(device), device_name=torch.npu.get_device_name(device),
            model_tp_reduction=model.tp_reduction,
            runtime_env={name: os.environ.get(name) for name in (
                "HCCL_OP_EXPANSION_MODE", "HCCL_BUFFSIZE",
                "HCCL_DETERMINISTIC", "TASK_QUEUE_ENABLE",
                "ASCEND_LAUNCH_BLOCKING", "ASCEND_CUSTOM_OPP_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
            )},
        )
        if args.cpu_affinity:
            bind_cpu_threads(rank, device, report)
        with torch.inference_mode():
            if rank == 0:
                backend = CoordinatedBackend(executor, channel)
                report["requests"] = drive_requests(backend, teacher[0], args)
            else:
                executor.serve(channel)
        report.update(command_serial=executor.serial,
                      final_position=model.position)
        if not executor.closed or model.position != 0:
            raise RuntimeError("The request stream did not close after reset")
        report["status"] = "measured" if rank == 0 else "completed"
        path = args.output / f"rank-{rank}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        ranks = [None] * 16
        distributed.all_gather_object(ranks, {
            "rank": rank, "report_sha256": digest(path),
            "command_serial": executor.serial,
            "checkpoint_sha256": checkpoint,
            "prompt_sha256": inputs["prompt_sha256"],
            "capture_manifest_sha256": inputs["capture_manifest_sha256"],
            "source_sha256": report["source_sha256"],
            "packages": report["packages"], "first_token": teacher[0],
        }, group=channel.group)
        keys = ("command_serial", "checkpoint_sha256", "prompt_sha256",
                "capture_manifest_sha256", "source_sha256", "packages",
                "first_token")
        if any(any(item[key] != ranks[0][key] for key in keys)
               for item in ranks):
            raise RuntimeError("Rank execution identities differ")
        if rank == 0:
            measured = [item for item in report["requests"]
                        if not item["warmup"]]
            values = [item["tpot_ms"] for item in measured]
            summary = {
                "status": "measured", "scope": report["scope"],
                "precision_verified": False, "formal_1024_token_run": False,
                "steps": args.steps, "warmup": args.warmup,
                "requests": args.requests, "tpot_ms": {
                    "median": statistics.median(values),
                    "mean": statistics.mean(values),
                    "min": min(values), "max": max(values),
                }, "ranks": ranks,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n",
            )
    finally:
        model.shutdown()


def main():
    args = parse_arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    report = {
        "status": "error", "rank": rank, "world_size": 16,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "source_commit": args.source_commit, "capacity": args.capacity,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "scope": "M1 internal TP16 session delivery; HTTP/HIXL excluded",
        "precision_verified": False, "formal_1024_token_run": False,
        "profiled": False, "forward_hooks": False, "mtp_width": 1,
        "cpu_affinity_enabled": args.cpu_affinity,
        "steps": args.steps, "warmup": args.warmup,
        "measured_requests": args.requests, "ignore_eos": True,
        "clock": "time.perf_counter_ns at token callback completion",
        "tpot_formula": "(last_callback_ns-first_callback_ns)/steps/1e6",
        "prefix_readiness": (
            "Each rank synchronizes prefix copies, then one Gloo barrier; "
            "outside timing; this barrier is absent from the raw profile"
        ),
    }
    try:
        run(args, report)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        (args.output / f"rank-{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n",
        )
    print(json.dumps({"rank": rank, "status": report["status"],
                      "output": str(args.output)}))
    return int(report["status"] not in ("measured", "completed"))


if __name__ == "__main__":
    raise SystemExit(main())
