"""Compare real M1 eager/ACL Graph forwards and replay after reset on TP16."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import traceback

from bench_cann_decode import ReadyPrefix, bind_cpu_threads, drive_requests
from run_cann_model import check_prefix, digest, load_capture


def snapshot(engine, step, position):
    return {
        "token_ids": step.token_ids.detach().cpu().clone(),
        "logits": step.logits.detach().cpu().clone(),
        "last_hidden": step.last_hidden.detach().cpu().clone(),
        **{f"cache.{layer}.{name}": value[position].detach().cpu().clone()
           for layer, fields in enumerate(engine.cache)
           for name, value in fields.items()},
    }


def compare(actual, expected, stats, *, exact=False):
    import torch

    if actual.keys() != expected.keys():
        raise ValueError("Graph snapshot fields changed")
    for name, value in actual.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Graph tensor contract changed: {name}")
        equal = torch.equal(value, reference)
        if exact and not equal:
            raise ValueError(f"Graph reset differs: {name}")
        if name == "token_ids" and not equal:
            raise ValueError(
                f"Graph successor {value.tolist()} != {reference.tolist()}"
            )
        if not value.is_floating_point() and not equal:
            raise ValueError(f"Graph integer cache differs: {name}")
        stats["tensors"] += 1
        stats["bit_exact"] += int(equal)
        if value.is_floating_point():
            if not torch.isfinite(value).all():
                raise ValueError(f"Graph output is nonfinite: {name}")
            diff = value.double() - reference.double()
            error = float(diff.norm() / reference.double().norm().clamp_min(
                1e-12,
            ))
            if error > stats["max_nrmse"]:
                stats.update(max_nrmse=error, max_nrmse_field=name)


def run(args, report):
    import torch
    import torch.distributed as distributed
    import torch_npu  # noqa: F401
    from vllm.distributed import get_world_group
    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

    rank = report["rank"]
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise ValueError("Use one local sixteen-rank group")
    torch.set_num_threads(1)
    checkpoint = checkpoint_fingerprint(args.checkpoint)
    prefixes, teacher, inputs = load_capture(
        args.capture_case, rank, checkpoint,
    )
    if inputs["initial_position"] != 8192:
        raise ValueError("Use the captured full-P 8K prefix")
    engine = CannModel(args.checkpoint, f"npu:{rank}", rank,
                       capacity=9344, gpu_memory_utilization=0.85)
    directory = args.output / f"rank-{rank}"
    directory.mkdir(parents=True, exist_ok=False)
    report.update(checkpoint_sha256=checkpoint, inputs=inputs,
                  source_sha256=engine.source_metadata(),
                  packages={name: version(name) for name in (
                      "torch", "torch-npu", "vllm", "vllm-ascend",
                  )})
    if args.cpu_affinity:
        bind_cpu_threads(rank, engine.device, report)
    tokens, artifacts, comparisons, checks = [], {}, {}, []
    try:
        with torch.inference_mode():
            for phase in ("eager", "graph", "reset"):
                distributed.barrier(group=get_world_group().cpu_group)
                engine.reset()
                engine.inject(prefixes)
                for fields in engine.cache:
                    for value in fields.values():
                        tail = value[8192:8192 + args.steps]
                        tail.fill_(float("nan")
                                   if value.is_floating_point() else -128)
                if phase == "graph":
                    engine.enable_graph()
                token = teacher[0]
                stats = dict(tensors=0, bit_exact=0, max_nrmse=0.0,
                             max_nrmse_field=None)
                for number in range(args.steps):
                    if phase != "eager":
                        token = tokens[number]
                    if (engine.graph is not None
                            and engine.graph.capture_count):
                        engine.graph.validate_metadata(engine._metadata())
                    position = engine.position
                    report.update(phase=phase, position=position)
                    step = engine.verify((token,))
                    actual = snapshot(engine, step, position)
                    path = directory / f"{phase}-{position}.pt"
                    torch.save(actual, path)
                    artifacts[str(path)] = digest(path)
                    if phase == "eager":
                        tokens.append(token)
                    else:
                        reference = directory / f"eager-{position}.pt"
                        expected = torch.load(reference, weights_only=True)
                        compare(actual, expected, stats)
                        if phase == "reset":
                            first = directory / f"graph-{position}.pt"
                            compare(actual, torch.load(first,
                                                       weights_only=True),
                                    stats, exact=True)
                    token = int(actual["token_ids"].item())
                comparisons[phase] = stats
                check_prefix(engine, prefixes, checks, phase)
            if not all(item["passed"] for item in checks):
                raise ValueError("Graph changed a frozen prefix row")
            graph = engine.graph
            if (graph.capture_count != 1
                    or graph.replay_count != 2 * args.steps):
                raise ValueError("Expected one capture and subsequent replays")
            entries = graph.wrapper.concrete_aclgraph_entries
            if len(entries) != 1 or any(
                entry.aclgraph is None for entry in entries.values()
            ):
                raise ValueError("Expected one actual captured M1 graph")
            entry = entries[graph.descriptor]
            if entry.input_addresses != [
                engine.input_ids.data_ptr(), engine.positions.data_ptr(),
            ]:
                raise ValueError("Graph input addresses differ from M1")
            report.update(
                status="pass", comparisons=comparisons, teacher_tokens=tokens,
                prefix_checks=len(checks), captures=graph.capture_count,
                replays=graph.replay_count, trace_sha256=artifacts,
                source_sha256={**report["source_sha256"],
                               **engine.source_metadata(),
                               str(Path(__file__)): digest(Path(__file__))},
            )
            accepted = [None] * 16
            distributed.all_gather_object(
                accepted, {"rank": rank, "comparisons": comparisons},
                group=get_world_group().cpu_group,
            )
            report["all_rank_comparisons"] = accepted
            if args.requests:
                from tilert.models.glm_5_2_ascend.tp_runtime import (
                    CoordinatedBackend, RankExecutor, TorchControlChannel,
                )

                engine.reset()
                channel = TorchControlChannel(get_world_group().cpu_group)
                executor = RankExecutor(ReadyPrefix(engine, channel), rank,
                                        lambda request, _rank: prefixes)
                if rank == 0:
                    backend = CoordinatedBackend(executor, channel)
                    report["requests"] = drive_requests(backend, teacher[0],
                                                        args)
                else:
                    executor.serve(channel)
                report.update(tpot_measured=True,
                              command_serial=executor.serial,
                              final_position=engine.position)
    finally:
        if engine.graph is not None:
            entries = engine.graph.wrapper.concrete_aclgraph_entries
            report["graph_entries"] = [{
                "descriptor": str(key),
                "aclgraph_type": type(entry.aclgraph).__name__,
                "has_aclgraph": entry.aclgraph is not None,
                "input_addresses": entry.input_addresses,
            } for key, entry in entries.items()]
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "capture-case", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cpu-affinity", action="store_true")
    parser.add_argument("--requests", type=int, default=0,
                        help="Profiler-off short requests after comparison")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1024:
        parser.error("steps must fit the 8K prefix and 9344-row cache")
    if args.requests < 0 or args.warmup < 0:
        parser.error("requests and warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    report = dict(status="error", rank=rank, scope="M1 eager/graph diagnosis",
                  precision_verified=False, tpot_measured=False,
                  steps=args.steps, command=list(os.sys.argv))
    try:
        run(args, report)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
    (args.output / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2) + "\n",
    )
    print(json.dumps({key: report[key] for key in ("rank", "status")}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
