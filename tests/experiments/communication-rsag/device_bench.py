"""ABBA worlds for the complete two-reduction+CUBE probe; no profiler."""

import argparse
import ctypes
import datetime
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time

from device_check import (
    NativeProbe, bind_library, compare, consensus, digest, oracle, source, weight,
)


HERE = Path(__file__).resolve().parent
CACHE_VARIABLES = ("ASCEND_CACHE_PATH", "XDG_CACHE_HOME",
                   "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT")
ARMS = ("control", "candidate", "candidate", "control")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Quantile needs raw observations")
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)]
                             - ordered[lower]) * (index - lower)


def stats(values):
    return {"count": len(values), "p50": quantile(values, 0.5),
            "p95": quantile(values, 0.95), "min": min(values), "max": max(values)}


def registration(config, cycles):
    return {
        "schema": 1, "config": config, "abba_cycles": cycles,
        "worlds": [{"id": cycle * 4 + position, "cycle": cycle,
                    "position": position, "arm": arm}
                   for cycle in range(cycles) for position, arm in enumerate(ARMS)],
        "scope": "single-node TP complete probe; two reductions+CUBE each repetition",
        "semantic_boundary": "both arms use one fixed TreePlan; no claim of installed HCCL "
                             "equivalence, isolated cold-call latency or decoder TPOT",
        "measurement": {
            "unit": "one captured graph containing K complete probes, divided by K",
            "host": "(latest all-rank synchronized completion - earliest start)/K",
            "event": "maximum per-rank NPU event span/K",
            "clock": "same-host CLOCK_MONOTONIC; perf_counter and real-time anchors recorded",
            "start": "gloo agrees a future CLOCK_MONOTONIC deadline; actual submission envelope retained",
            "outside": "input change, CPU oracle, sentinel fill, pressure, NPU sync, gloo barrier",
            "cache": "256 MiB or larger full read/write; first repetition is pressure-conditioned",
            "warm_repetitions_fraction": "(K-1)/K; this is not isolated cold-call latency",
            "first_repetition_fraction": 1 / config["replays_per_sample"],
            "cache_eviction_proven": False,
            "outliers": "none removed", "profiler": "disabled",
        },
        "hypothesis": {
            "target_ranks": 16,
            "minimum_world_pairs": 30,
            "p50_saving_fraction_minimum": 0.10,
            "p95_candidate_over_control_maximum": 1.0,
            "maximum_submission_envelope_per_probe_over_control_p50": 0.01,
            "acceptance": "every rows, both host/event; global p50/p95 and paired world p50; "
                          "all numerical/rank-completion checks pass; >=30 world pairs",
            "pilot": "fewer worlds report raw statistics but cannot accept the hypothesis",
        },
        "runtime_sources": {name: digest(HERE / name)
                            for name in ("device_bench.py", "device_check.py")},
    }


def validate_config(config):
    if (not 2 <= config["ranks"] <= 16 or not config["rows"]
            or any(not 1 <= rows <= 32 for rows in config["rows"])
            or len(set(config["rows"])) != len(config["rows"])
            or config["warmup"] < 10 or config["samples"] < 30
            or config["pressure_mib"] < 256 or config["blocks"] < 1
            or config["replays_per_sample"] < 1 or config["start_lead_ms"] < 5):
        raise ValueError("Require ranks2..16, rows1..32, warmup>=10, samples>=30, pressure>=256MiB")
    epoch_bound = 2 * (2 + config["replays_per_sample"]
                       * (1 + config["warmup"] + config["samples"]))
    if epoch_bound > 0xFFFFFFFE:
        raise ValueError("Window epoch lifetime would exceed uint32 bound")


def timing_bounds(timings, repetitions):
    if (repetitions < 1 or sorted(item["rank"] for item in timings) != list(range(len(timings)))
            or any(not item["start_ns"] <= item["enqueue_before_ns"]
                   <= item["enqueue_after_ns"] <= item["end_ns"] for item in timings)
            or any(item["end_ns"] <= item["start_ns"] for item in timings)
            or any(not math.isfinite(item["event_span_us"]) or item["event_span_us"] <= 0
                   for item in timings)):
        raise ValueError("Invalid raw rank clock/order/event duration")
    starts = [item["start_ns"] for item in timings]
    before = [item["enqueue_before_ns"] for item in timings]
    after = [item["enqueue_after_ns"] for item in timings]
    ends = [item["end_ns"] for item in timings]
    return {
        "start_min_ns": min(starts), "start_max_ns": max(starts),
        "enqueue_before_min_ns": min(before), "enqueue_before_max_ns": max(before),
        "enqueue_after_min_ns": min(after), "enqueue_after_max_ns": max(after),
        "completion_min_ns": min(ends), "completion_max_ns": max(ends),
        "launch_skew_us": (max(before) - min(before)) / 1000,
        "submission_envelope_us": (max(after) - min(starts)) / 1000,
        "world_host_us_per_probe": (max(ends) - min(starts)) / 1000 / repetitions,
        "world_event_us_per_probe": max(item["event_span_us"] for item in timings) / repetitions,
        "all_ranks_finished": len(timings),
    }


def now_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def rendezvous_deadline(dist, rank, lead_ms):
    deadline = [now_ns() + int(lead_ms * 1000000) if rank == 0 else None]
    dist.broadcast_object_list(deadline, src=0)
    received_ns = now_ns()
    while now_ns() < deadline[0]:
        pass
    return deadline[0], received_ns


def run_shape(torch, torch_npu, dist, native, config, rows, rank, ranks, raw):
    owner = ctypes.c_void_p()
    native.call("prepare", rows, rank, ranks, config["blocks"], ctypes.byref(owner))
    torch.npu.synchronize()
    dist.barrier()
    partial = torch_npu.npu_format_cast(
        torch.empty((rows, 6144), dtype=torch.bfloat16, device="npu"), 2)
    local_weight = torch_npu.npu_format_cast(weight(torch, rank).npu(), 2)
    guards = [torch.full((rows * 6144 + 512,), -37, dtype=torch.bfloat16,
                         device="npu") for _ in range(2)]
    first, output = [value[256:-256].reshape(rows, 6144) for value in guards]
    pressure = torch.zeros(config["pressure_mib"] * 1024 * 1024,
                           dtype=torch.uint8, device="npu")
    pointers = [value.data_ptr() for value in (partial, local_weight, first, output, pressure)]
    if (len(set(pointers)) != len(pointers) or any(pointer % 512 for pointer in pointers)
            or any(torch_npu.get_npu_format(value) != 2
                   for value in (partial, local_weight, first, output))):
        raise ValueError("Bench inputs/output/pressure need distinct 512B-aligned ND storage")
    graph = None
    records = []
    repetitions = config["replays_per_sample"]

    def execute():
        native.call("execute", owner, torch.npu.current_stream().npu_stream,
                    partial.data_ptr(), local_weight.data_ptr(),
                    first.data_ptr(), output.data_ptr())

    try:
        partial.copy_(source(torch, rows, rank, 0))
        torch.npu.synchronize()
        dist.barrier()
        for _ in range(2):
            execute()
        torch.npu.synchronize()
        dist.barrier()
        stream = torch.npu.Stream()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            for _ in range(repetitions):
                execute()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            for _ in range(config["warmup"]):
                graph.replay()
        torch.npu.synchronize()
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        with torch.npu.stream(stream):
            begin.record()
            end.record()
        end.synchronize()  # Materialize lazy event handles outside the interval.
        dist.barrier()
        for sample in range(config["samples"]):
            fixture = source(torch, rows, rank, sample + 2)
            expected_first, expected_output = oracle(torch, rows, ranks, sample + 2)
            partial.copy_(fixture)
            for value in guards:
                value.fill_(-37)
            pressure.add_(1)
            torch.npu.synchronize()
            expected_pressure = (sample + 1) % 256
            if (int(pressure[0].item()) != expected_pressure
                    or int(pressure[-1].item()) != expected_pressure):
                raise ValueError("Read/write pressure endpoint check failed")
            torch.npu.synchronize()
            dist.barrier()  # All coordination is outside both clocks.
            with torch.npu.stream(stream):
                deadline_ns, deadline_received_ns = rendezvous_deadline(
                    dist, rank, config["start_lead_ms"])
                start_ns = now_ns()
                begin.record()
                enqueue_before_ns = now_ns()
                graph.replay()
                enqueue_after_ns = now_ns()
                end.record()
                end.synchronize()
                end_ns = now_ns()
            timing = {"rank": rank, "start_ns": start_ns, "end_ns": end_ns,
                      "deadline_ns": deadline_ns, "deadline_received_ns": deadline_received_ns,
                      "enqueue_before_ns": enqueue_before_ns,
                      "enqueue_after_ns": enqueue_after_ns,
                      "event_span_us": begin.elapsed_time(end) * 1000,
                      "host_span_us": (end_ns - start_ns) / 1000}
            all_timings = [None] * ranks
            dist.all_gather_object(all_timings, timing)
            checks = {"first": compare(torch, first, expected_first),
                      "output": compare(torch, output, expected_output),
                      "input": compare(torch, partial, fixture),
                      "weight": compare(torch, local_weight, weight(torch, rank))}
            intact = all(bool((value[:256] == -37).all())
                         and bool((value[-256:] == -37).all()) for value in guards)
            passed = intact and all(item["bitwise_equal"] for item in checks.values())
            all_checks = [None] * ranks
            dist.all_gather_object(all_checks, passed)
            record = {"rows": rows, "sample": sample, "fixture_iteration": sample + 2,
                      "repetitions": repetitions, "timing": timing,
                      "all_rank_timing": all_timings,
                      "bounds": timing_bounds(all_timings, repetitions),
                      "pressure_mib": config["pressure_mib"],
                      "first_repetition_fraction": 1 / repetitions,
                      "pressure_value": expected_pressure, "checks": checks,
                      "guard_regions_intact": intact, "all_rank_checks": all_checks,
                      "pass": all(all_checks)}
            records.append(record)
            raw.write(json.dumps(record) + "\n")
            raw.flush()
            if not record["pass"]:
                raise RuntimeError("Complete-probe numerical gate failed")
    finally:
        if graph is not None:
            graph.reset()
        torch.npu.synchronize()
        dist.barrier()
        native.call("destroy", owner)
    return records


def worker(args):
    plan = json.loads(args.registration.read_text())
    config = plan["config"]
    validate_config(config)
    require_profiler_off()
    for name, expected in plan["runtime_sources"].items():
        if digest(HERE / name) != expected:
            raise ValueError(f"Registered runtime source changed: {name}")
    if not 0 <= args.world_id < len(plan["worlds"]):
        raise ValueError("World identifier is outside the registered ABBA schedule")
    world = plan["worlds"][args.world_id]
    rank, ranks = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if ranks != config["ranks"] or int(os.environ["LOCAL_WORLD_SIZE"]) != ranks:
        raise ValueError("This benchmark requires one single-node world")
    args.world_dir.mkdir(parents=True, exist_ok=True)
    caches = {}
    for name in CACHE_VARIABLES:
        directory = args.world_dir / "cache" / f"rank-{rank}" / name.lower()
        directory.mkdir(parents=True, exist_ok=False)
        os.environ[name] = str(directory)
        caches[name] = str(directory)
    library, binding = bind_library(args.prepared, world["arm"], args.frozen_driver)
    if binding != plan["libraries"][world["arm"]]:
        raise ValueError("Registered arm/source/ELF binding changed")
    import torch
    import torch.distributed as dist
    import torch_npu

    torch.set_num_threads(1)
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=300))
    consensus(dist, ranks, {"host": socket.gethostname(), "world": world,
                            "endpoint": args.endpoint, "config": config,
                            "library": binding["library_sha256"]})
    native = NativeProbe(library, world["arm"])
    native.call("initialize", rank, ranks, args.endpoint.encode(), 64 * 1024 * 1024)
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        name = line.split()[-1]
        if "shmem" in name and name.startswith("/") and Path(name).is_file():
            mapped.add(name)
    clock = time.get_clock_info("perf_counter")
    report = {"world": world, "rank": rank, "ranks": ranks, "binding": binding,
              "registration_sha256": digest(args.registration),
              "runtime_sources": plan["runtime_sources"], "config": config,
              "host": socket.gethostname(), "pid": os.getpid(),
              "endpoint": args.endpoint,
              "clock": {"implementation": clock.implementation,
                        "interval_clock": "clock_gettime_ns(CLOCK_MONOTONIC)",
                        "monotonic": clock.monotonic, "adjustable": clock.adjustable,
                        "resolution_seconds": clock.resolution,
                        "perf_counter_ns": time.perf_counter_ns(), "time_ns": time.time_ns(),
                        "monotonic_ns": now_ns()},
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "cache_paths": caches, "torch": torch.__version__,
              "torch_npu": torch_npu.__version__,
              "device": str(torch.npu.get_device_properties(torch.npu.current_device())),
              "loaded_shmem": [{"path": path, "sha256": digest(Path(path))}
                               for path in sorted(mapped)],
              "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "environment": {name: os.environ.get(name) for name in (
                  "OMP_NUM_THREADS", "MKL_NUM_THREADS", "ASCEND_LAUNCH_BLOCKING",
                  "HCCL_ALGO", "HCCL_ALGO_CONFIG", "HCCL_DETERMINISTIC")},
              "samples": [], "all_ranks_completed": False}
    try:
        with (args.world_dir / f"rank-{rank}-raw.jsonl").open("x") as raw:
            for rows in config["rows"]:
                report["samples"].extend(run_shape(
                    torch, torch_npu, dist, native, config, rows, rank, ranks, raw))
        dist.barrier()
        report["all_ranks_completed"] = True
        report["pass"] = all(record["pass"] for record in report["samples"])
    finally:
        torch.npu.synchronize()
        dist.barrier()
        native.call("finalize")
        dist.destroy_process_group()
        write_json(args.world_dir / f"rank-{rank}.json", report)


def summarize(directory):
    plan = json.loads((directory / "registration.json").read_text())
    config = plan["config"]
    validate_config(config)
    if plan["worlds"] != registration(config, plan["abba_cycles"])["worlds"]:
        raise ValueError("Registered world order is not ABBA")
    worlds = []
    for world in plan["worlds"]:
        location = directory / f"world-{world['id']:03d}-{world['arm']}"
        invocation = json.loads((location / "invocation.json").read_text())
        if invocation["returncode"] != 0:
            raise ValueError("A failed world cannot enter performance statistics")
        for phase in ("before", "after"):
            snapshot = location / f"occupancy-{phase}.txt"
            evidence = invocation[f"occupancy_{phase}"]
            if (not evidence["idle"] or digest(snapshot) != evidence["sha256"]
                    or idle_process_table(snapshot.read_text()) != evidence["cards"]):
                raise ValueError("World occupancy guard is missing, altered or non-idle")
        reports = [json.loads((location / f"rank-{rank}.json").read_text())
                   for rank in range(config["ranks"])]
        for rank, report in enumerate(reports):
            if (report["rank"] != rank or report["world"] != world
                    or report["config"] != config
                    or report["registration_sha256"] != digest(directory / "registration.json")
                    or report["binding"] != plan["libraries"][world["arm"]]
                    or report["host"] != plan["launch"]["hostname"]
                    or not report["all_ranks_completed"] or not report["pass"]
                    or len(report["samples"]) != len(config["rows"]) * config["samples"]):
                raise ValueError(f"Incomplete or inconsistent rank evidence in {location}")
        world_result = {"world": world, "rows": {}}
        for rows in config["rows"]:
            records = [item for item in reports[0]["samples"] if item["rows"] == rows]
            if [item["sample"] for item in records] != list(range(config["samples"])):
                raise ValueError("Raw sample identifiers are missing or repeated")
            for record in records:
                index = reports[0]["samples"].index(record)
                for rank, report in enumerate(reports):
                    own = report["samples"][index]
                    if (own["bounds"] != record["bounds"]
                            or own["all_rank_timing"] != record["all_rank_timing"]
                            or own["timing"] != record["all_rank_timing"][rank]
                            or own["sample"] != record["sample"] or own["rows"] != rows
                            or own["repetitions"] != config["replays_per_sample"]
                            or not own["guard_regions_intact"]
                            or len(own["all_rank_checks"]) != config["ranks"]
                            or not all(own["all_rank_checks"])
                            or not all(own["checks"][name]["bitwise_equal"]
                                       for name in ("first", "output", "input", "weight"))
                            or not own["pass"]):
                        raise ValueError("Per-rank raw clocks disagree with collective evidence")
                recomputed = timing_bounds(record["all_rank_timing"], config["replays_per_sample"])
                if recomputed != record["bounds"] or recomputed["all_ranks_finished"] != config["ranks"]:
                    raise ValueError("Raw rank completion bounds do not reproduce")
            world_result["rows"][str(rows)] = {
                metric: {"raw": [item["bounds"][f"world_{metric}_us_per_probe"] for item in records],
                         **stats([item["bounds"][f"world_{metric}_us_per_probe"] for item in records])}
                for metric in ("host", "event")}
            world_result["rows"][str(rows)]["submission_envelope_us_per_probe"] = [
                item["bounds"]["submission_envelope_us"] / config["replays_per_sample"]
                for item in records]
        worlds.append(world_result)
    result = {"registration": plan, "worlds": worlds, "rows": {},
              "target_rank_count_met": config["ranks"] == plan["hypothesis"]["target_ranks"],
              "co_tenant_evidence": "every world has hash-verified idle process tables before/after; "
                                    "this cannot observe a transient co-tenant between snapshots",
              "world_pairs": plan["abba_cycles"] * 2}
    enough_pairs = result["world_pairs"] >= plan["hypothesis"]["minimum_world_pairs"]
    for rows in config["rows"]:
        key = str(rows)
        metrics = {}
        for metric in ("host", "event"):
            grouped = {arm: [value for world in worlds if world["world"]["arm"] == arm
                             for value in world["rows"][key][metric]["raw"]]
                       for arm in ("control", "candidate")}
            aggregate = {arm: stats(values) for arm, values in grouped.items()}
            pairs = []
            for cycle in range(plan["abba_cycles"]):
                for control_position, candidate_position in ((0, 1), (3, 2)):
                    control = worlds[cycle * 4 + control_position]["rows"][key][metric]
                    candidate = worlds[cycle * 4 + candidate_position]["rows"][key][metric]
                    pairs.append({"control_world": cycle * 4 + control_position,
                                  "candidate_world": cycle * 4 + candidate_position,
                                  "p50_ratio": candidate["p50"] / control["p50"],
                                  "p50_saving_fraction": 1 - candidate["p50"] / control["p50"],
                                  "p95_saving_fraction": 1 - candidate["p95"] / control["p95"]})
            saving = 1 - aggregate["candidate"]["p50"] / aggregate["control"]["p50"]
            p95_ratio = aggregate["candidate"]["p95"] / aggregate["control"]["p95"]
            paired = statistics.median(pair["p50_saving_fraction"] for pair in pairs)
            paired_ratio = statistics.median(pair["p50_ratio"] for pair in pairs)
            metrics[metric] = {"aggregate": aggregate, "pairs": pairs,
                               "p50_saving_fraction": saving, "p95_ratio": p95_ratio,
                               "paired_world_p50_saving_median": paired,
                               "hypothesis_pass": aggregate["candidate"]["p50"]
                               <= 0.90 * aggregate["control"]["p50"]
                               and paired_ratio <= 0.90 and p95_ratio <= 1.0}
        envelope = max(value for world in worlds
                       for value in world["rows"][key]["submission_envelope_us_per_probe"])
        denominator = metrics["event"]["aggregate"]["control"]["p50"]
        skew_valid = envelope / denominator <= 0.01
        result["rows"][key] = {"metrics": metrics,
                               "maximum_submission_envelope_us_per_probe": envelope,
                               "envelope_over_control_event_p50": envelope / denominator,
                               "launch_skew_valid": skew_valid,
                               "accepted": enough_pairs and skew_valid and result["target_rank_count_met"]
                               and all(value["hypothesis_pass"] for value in metrics.values())}
    result["pilot_only"] = not enough_pairs
    result["accepted"] = all(value["accepted"] for value in result["rows"].values())
    write_json(directory / "summary.json", result)
    return result


def idle_process_table(output):
    header = re.search(r"\|\s*NPU\s+Chip\s*\|\s*Process id\s*\|", output,
                       flags=re.IGNORECASE)
    cards = {int(value) for value in re.findall(r"^\|\s*(\d+)\s+Ascend\w*\s*\|", output,
                                               flags=re.MULTILINE)}
    if header is None or not cards:
        raise ValueError("Cannot identify the full npu-smi inventory/process table")
    idle = set()
    for line in output[header.end():].splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith("+"):
            continue
        empty = re.fullmatch(r"\|\s*No running processes found in NPU (\d+)\s*\|", line)
        if empty:
            idle.add(int(empty.group(1)))
        elif re.match(r"\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|", line):
            raise ValueError(f"External or leftover NPU process found: {line}")
        else:
            raise ValueError(f"Unrecognized NPU process row: {line}")
    if idle != cards:
        raise ValueError(f"Missing idle process evidence: cards={cards}, idle={idle}")
    return sorted(idle)


def occupancy(path):
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True,
                                text=True, timeout=20, check=False)
        path.write_text(result.stdout + result.stderr)
        if result.returncode != 0 or result.stderr.strip():
            raise ValueError(f"npu-smi failed: rc={result.returncode}, stderr={result.stderr!r}")
        cards = idle_process_table(result.stdout)
        return {"path": str(path), "sha256": digest(path), "idle": True, "cards": cards}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as error:
        if not path.exists():
            path.write_text(str(error) + "\n")
        return {"path": str(path), "sha256": digest(path), "idle": False, "error": str(error)}


def require_profiler_off():
    if any(os.environ.get(name, "").lower() in ("true", "on", "1")
           for name in ("PROFILING_MODE", "ASCEND_PROFILING_MODE")):
        raise ValueError("Benchmark requires profiler environment disabled")


def launch(args):
    config = {name: getattr(args, name) for name in (
        "ranks", "rows", "blocks", "warmup", "samples", "pressure_mib",
        "replays_per_sample", "start_lead_ms")}
    validate_config(config)
    if args.abba_cycles < 1 or not 1024 <= args.base_port <= 65535 - 8 * args.abba_cycles:
        raise ValueError("Invalid ABBA cycles or port range")
    require_profiler_off()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = registration(config, args.abba_cycles)
    prepared = args.prepared.resolve()
    frozen_driver = None if args.frozen_driver is None else args.frozen_driver.resolve()
    plan["libraries"] = {arm: bind_library(prepared, arm, frozen_driver)[1]
                         for arm in ("control", "candidate")}
    for name in ("manifest.json", "build-report.json"):
        (args.output / ("compiled-" + name)).write_bytes((prepared / name).read_bytes())
    plan["launch"] = {"hostname": socket.gethostname(), "python": sys.executable,
                      "prepared": str(prepared), "endpoint_host": args.endpoint_host,
                      "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                      "source_identity": "registered actual source SHA256 and frozen compiled manifest"}
    registration_path = args.output.resolve() / "registration.json"
    write_json(registration_path, plan)  # Commit the hypothesis before any worker starts.
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for world in plan["worlds"]:
        directory = args.output.resolve() / f"world-{world['id']:03d}-{world['arm']}"
        directory.mkdir()
        port = args.base_port + world["id"] * 2
        command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
                   f"--nproc-per-node={args.ranks}", "--master-addr=127.0.0.1",
                   f"--master-port={port}", str(HERE / "device_bench.py"), "worker",
                   "--prepared", str(prepared), "--registration", str(registration_path),
                   "--world-id", str(world["id"]), "--world-dir", str(directory),
                   "--endpoint", f"tcp://{args.endpoint_host}:{port + 1}"]
        if frozen_driver is not None:
            command.extend(("--frozen-driver", str(frozen_driver)))
        invocation = {"command": command, "start_monotonic_ns": now_ns(),
                      "occupancy_before": occupancy(directory / "occupancy-before.txt")}
        write_json(directory / "invocation.json", invocation)
        if not invocation["occupancy_before"]["idle"]:
            raise RuntimeError(f"World {world['id']} aborted before launch: NPU occupancy guard failed")
        with (directory / "torchrun.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.world_timeout, check=False)
        invocation.update({"returncode": result.returncode, "end_monotonic_ns": now_ns(),
                           "occupancy_after": occupancy(directory / "occupancy-after.txt")})
        write_json(directory / "invocation.json", invocation)
        if result.returncode or not invocation["occupancy_after"]["idle"]:
            raise RuntimeError(f"World {world['id']} failed; all raw/log files retained in {directory}")
    result = summarize(args.output)
    print(json.dumps({"accepted": result["accepted"], "pilot_only": result["pilot_only"],
                      "world_pairs": result["world_pairs"], "output": str(args.output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("worker")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--registration", type=Path, required=True)
    run.add_argument("--world-id", type=int, required=True)
    run.add_argument("--world-dir", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--frozen-driver", type=Path)
    report = commands.add_parser("summarize")
    report.add_argument("--output", type=Path, required=True)
    start = commands.add_parser("launch")
    start.add_argument("--prepared", type=Path, required=True)
    start.add_argument("--frozen-driver", type=Path)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--endpoint-host", required=True)
    start.add_argument("--ranks", type=int, default=16)
    start.add_argument("--rows", type=int, nargs="+", default=[1, 12, 32])
    start.add_argument("--blocks", type=int, default=24)
    start.add_argument("--abba-cycles", type=int, default=15)
    start.add_argument("--samples", type=int, default=30)
    start.add_argument("--warmup", type=int, default=10)
    start.add_argument("--pressure-mib", type=int, default=256)
    start.add_argument("--replays-per-sample", type=int, default=32)
    start.add_argument("--start-lead-ms", type=float, default=20)
    start.add_argument("--base-port", type=int, default=29800)
    start.add_argument("--world-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args)
    elif args.command == "launch":
        launch(args)
    else:
        result = summarize(args.output)
        print(json.dumps({"accepted": result["accepted"], "pilot_only": result["pilot_only"],
                          "world_pairs": result["world_pairs"], "rows": result["rows"]}, indent=2))


if __name__ == "__main__":
    main()
