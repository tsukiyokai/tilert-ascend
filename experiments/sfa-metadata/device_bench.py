"""Pair isolated A3 processes running synthetic native Prolog plus SFA."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter_ns
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/glm52/whole_layer"))
PREFIX = "TRT_SFA_RESULT "
CACHE_VARIABLES = (
    "ASCEND_CACHE_PATH", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT",
)


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tensor_identity(torch, value):
    value = value.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(
                value.view(torch.uint8).numpy().tobytes()).hexdigest()}


def signature(value):
    encoded = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def stats(values):
    ordered = sorted(values)
    return {"p50": statistics.median(ordered),
            "p90": ordered[round(.90 * (len(ordered) - 1))],
            "p95": ordered[round(.95 * (len(ordered) - 1))],
            "p99": ordered[round(.99 * (len(ordered) - 1))],
            "std": statistics.pstdev(ordered)}


def respond(value):
    print(PREFIX + json.dumps(value), flush=True)


def receive(process):
    for line in process.stdout:
        if line.startswith(PREFIX):
            response = json.loads(line[len(PREFIX):])
            if "error" in response:
                raise RuntimeError(response["error"])
            return response
    raise RuntimeError(f"SFA worker exited with {process.poll()}")


def exchange(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    return receive(process)


def native_boundaries(torch, case, width):
    query_bytes = width * 4 * 512 * 2
    rope_offset = (query_bytes + 511) // 512 * 512
    rope_bytes = width * 4 * 64 * 2
    values = case.scratch[:rope_offset + rope_bytes].cpu()
    return {
        "query": values[:query_bytes].view(torch.bfloat16).reshape(
            width, 4, 512),
        "query_rope": values[rope_offset:].view(torch.bfloat16).reshape(
            width, 4, 64),
        "kv_cache": case.cache_native.cpu(),
        "rope_cache": case.rope_native.cpu(),
        "attention": case.output.cpu(),
    }


def prepare_case(torch, torch_npu, reference, args, weights, width):
    gate_args = SimpleNamespace(
        device=args.device, sequence=args.sequence, replays=4, capture=True,
        no_timing=True, atol=0.0, rtol=0.0,
    )
    gate = reference.run_case(
        torch, torch_npu, gate_args, weights, width, args.blocks,
    )
    if not gate["pass"] or not all(
        boundary["bitwise_equal"]
        for replay in gate["checks"]
        for boundary in replay.values() if isinstance(boundary, dict)
    ):
        raise ValueError(f"Actual run_case precision gate failed: {gate}")
    fixture_args = SimpleNamespace(
        device=args.device, sequence=args.sequence,
        replays=args.warmup + args.samples,
    )
    case = reference.make_case_fixture(
        torch, torch_npu, fixture_args, weights, width, args.blocks,
    )
    native_graph, _ = reference.capture(torch, case.native, case.stream)
    vendor_graph, vendor_outputs = reference.capture(
        torch, case.vendor, case.stream,
    )
    case.stream.synchronize()
    return SimpleNamespace(
        case=case, gate=gate, native_graph=native_graph,
        vendor_graph=vendor_graph, vendor_outputs=vendor_outputs,
        begin=torch.npu.Event(enable_timing=True),
        end=torch.npu.Event(enable_timing=True), width=width,
        cache_base=tensor_identity(torch, case.cache_base),
        rope_base=tensor_identity(torch, case.rope_base),
    )


def measure(torch, reference, state, generation, pressure, pressure_value):
    case, stream = state.case, state.case.stream
    with torch.npu.stream(stream):
        case.update(generation)
    stream.synchronize()
    # Read the actual restored buffers; these transfers are outside timing.
    inputs = {name: tensor_identity(torch, getattr(case, name)) for name in (
        "x", "x_scale", "sin", "cos", "slots", "topk", "query_lengths",
        "kv_lengths", "page_table", "cache_native", "rope_native",
    )}
    if (inputs["cache_native"] != state.cache_base or
            inputs["rope_native"] != state.rope_base):
        raise ValueError("Native KV was not restored to the frozen start")
    with torch.npu.stream(stream):
        pressure.add_(1)
    stream.synchronize()
    if (int(pressure[0].item()) != pressure_value or
            int(pressure[-1].item()) != pressure_value):
        raise ValueError("Pressure buffer did not change as expected")
    stream.synchronize()
    with torch.npu.stream(stream):
        wall_start = perf_counter_ns()
        state.begin.record(stream)
        state.native_graph.replay()
        state.end.record(stream)
        state.end.synchronize()
        wall_us = (perf_counter_ns() - wall_start) / 1000
    event_us = state.begin.elapsed_time(state.end) * 1000
    actual = native_boundaries(torch, case, state.width)
    with torch.npu.stream(stream):
        state.vendor_graph.replay()
    stream.synchronize()
    expected = {
        "query": state.vendor_outputs[0],
        "query_rope": state.vendor_outputs[1],
        "kv_cache": case.cache_vendor, "rope_cache": case.rope_vendor,
        "attention": state.vendor_outputs[2],
    }
    checks = {
        name: reference.metric(torch, value, expected[name])
        for name, value in actual.items()
    }
    if not all(value["pass"] and value["bitwise_equal"]
               for value in checks.values()):
        raise ValueError(f"Timed generation {generation} differs: {checks}")
    outputs = {name: tensor_identity(torch, value)
               for name, value in actual.items()}
    return {
        "generation": generation, "wall_us": wall_us,
        "event_us": event_us, "input_signature": signature(inputs),
        "output_signature": signature(outputs), "outputs": outputs,
        "checks": checks, "pressure_byte": pressure_value,
    }


def worker(args):
    import torch
    import torch_npu
    import attention_check as reference

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    if not any(sku in device_name for sku in ("9392", "9382")):
        raise ValueError(f"Real Ascend A3 required, got {device_name}")
    torch.npu.config.allow_internal_format = True
    torch.ops.load_library(str(args.library.resolve()))
    fixture_args = SimpleNamespace(fixture=None, widths=args.widths)
    inputs = reference.fixture(torch, fixture_args)
    input_identity = {name: tensor_identity(torch, value)
                      for name, value in inputs.items()}
    weights = {}
    for name, value in inputs.items():
        physical_format = 29 if name in {
            "weight_dq", "weight_uq_qr", "weight_dkv_kr",
        } else 2
        weights[name] = value if name == "hidden" else (
            torch_npu.npu_format_cast(value.to(args.device), physical_format))
    pressure = torch.zeros(args.pressure_mib * 1024 * 1024,
                           dtype=torch.uint8, device=args.device)
    pressure_value, state = 0, None
    respond({"ready": True, "pid": os.getpid(), "device_name": device_name,
             "fixture": input_identity,
             "provenance": reference.provenance(torch, torch_npu, args),
             "cpu_affinity": sorted(os.sched_getaffinity(0)),
             "visible_devices": {name: os.environ.get(name) for name in (
                 "ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES",
                 "ASCEND_DEVICE_ID",
             )},
             "cache_paths": {name: os.environ[name]
                             for name in CACHE_VARIABLES}})
    with torch.inference_mode():
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "quit":
                break
            if request["command"] == "prepare":
                if state is not None:
                    state.case.stream.synchronize()
                    state.native_graph.reset()
                    state.vendor_graph.reset()
                    state = None
                state = prepare_case(
                    torch, torch_npu, reference, args, weights,
                    request["width"],
                )
                respond({"gate": state.gate,
                         "cache_base": state.cache_base,
                         "rope_base": state.rope_base})
            elif request["command"] == "measure":
                pressure_value = (pressure_value + 1) % 256
                respond(measure(torch, reference, state, request["generation"],
                                pressure, pressure_value))
            elif request["command"] == "identity":
                paths = {entry.split()[-1] for entry in
                         Path("/proc/self/maps").read_text().splitlines()}
                paths = sorted(path for path in paths if path.startswith("/")
                               and ".so" in path and Path(path).is_file()
                               and ("/Ascend/" in path or "/torch_npu/" in path
                                    or Path(path).resolve() == args.library))
                respond({"libraries": [file_identity(path) for path in paths],
                         "cache_files": [file_identity(path) for path in
                                         Path.cwd().rglob("*")
                                         if path.is_file()]})
            else:
                raise ValueError(f"Unknown worker command: {request}")
    if state is not None:
        state.case.stream.synchronize()
        state.native_graph.reset()
        state.vendor_graph.reset()


def paired_summary(samples):
    for sample in samples:
        baseline, candidate = sample["baseline"], sample["candidate"]
        if (baseline["generation"] != candidate["generation"] or
                baseline["input_signature"] != candidate["input_signature"] or
                baseline["output_signature"] != candidate["output_signature"]):
            raise ValueError("The A/B pair used different inputs or outputs")
    result = {}
    for clock in ("wall_us", "event_us"):
        saving = [sample["baseline"][clock] - sample["candidate"][clock]
                  for sample in samples]
        result[clock] = {
            arm: stats([sample[arm][clock] for sample in samples])
            for arm in ("baseline", "candidate")
        }
        result[clock]["paired_saving"] = stats(saving)
        result[clock]["paired_saving_percent"] = stats([
            (sample["baseline"][clock] - sample["candidate"][clock])
            / sample["baseline"][clock] * 100 for sample in samples
        ])
        result[clock]["wins"] = sum(value > 0 for value in saving)
    return result


def benchmark(args):
    from tilert.pd_vllm.decode_timing import atomic_json

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "one real A3 die, synthetic native Prolog plus SFA graph",
        "hypothesis": "Staged metadata preserves all five boundaries and "
                      "reduces paired width32 wall p50 by at least 5 percent",
        "command": sys.argv, "source": file_identity(__file__),
        "fixture_source": file_identity(
            ROOT / "tests/glm52/whole_layer/attention_check.py"),
        "protocol": {
            "warmup_pairs": args.warmup, "pairs": args.samples,
            "gate_replays": 4, "capture_warmup_calls_per_graph": 3,
            "graph_capture": True, "profiler": False, "order": "AB/BA",
            "wall_clock": "perf_counter_ns includes two events and final sync",
            "device_clock": "NPU events around one native graph replay",
            "outside_timing": "restore KV; change inputs/topk/page table; "
                              "read input hashes; read/write pressure; sync",
            "pressure_mib_per_arm": args.pressure_mib, "outlier_rule": "none",
            "quantile": "nearest index round(q*(N-1)); p50 is median",
        },
        "precision_verified": False, "performance_measured": False,
        "workers": {}, "gates": {}, "results": [],
    }
    processes, logs = {}, []
    try:
        for arm in ("baseline", "candidate"):
            cache = args.output.parent / f"cache-{arm}"
            cache.mkdir()
            env = {**os.environ, "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1"}
            for name in CACHE_VARIABLES:
                path = cache / name.lower()
                path.mkdir()
                env[name] = str(path.resolve())
            log = (args.output.parent / f"{arm}.log").open("w")
            logs.append(log)
            library = getattr(args, arm + "_library")
            process = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--library", str(library), "--device", args.device,
                "--widths", *(str(width) for width in args.widths),
                "--sequence", str(args.sequence), "--blocks", str(args.blocks),
                "--warmup", str(args.warmup), "--samples", str(args.samples),
                "--pressure-mib", str(args.pressure_mib),
            ], cwd=cache, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=log, text=True)
            processes[arm] = process
            report["workers"][arm] = receive(process)
        if report["workers"]["baseline"]["fixture"] != (
                report["workers"]["candidate"]["fixture"]):
            raise ValueError("Frozen CPU weights differ between the two arms")
        for width in args.widths:
            report["gates"][str(width)] = {
                arm: exchange(process, {"command": "prepare", "width": width})
                for arm, process in processes.items()
            }
            samples = []
            for number in range(-args.warmup, args.samples):
                order = ["baseline", "candidate"]
                if number % 2:
                    order.reverse()
                pair = {
                    arm: exchange(processes[arm], {
                        "command": "measure",
                        "generation": number + args.warmup,
                    }) for arm in order
                }
                sample = {"pair": number, "order": order, **pair}
                paired_summary([sample])
                if number >= 0:
                    samples.append(sample)
            report["results"].append({"width": width, "samples": samples,
                                      **paired_summary(samples)})
            atomic_json(args.output, report)
            print(json.dumps({"width": width, **paired_summary(samples)}),
                  flush=True)
        for arm, process in processes.items():
            report["workers"][arm]["loaded"] = exchange(
                process, {"command": "identity"})
        width32 = next((result for result in report["results"]
                        if result["width"] == 32), None)
        report["hypothesis_passed"] = (
            width32["wall_us"]["paired_saving_percent"]["p50"] >= 5
            if width32 is not None else None
        )
        report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.stdin.write('{"command": "quit"}\n')
                process.stdin.flush()
                process.wait(timeout=30)
        for log in logs:
            log.close()
        atomic_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--baseline-library", type=Path)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--sequence", type=int, default=8192)
    parser.add_argument("--blocks", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--pressure-mib", type=int, default=256)
    args = parser.parse_args()
    if args.warmup < 10 or args.samples < 30 or args.pressure_mib <= 192:
        parser.error("Require >=10 warmup, >=30 pairs and >192 MiB pressure")
    if not args.widths or any(width not in (1, 8, 32)
                             for width in args.widths):
        parser.error("Supported measured widths are 1, 8 and 32")
    if args.worker:
        if args.library is None:
            parser.error("--library is required for a worker")
        args.library = args.library.resolve()
        args.fixture = None
        worker(args)
    else:
        if any(path is None for path in (
            args.baseline_library, args.candidate_library, args.output,
        )):
            parser.error("Two arm libraries and --output are required")
        args.baseline_library = args.baseline_library.resolve()
        args.candidate_library = args.candidate_library.resolve()
        args.output = args.output.resolve()
        benchmark(args)


if __name__ == "__main__":
    main()
