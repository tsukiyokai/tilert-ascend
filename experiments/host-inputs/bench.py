"""Measure the actual CannModel input preparer, without model/TP execution."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREFIX = "HOST_INPUT_RESULT "
ARMS = ("separate", "packet")
CACHES = (
    "ASCEND_CACHE_PATH",
    "XDG_CACHE_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "VLLM_CACHE_ROOT",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload(width, generation):
    position = 8192 + generation % 32 * 32
    tokens = tuple(
        (generation * 113 + row * 127) % 154880 for row in range(width)
    )
    return position, tokens


def stats(values):
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        **{
            f"p{p}": ordered[round(p / 100 * (len(ordered) - 1))]
            for p in (90, 95, 99)
        },
        "min": min(ordered),
        "max": max(ordered),
        "std": statistics.pstdev(ordered),
    }


def respond(value):
    print(PREFIX + json.dumps(value), flush=True)


def worker(args):
    import torch
    import torch_npu

    from tilert.models.glm_5_2_ascend.cann_model import CannModel

    torch.set_num_threads(2)
    torch_npu.npu.set_device(0)
    device = torch.npu.get_device_name(0)
    if not any(sku in device for sku in ("9382", "9392")):
        raise ValueError("Use an actual A3 die")
    engine = CannModel.__new__(CannModel)
    engine.device = torch.device("npu:0")
    engine.layout = {"capacity_tokens": 9344}
    engine.capacity = 9344
    engine.packed_host_inputs = args.worker == "packet"
    engine.attention_kernel = "sfa"
    engine.graph_width = 32
    engine.hoist_metadata = False
    engine.builders = []
    engine._allocate_inputs()
    pressure = torch.zeros(256 * 1024**2, dtype=torch.uint8, device="npu:0")
    stream = torch.npu.Stream()
    begin, end = (
        torch.npu.Event(enable_timing=True),
        torch.npu.Event(enable_timing=True),
    )

    def tensors():
        return (
            engine.input_ids,
            engine.positions,
            engine.slots,
            engine.seq_lens,
            engine.query,
        )

    def check(actual, width, generation, depths=None):
        position, tokens = payload(width, generation)
        expected = (
            list(tokens),
            [
                position + row
                for row in (range(width) if depths is None else depths)
            ],
            list(range(position, position + width)),
            [position + width],
            [0, width],
        )
        values = [tensor.cpu().tolist() for tensor in actual]
        if values != list(expected):
            raise ValueError("Persistent consumer inputs differ")
        if [str(t.dtype) for t in actual] != [
            "torch.int32",
            "torch.int64",
            "torch.int32",
            "torch.int32",
            "torch.int32",
        ]:
            raise ValueError("Consumer input dtypes differ")
        return hashlib.sha256(json.dumps(values).encode()).hexdigest()

    def prepare(width, generation, depths=None):
        engine.position, tokens = payload(width, generation)
        returned_width, metadata = engine._prepare_inputs(
            tokens, depths=depths
        )
        if returned_width != width or metadata != {}:
            raise ValueError("Input preparer contract differs")

    graph_checks = []
    with torch.inference_mode(), torch.npu.stream(stream):
        for width in (1, 2, 4, 6, 8, 12, 16, 24, 32):
            prepare(width, 0)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                consumed = tuple(value.clone() for value in tensors())
            torch.npu.synchronize()
            snapshots = []
            for generation in range(32):
                depths = tuple(row // 2 for row in range(width))
                prepare(width, generation, depths)
                graph.replay()
                snapshots.append(tuple(value.clone() for value in consumed))
            torch.npu.synchronize()
            for generation, values in enumerate(snapshots):
                check(
                    values,
                    width,
                    generation,
                    tuple(row // 2 for row in range(width)),
                )
            graph.reset()
            graph_checks.append(
                {
                    "width": width,
                    "queued_replays": 32,
                    "consumer_values_exact": True,
                }
            )
        begin.record()
        end.record()
        end.synchronize()
        respond(
            {
                "ready": True,
                "pid": os.getpid(),
                "device": device,
                "graph_checks": graph_checks,
                "cache_root": str(Path.cwd()),
                "source": {
                    str(path.relative_to(ROOT)): sha(path)
                    for path in (
                        Path(__file__),
                        ROOT / "tilert/models/glm_5_2_ascend/cann_model.py",
                        ROOT / "tilert/models/glm_5_2_ascend/host_inputs.py",
                    )
                },
            }
        )
        addresses = {}
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "quit":
                break
            if request["command"] == "identity":
                libraries = set()
                for entry in Path("/proc/self/maps").read_text().splitlines():
                    path = entry.split()[-1]
                    if (
                        path.startswith("/")
                        and ".so" in path
                        and any(
                            word in path.lower()
                            for word in ("ascend", "torch_npu")
                        )
                        and Path(path).is_file()
                    ):
                        libraries.add(path)
                respond(
                    {
                        "libraries_sha256": {
                            path: sha(path) for path in sorted(libraries)
                        },
                        "packages": {
                            "torch": torch.__version__,
                            "torch_npu": torch_npu.__version__,
                        },
                        "cache_files_sha256": {
                            str(path.relative_to(Path.cwd())): sha(path)
                            for path in sorted(Path.cwd().rglob("*"))
                            if path.is_file()
                        },
                    }
                )
                continue
            width, number = request["width"], request["number"]
            pressure.add_(1)
            torch.npu.synchronize()
            begin.record()
            start = perf_counter_ns()
            prepare(width, number)
            end.record()
            torch.npu.synchronize()
            wall_us = (perf_counter_ns() - start) / 1000
            event_us = begin.elapsed_time(end) * 1000
            values = tensors()
            signature = check(values, width, number)
            pointers = tuple(value.data_ptr() for value in values)
            if width in addresses and addresses[width] != pointers:
                raise ValueError("Persistent input addresses changed")
            addresses[width] = pointers
            respond(
                {
                    "width": width,
                    "number": number,
                    "wall_us": wall_us,
                    "event_us": event_us,
                    "consumer_signature": signature,
                    "input_addresses": pointers,
                }
            )


def receive(process, log):
    for line in process.stdout:
        log.write(line)
        log.flush()
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
    raise RuntimeError(f"Input worker exited: {process.poll()}")


def exchange(process, log, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    return receive(process, log)


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Actual CannModel._prepare_inputs and common metadata; "
        "no vendor builders, weights, TP or model forward",
        "source_commit": args.source_commit,
        "protocol": {
            "warmup_pairs": 12,
            "measured_pairs": 60,
            "pressure_bytes": 256 * 1024**2,
            "pressure_outside_both_clocks": True,
            "order": "AB/BA alternating",
            "profiler": False,
            "graph_in_timing": False,
            "graph_correctness_queued_replays": 32,
        },
        "prediction": "width32 wall/event p50 <= .8 separate, p95 no regression",
        "formal_precision_gate_passed": False,
        "TPOT_measured": False,
        "workers": {},
        "widths": {},
    }
    processes, logs = {}, []
    try:
        for arm in ARMS:
            directory = args.output / arm
            directory.mkdir()
            env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            for variable in CACHES:
                cache = directory / variable.lower()
                cache.mkdir()
                env[variable] = str(cache.resolve())
            log = (args.output / (arm + ".stdout.log")).open("w")
            stderr = (args.output / (arm + ".stderr.log")).open("w")
            logs.extend((log, stderr))
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                arm,
                "--output",
                str(args.output.resolve()),
                "--source-commit",
                args.source_commit,
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                cwd=directory,
                env=env,
            )
            processes[arm] = process, log
            report["workers"][arm] = receive(process, log)
        if (
            report["workers"]["separate"]["source"]
            != report["workers"]["packet"]["source"]
        ):
            raise ValueError("Workers imported different sources")
        for width in (1, 32):
            result = {"warmup": [], "samples": []}
            report["widths"][str(width)] = result
            for number in range(-12, 60):
                order = ARMS if number % 2 == 0 else ARMS[::-1]
                row = {"number": number, "order": order}
                for arm in order:
                    process, log = processes[arm]
                    row[arm] = exchange(
                        process,
                        log,
                        {
                            "command": "measure",
                            "width": width,
                            "number": number,
                        },
                    )
                    if (row[arm]["number"], row[arm]["width"]) != (
                        number,
                        width,
                    ):
                        raise ValueError("Worker sample identity differs")
                if (
                    row["separate"]["consumer_signature"]
                    != row["packet"]["consumer_signature"]
                ):
                    raise ValueError("Arms supply different consumer inputs")
                result["warmup" if number < 0 else "samples"].append(row)
            result["timings"] = {
                arm: {
                    clock: stats(
                        [row[arm][clock] for row in result["samples"]]
                    )
                    for clock in ("wall_us", "event_us")
                }
                for arm in ARMS
            }
            result["paired_savings_us"] = {
                clock: stats(
                    [
                        row["separate"][clock] - row["packet"][clock]
                        for row in result["samples"]
                    ]
                )
                for clock in ("wall_us", "event_us")
            }
            timing = result["timings"]
            result["prediction_met"] = all(
                timing["packet"][clock]["p50"]
                <= 0.8 * timing["separate"][clock]["p50"]
                and timing["packet"][clock]["p95"]
                <= timing["separate"][clock]["p95"]
                for clock in ("wall_us", "event_us")
            )
            print(
                json.dumps(
                    {
                        "width": width,
                        "timings": timing,
                        "prediction_met": result["prediction_met"],
                    }
                ),
                flush=True,
            )
        for arm, (process, log) in processes.items():
            report["workers"][arm]["final_identity"] = exchange(
                process, log, {"command": "identity"}
            )
        report["status"] = "measured"
    finally:
        for process, _ in processes.values():
            if process.poll() is None:
                try:
                    process.stdin.write('{"command":"quit"}\n')
                    process.stdin.flush()
                    process.wait(timeout=30)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait(timeout=10)
        report["worker_exit_codes"] = {
            arm: process.returncode for arm, (process, _) in processes.items()
        }
        for log in logs:
            log.close()
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    if any(code != 0 for code in report["worker_exit_codes"].values()):
        raise RuntimeError("A host-input worker failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--worker", choices=ARMS)
    args = parser.parse_args()
    worker(args) if args.worker else benchmark(args)


if __name__ == "__main__":
    main()
