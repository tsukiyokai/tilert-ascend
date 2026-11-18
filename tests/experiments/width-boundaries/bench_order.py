"""Three-process SFA order benchmark on the frozen rank12 width1/32 inputs."""

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import sfa_order as numerical

torch = numerical.torch
ARMS = ("native", "preordered", "sort_each")
ORDERS = tuple(itertools.permutations(ARMS))
HELPER_SHA = "321a364865dca62523fe815f5bcee1b1bcd861fa83180f4799fb4c85a92ed6a5"
SORTED_WIDE_SHA = "c00bb49545de34fbb9a18f3f342b7f1100ac3274523e5403415fdcaef690030e"
CACHE_ENV = (
    "ASCEND_CACHE_PATH",
    "XDG_CACHE_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "VLLM_CACHE_ROOT",
)
PREFIX = "ORDER_RESULT "
require, digest = numerical.require, numerical.digest


def logical(value, table, length):
    positions = torch.arange(length)
    pages = table[0].long()[positions // 128]
    require(
        bool(((pages >= 0) & (pages < value.shape[0])).all()),
        "Invalid actual page table",
    )
    return value.reshape(-1, value.shape[-1])[pages * 128 + positions % 128]


def fixtures(capture):
    require(
        digest(Path(numerical.__file__)) == HELPER_SHA,
        "Frozen numerical helper changed",
    )
    one = numerical.admit(capture)
    report = json.loads((capture / "rank-12.json").read_text())
    require(
        report["trace_sha256"]["sorted-wide.pt"] == SORTED_WIDE_SHA,
        "Sorted wide report binding differs",
    )
    path = capture / "rank-12/sorted-wide.pt"
    sorted_identity = numerical.bind(path, SORTED_WIDE_SHA)
    original = torch.load(
        capture / "rank-12/original-wide.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    sorted_frame = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    numerical.sources(
        sorted_frame["observed"]["metadata"]["source_sha256"],
        numerical.OBSERVED_SOURCES,
    )
    numerical.extract(original, wide=True)
    numerical.extract(sorted_frame, wide=True)
    a, b = (frame["observed"]["tensors"] for frame in (original, sorted_frame))
    key = numerical.PREFIX
    for name in ("ql_nope", "q_pe"):
        require(
            numerical.exact(a[key + name], b[key + name]),
            "Full width32 query differs: " + name,
        )
    for number in (0, 1):
        name = f"kv_cache.{number}"
        require(
            numerical.exact(
                logical(a[key + name], a[key + "block_table"], 8224),
                logical(b[key + name], b[key + "block_table"], 8224),
            ),
            "Full width32 effective KV differs",
        )
    indices = a[key + "topk_indices"]
    ordered = indices.float().sort(dim=-1).values.to(indices.dtype)
    require(
        numerical.exact(indices, b["intervention.original_indices"])
        and numerical.exact(ordered, b[key + "topk_indices"]),
        "Full width32 sorted consumer/source indices differ",
    )
    for row, value in enumerate(indices.reshape(32, -1)):
        require(
            value.numel() == value.unique().numel() == 2048
            and bool(((value >= 0) & (value <= 8192 + row)).all()),
            "Invalid full width32 causal indices",
        )
    lengths = {
        name: numerical.length(original, name)
        for name in ("actual_seq_lengths_query", "actual_seq_lengths_key")
    }
    require(
        all(
            numerical.length_identity(value)
            == numerical.length_identity(numerical.length(sorted_frame, name))
            for name, value in lengths.items()
        ),
        "Width32 metadata type/value differs",
    )
    require(
        all(sorted_frame["observed"]["coverage"].values()),
        "Sorted wide coverage failed",
    )
    inputs = {name: a[key + name].clone() for name in one["inputs"]}
    expected = {
        "native": a[key + "out"].clone(),
        "preordered": b[key + "out"].clone(),
        "sort_each": b[key + "out"].clone(),
    }
    require(
        all(
            value.shape == (32, 4, 512)
            and value.dtype == torch.bfloat16
            and bool(torch.isfinite(value).all())
            for value in expected.values()
        ),
        "Frozen full width32 output differs",
    )
    shared = {
        "reference_sha256": numerical.REFERENCE_SHA,
        "identities": one["identities"],
        "checkpoint_sha256": one["checkpoint_sha256"],
        "capture_inputs": one["capture_inputs"],
    }
    return {
        1: {
            **shared,
            "width": 1,
            "inputs": one["inputs"],
            "ordered": one["orders"]["sorted"],
            "lengths": one["lengths"],
            "scale": one["scale"],
            "expected": {
                "native": one["expected"]["original"],
                "preordered": one["expected"]["sorted"],
                "sort_each": one["expected"]["sorted"],
            },
        },
        32: {
            **shared,
            "width": 32,
            "inputs": inputs,
            "ordered": ordered,
            "lengths": lengths,
            "scale": one["scale"],
            "expected": expected,
            "sorted_wide_identity": sorted_identity,
        },
    }


def make_execute(arm, function, inputs, ordered, lengths, scale):
    metadata, impl = (
        SimpleNamespace(block_table=inputs["block_table"]),
        SimpleNamespace(scale=scale),
    )

    def execute():
        indices = inputs["topk_indices"]
        if arm == "preordered":
            indices = ordered
        elif arm == "sort_each":
            indices = indices.float().sort(dim=-1).values.to(indices.dtype)
        return function(
            None,
            impl,
            inputs["ql_nope"],
            inputs["q_pe"],
            (inputs["kv_cache.0"], inputs["kv_cache.1"]),
            indices,
            metadata,
            lengths["actual_seq_lengths_query"],
            lengths["actual_seq_lengths_key"],
            block_table=inputs["block_table"],
        )

    return execute


def measure(execute, pressure, begin, end):
    pressure.add_(1)
    torch.npu.synchronize()
    begin.record()
    start = perf_counter_ns()
    output = execute()
    end.record()
    torch.npu.synchronize()
    return output, {
        "wall_us": (perf_counter_ns() - start) / 1000,
        "event_us": begin.elapsed_time(end) * 1000,
    }


def respond(value):
    print(PREFIX + json.dumps(value), flush=True)


def worker(args):
    import torch_npu

    torch.set_num_threads(2)
    require(digest(Path(numerical.__file__)) == HELPER_SHA, "Numerical helper changed")
    numerical.bind(args.fixture, args.fixture_sha)
    numerical.bind(args.device_source, numerical.DEVICE_SHA)
    numerical.bind(args.library, numerical.LIBRARY_SHA)
    fixture = torch.load(args.fixture, map_location="cpu", mmap=True, weights_only=True)
    require(fixture["width"] == args.width, "Worker fixture width differs")
    for name in CACHE_ENV:
        path = Path(os.environ[name]).resolve()
        require(
            path.is_dir() and path.is_relative_to(Path.cwd().resolve()),
            "Worker cache is not private",
        )
    torch_npu.npu.set_device(0)
    device = torch.npu.get_device_name(0)
    require(any(sku in device for sku in ("9382", "9392")), "Actual A3 required")
    torch.npu.matmul.allow_hf32 = False
    torch.ops.load_library(str(args.library.resolve()))
    function, ast_sha = numerical.method(args.device_source)
    inputs = {name: value.to("npu:0") for name, value in fixture["inputs"].items()}
    ordered = fixture["ordered"].to("npu:0")
    lengths = {
        name: numerical.runtime_length(value, "npu:0")
        for name, value in fixture["lengths"].items()
    }
    execute = make_execute(
        args.worker, function, inputs, ordered, lengths, fixture["scale"]
    )
    expected_sha = numerical.raw_sha(fixture["expected"][args.worker])

    def fingerprints():
        return {
            **{name: numerical.raw_sha(value) for name, value in inputs.items()},
            "ordered": numerical.raw_sha(ordered),
            **{
                "length." + name: numerical.raw_sha(value)
                if isinstance(value, torch.Tensor)
                else list(value)
                for name, value in lengths.items()
            },
        }

    before = fingerprints()
    with torch.inference_mode():
        initial = execute()
        torch.npu.synchronize()
        require(
            numerical.raw_sha(initial) == expected_sha and fingerprints() == before,
            "Startup does not reproduce the frozen SFA output or modifies input",
        )
        pressure = torch.zeros(256 * 1024**2, dtype=torch.uint8, device="npu:0")
        begin, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        respond(
            {
                "ready": True,
                "arm": args.worker,
                "width": args.width,
                "pid": os.getpid(),
                "device": device,
                "cache_root": str(Path.cwd()),
                "input_fingerprints": before,
                "fixture_sha256": args.fixture_sha,
                "expected_output_sha256": expected_sha,
                "startup_output_sha256": numerical.raw_sha(initial),
                "script_sha256": digest(Path(__file__)),
                "helper_sha256": HELPER_SHA,
                "original_method_ast_source_sha256": ast_sha,
                "lengths": {
                    name: numerical.length_identity(value)
                    for name, value in fixture["lengths"].items()
                },
            }
        )
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "quit":
                break
            if request["command"] == "identity":
                respond(
                    {
                        "inputs_unchanged": fingerprints() == before,
                        "libraries_sha256": numerical.mapped_libraries(args.library),
                        "packages": {
                            "torch": torch.__version__,
                            "torch-npu": version("torch-npu"),
                        },
                        "cache_files_sha256": {
                            str(path.relative_to(Path.cwd())): digest(path)
                            for path in sorted(Path.cwd().rglob("*"))
                            if path.is_file()
                        },
                    }
                )
                continue
            require(request["command"] == "measure", "Unknown worker command")
            output, times = measure(execute, pressure, begin, end)
            sha = numerical.raw_sha(output)
            after = fingerprints()
            require(
                sha == expected_sha,
                "Measured sample differs from its entire frozen output",
            )
            require(after == before, "Measured sample modified an input")
            respond(
                {
                    "number": request["number"],
                    "arm": args.worker,
                    "width": args.width,
                    **times,
                    "output_sha256": sha,
                    "inputs_unchanged": True,
                    "input_fingerprints": after,
                }
            )


def receive(process, log):
    for line in process.stdout:
        log.write(line)
        log.flush()
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
    raise RuntimeError(f"Worker exited with {process.poll()}; inspect its log")


def exchange(process, log, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    return receive(process, log)


def stats(values):
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p90": ordered[round(0.90 * (len(ordered) - 1))],
        "p95": ordered[round(0.95 * (len(ordered) - 1))],
        "p99": ordered[round(0.99 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
        "std": statistics.pstdev(ordered),
    }


def summarize(samples):
    times = {
        arm: {
            clock: stats([sample[arm][clock] for sample in samples])
            for clock in ("wall_us", "event_us")
        }
        for arm in ARMS
    }
    passes = {
        arm: all(
            times[arm][clock][quantile] <= 0.95 * times["native"][clock][quantile]
            for clock in ("wall_us", "event_us")
            for quantile in ("p50", "p95")
        )
        for arm in ("preordered", "sort_each")
    }
    return {
        "timings": times,
        "preordered_prediction_met": passes["preordered"],
        "sort_each_including_cost_prediction_met": passes["sort_each"],
        "paired_savings_us": {
            arm: {
                clock: {
                    **stats(
                        [row["native"][clock] - row[arm][clock] for row in samples]
                    ),
                    "wins": sum(
                        row[arm][clock] < row["native"][clock] for row in samples
                    ),
                }
                for clock in ("wall_us", "event_us")
            }
            for arm in ("preordered", "sort_each")
        },
    }


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "sfa-order-component-benchmark-v1",
        "status": "preparing",
        "scope": "One full SFA call, frozen rank12 inputs, width1 and actual full width32; no UV/TP/TPOT",
        "source_commit": args.source_commit,
        "source_sha256": {
            "bench_order.py": digest(Path(__file__)),
            "sfa_order.py": digest(Path(numerical.__file__)),
        },
        "device_source": numerical.bind(args.device_source, numerical.DEVICE_SHA),
        "library": numerical.bind(args.library, numerical.LIBRARY_SHA),
        "protocol": {
            "warmup_groups": 12,
            "measured_groups": 60,
            "orders": ORDERS,
            "pressure_bytes": 256 * 1024**2,
            "pressure": "uint8 inplace read/write add, synchronized outside both clocks",
            "interval": "one SFA; sort_each includes FP32 cast, position sort and integer cast",
            "output_and_input_sha": "entire tensors after synchronization, outside timing",
            "preordered": "sorted once in CPU preparation, excluded from interval",
            "quantiles": "median and nearest round(p*(n-1))",
            "profiler": False,
            "graph": False,
        },
        "prediction": "preordered wall/event p50/p95 <= .95 native; sort_each cost-inclusive result separate",
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "widths": {},
    }
    try:
        torch.set_num_threads(2)
        prepared = fixtures(args.capture)
        for width, fixture in prepared.items():
            fixture_path = args.output / f"inputs-width{width}.pt"
            torch.save(fixture, fixture_path)
            fixture_sha = digest(fixture_path)
            result = {
                "fixture_sha256": fixture_sha,
                "capture_identities": fixture["identities"],
                "workers": {},
                "warmup": [],
                "samples": [],
            }
            if width == 32:
                result["sorted_wide_identity"] = fixture["sorted_wide_identity"]
            report["widths"][str(width)] = result
            processes, streams = {}, []
            try:
                for arm in ARMS:
                    cache = args.output / f"cache-width{width}-{arm}"
                    cache.mkdir()
                    environment = {
                        **os.environ,
                        "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2",
                    }
                    for name in CACHE_ENV:
                        directory = cache / name.lower()
                        directory.mkdir()
                        environment[name] = str(directory.resolve())
                    stdout = (args.output / f"width{width}-{arm}.stdout.log").open("w")
                    stderr = (args.output / f"width{width}-{arm}.stderr.log").open("w")
                    streams.extend((stdout, stderr))
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        arm,
                        "--width",
                        str(width),
                        "--fixture",
                        str(fixture_path.resolve()),
                        "--fixture-sha",
                        fixture_sha,
                    ]
                    for key in ("capture", "device_source", "library", "output"):
                        command.extend(
                            (
                                "--" + key.replace("_", "-"),
                                str(getattr(args, key).resolve()),
                            )
                        )
                    command.extend(("--source-commit", args.source_commit))
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=stderr,
                        text=True,
                        cwd=cache,
                        env=environment,
                    )
                    processes[arm] = (process, stdout)
                    ready = receive(process, stdout)
                    require(
                        ready["arm"] == arm
                        and ready["width"] == width
                        and ready["fixture_sha256"] == fixture_sha
                        and ready["script_sha256"]
                        == report["source_sha256"]["bench_order.py"],
                        "Worker identity differs",
                    )
                    result["workers"][arm] = ready
                require(
                    len({item["cache_root"] for item in result["workers"].values()})
                    == 3,
                    "Workers share a cache root",
                )
                for number in range(-12, 60):
                    order = ORDERS[number % 6]
                    sample = {"number": number, "order": order}
                    for arm in order:
                        process, log = processes[arm]
                        response = exchange(
                            process, log, {"command": "measure", "number": number}
                        )
                        require(
                            response["number"] == number
                            and response["arm"] == arm
                            and response["width"] == width
                            and response["inputs_unchanged"],
                            "Sample identity differs",
                        )
                        sample[arm] = response
                    result["warmup" if number < 0 else "samples"].append(sample)
                result.update(summarize(result["samples"]))
                for arm, (process, log) in processes.items():
                    identity = exchange(process, log, {"command": "identity"})
                    require(
                        identity["inputs_unchanged"]
                        and identity["libraries_sha256"][str(args.library.resolve())]
                        == numerical.LIBRARY_SHA,
                        "Worker input or loaded library identity changed",
                    )
                    result["workers"][arm]["final_identity"] = identity
                print(
                    json.dumps(
                        {
                            "width": width,
                            "timings": result["timings"],
                            "preordered_prediction_met": result[
                                "preordered_prediction_met"
                            ],
                            "sort_each_including_cost_prediction_met": result[
                                "sort_each_including_cost_prediction_met"
                            ],
                        }
                    ),
                    flush=True,
                )
            finally:
                for process, log in processes.values():
                    if process.poll() is None:
                        try:
                            process.stdin.write('{"command":"quit"}\n')
                            process.stdin.flush()
                            process.wait(timeout=30)
                        except (BrokenPipeError, subprocess.TimeoutExpired):
                            process.kill()
                            process.wait(timeout=10)
                for stream in streams:
                    stream.close()
            require(
                all(process.returncode == 0 for process, _ in processes.values()),
                "A worker failed",
            )
        report.update(status="measured", performance_measured=True)
    except Exception:  # noqa: BLE001 - persist failed benchmark receipt, return nonzero below.
        report.update(status="fail", error=traceback.format_exc())
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return 0 if report["status"] == "measured" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "device-source", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--worker", choices=ARMS)
    parser.add_argument("--width", type=int, choices=(1, 32))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-sha")
    args = parser.parse_args()
    if args.worker:
        require(
            args.width and args.fixture and args.fixture_sha,
            "Worker needs a bound width fixture",
        )
        worker(args)
        return 0
    return benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
