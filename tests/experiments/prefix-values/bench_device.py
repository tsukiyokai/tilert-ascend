"""Compare frozen native, expanded and cached M1 attention on one A3 die."""

import argparse
import ast
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

from check_device import FIXTURE_SHA, ORIGINAL_SHA, digest, module

PREFIX = "TRT_RESULT "
ARMS = ("native", "expanded", "cached")
DEVICE_SHA = "b7062ab6757b52a1ce95844dd2c190b7154e5dbb11cc1b31545a13c1628abbb8"
SFA_SHA = "b5ad39a05b1323ac5048962f137969158298e4f637b5bd3e401cb4c4248ecfa0"
CACHED_SHA = "a531c7db7921e9faad64259ee7810d3b3439b4307b116e6f20de64ce5fdf9a43"
NATIVE_SHA = "5abcd3f3418299a7bb4ac5e8b5df1cc9201c3a4e1b09e0b100f8ca6bb4c7f2f6"


def respond(value):
    print(PREFIX + json.dumps(value), flush=True)


def tensor_sha(torch, value):
    raw = value.cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def frozen_function(path, name, namespace):
    source = path.read_text()
    functions = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise ValueError(f"Expected exactly one frozen method: {name}")
    function = functions[0]
    function.decorator_list = []
    tree = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - SHA-pinned original methods.
    return namespace[name]


def worker(args):
    import torch
    import torch_npu

    torch.set_num_threads(8)
    torch.npu.set_device(0)
    name = torch.npu.get_device_name(0)
    if not any(sku in name for sku in ("9392", "9382")):
        raise ValueError(f"A3 required: {name}")
    for path, sha in [
        (args.fixture, FIXTURE_SHA),
        (args.original, ORIGINAL_SHA),
        (args.cached, CACHED_SHA),
        (args.device_source, DEVICE_SHA),
        (args.sfa_source, SFA_SHA),
        (args.native_capture, NATIVE_SHA),
    ]:
        if digest(path) != sha:
            raise ValueError(f"Frozen source/input differs: {path}")
    fixture = torch.load(args.fixture, weights_only=True, map_location="cpu")
    native = torch.load(
        args.native_capture, weights_only=True, map_location="cpu"
    )
    receipt = json.loads(args.bank_receipt.read_text())
    if (
        receipt["rank"] != 2
        or receipt["file_sha256"] != digest(args.bank)
        or receipt["prefix"]["prefix_sha256"]
        != fixture["capture_inputs_identity"]["prefix_sha256"]
        or receipt["checkpoint"]["metadata_sha256"]
        != fixture["identity"]["checkpoint_metadata_sha256"]
    ):
        raise ValueError("Bank identity differs from the frozen fixture")
    for key, value in fixture["inputs"].items():
        other = native["boundaries"]["sfa." + key]
        if key == "block_table":
            continue
        if key in ("key_cache", "rope_cache"):
            positions = torch.arange(8193)
            table = fixture["inputs"]["block_table"][0].long()
            physical = table[positions // 128] * 128 + positions % 128
            value = value.flatten(0, 1)[physical]
            table = native["boundaries"]["sfa.block_table"][0].long()
            physical = table[positions // 128] * 128 + positions % 128
            other = other.flatten(0, 1)[physical]
        if not torch.equal(value, other):
            raise ValueError(f"Native and expanded input differs: {key}")
    bank = torch.load(
        args.bank, weights_only=True, map_location="cpu", mmap=True
    )
    if (
        bank.shape != (78, 4, 8192, 256)
        or bank.dtype != torch.bfloat16
        or not torch.equal(bank[0], fixture["cpu"]["value_prefix"])
    ):
        raise ValueError("Bank is not the previously verified prefix")
    inputs = {
        key: value.to("npu:0") for key, value in fixture["inputs"].items()
    }
    before = {key: tensor_sha(torch, value) for key, value in inputs.items()}
    torch.npu.matmul.allow_hf32 = False
    impl = SimpleNamespace(
        W_UV=fixture["weights"]["uv"].to("npu:0"),
        scale=0.0625,
        local_num_heads=4,
        kv_lora_rank=512,
        v_head_dim=256,
        _execute_sparse_flash_attention_process=None,
        _v_up_proj=None,
    )
    prefix = bank[0].to("npu:0")
    metadata = SimpleNamespace(block_table=inputs["block_table"])
    if args.worker == "native":
        torch.ops.load_library(str(args.custom_library))
        scope = {"torch": torch, "torch_npu": torch_npu}
        attention = frozen_function(
            args.device_source, "execute_sparse_flash_attention_process", scope
        )
        projection = frozen_function(args.sfa_source, "_v_up_proj", scope)
        impl._execute_sparse_flash_attention_process = lambda *values: (
            attention(None, impl, *values)
        )
        impl._v_up_proj = lambda value: projection(impl, value)
        expected = native["boundaries"]["o_proj.input"]
    elif args.worker == "expanded":
        module(args.original, "expanded").install(impl)
        expected = fixture["full_npu"]["o_proj_input"]
    else:
        module(args.cached, "cached").install(impl, prefix)
        expected = fixture["cpu"]["pv"].reshape(1, 1024)

    def execute():
        output = impl._execute_sparse_flash_attention_process(
            inputs["query_nope"],
            inputs["query_rope"],
            (inputs["key_cache"], inputs["rope_cache"]),
            inputs["indices"],
            metadata,
            inputs["query_lengths"],
            inputs["key_lengths"],
        )
        return impl._v_up_proj(output)

    pressure = torch.zeros(
        args.pressure_mib * 1024**2, dtype=torch.uint8, device="npu:0"
    )
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    output_sha = tensor_sha(torch, expected)
    respond(
        {
            "ready": True,
            "pid": os.getpid(),
            "device": name,
            "cache_root": str(Path.cwd()),
            "expected_sha256": output_sha,
            "native_capture_sha256": digest(args.native_capture),
            "bank_file_sha256": receipt["file_sha256"],
        }
    )
    with torch.inference_mode():
        for line in sys.stdin:
            request = json.loads(line)
            if request["command"] == "quit":
                break
            if request["command"] == "identity":
                after = {
                    key: tensor_sha(torch, value)
                    for key, value in inputs.items()
                }
                paths = set()
                for row in Path("/proc/self/maps").read_text().splitlines():
                    path = row.split()[-1]
                    if (
                        path.startswith("/")
                        and ".so" in path
                        and (
                            "/Ascend/" in path
                            or "/ascend/" in path
                            or "/torch_npu/" in path
                            or "vllm_ascend_C" in path
                        )
                        and Path(path).is_file()
                    ):
                        paths.add(path)
                respond(
                    {
                        "inputs_unchanged": before == after,
                        "prefix_unchanged": tensor_sha(torch, prefix)
                        == tensor_sha(torch, bank[0]),
                        "packages": {
                            key: version(key) for key in ("torch", "torch-npu")
                        },
                        "libraries": {
                            path: digest(path) for path in sorted(paths)
                        },
                        "cache_files": {
                            str(path.relative_to(Path.cwd())): digest(path)
                            for path in sorted(Path.cwd().rglob("*"))
                            if path.is_file()
                        },
                    }
                )
                continue
            pressure.add_(1)
            torch.npu.synchronize()
            start_event.record()
            start = perf_counter_ns()
            output = execute()
            end_event.record()
            torch.npu.synchronize()
            wall_us = (perf_counter_ns() - start) / 1000
            actual_sha = tensor_sha(torch, output)
            if actual_sha != output_sha:
                raise ValueError(
                    f"{args.worker} does not reproduce its frozen output"
                )
            respond(
                {
                    "wall_us": wall_us,
                    "event_us": start_event.elapsed_time(end_event) * 1000,
                    "output_sha256": actual_sha,
                }
            )


def exchange(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    for line in process.stdout:
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
    raise RuntimeError(f"Worker ended with {process.poll()}")


def ready(process):
    for line in process.stdout:
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX) :])
    raise RuntimeError(f"Worker startup ended with {process.poll()}")


def stats(values):
    values = sorted(values)
    return {
        "p50": statistics.median(values),
        "p90": values[round(0.90 * (len(values) - 1))],
        "p95": values[round(0.95 * (len(values) - 1))],
        "p99": values[round(0.99 * (len(values) - 1))],
        "std": statistics.pstdev(values),
    }


def benchmark(args):
    args.output.parent.mkdir(exist_ok=False, parents=True)
    report = {
        "status": "running",
        "scope": "A3 rank2 eager M1/8193 attention and UV; one frozen actual input",
        "created_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "command": sys.argv,
        "source_commit": args.source_commit,
        "script_sha256": digest(__file__),
        "prediction": "Cached wall p50 <= 0.8 expanded; performance adoption requires cached wall/event p50 <= 0.95 native",
        "protocol": {
            "warmup": args.warmup,
            "samples": args.samples,
            "order": "all six permutations repeated",
            "profiler": False,
            "graph": False,
            "pressure_mib": args.pressure_mib,
            "outlier_rule": "none",
            "timing": "pressure outside; full attention and UV inside synchronized wall/event",
        },
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "workers": {},
        "samples": [],
    }
    processes, logs = {}, []
    try:
        for arm in ARMS:
            cache = args.output.parent / ("cache-" + arm)
            cache.mkdir()
            env = {
                **os.environ,
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
            }
            for name in (
                "ASCEND_CACHE_PATH",
                "XDG_CACHE_HOME",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "VLLM_CACHE_ROOT",
            ):
                directory = cache / name.lower()
                directory.mkdir()
                env[name] = str(directory)
            log = (args.output.parent / (arm + ".log")).open("w")
            logs.append(log)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                arm,
                "--source-commit",
                args.source_commit,
                "--pressure-mib",
                str(args.pressure_mib),
            ]
            for key in (
                "fixture",
                "native_capture",
                "bank",
                "bank_receipt",
                "original",
                "cached",
                "device_source",
                "sfa_source",
                "custom_library",
                "output",
            ):
                command.extend(
                    [
                        "--" + key.replace("_", "-"),
                        str(getattr(args, key).resolve()),
                    ]
                )
            processes[arm] = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                cwd=cache,
                env=env,
            )
            report["workers"][arm] = ready(processes[arm])
        orders = (
            ("native", "expanded", "cached"),
            ("cached", "expanded", "native"),
            ("expanded", "native", "cached"),
            ("cached", "native", "expanded"),
            ("native", "cached", "expanded"),
            ("expanded", "cached", "native"),
        )
        for number in range(-args.warmup, args.samples):
            order = orders[number % len(orders)]
            sample = {"number": number, "order": order}
            for arm in order:
                sample[arm] = exchange(processes[arm], {"command": "measure"})
            if number >= 0:
                report["samples"].append(sample)
        report["timings"] = {
            arm: {
                clock: stats([row[arm][clock] for row in report["samples"]])
                for clock in ("wall_us", "event_us")
            }
            for arm in ARMS
        }
        report["paired_savings"] = {
            arm: {
                clock: stats(
                    [
                        row[arm][clock] - row["cached"][clock]
                        for row in report["samples"]
                    ]
                )
                for clock in ("wall_us", "event_us")
            }
            for arm in ("native", "expanded")
        }
        for arm, process in processes.items():
            identity = exchange(process, {"command": "identity"})
            report["workers"][arm]["identity"] = identity
            if (
                not identity["inputs_unchanged"]
                or not identity["prefix_unchanged"]
            ):
                raise ValueError(f"{arm} mutated its inputs")
        times = report["timings"]
        report["projection_prediction_met"] = (
            times["cached"]["wall_us"]["p50"]
            <= 0.8 * times["expanded"]["wall_us"]["p50"]
        )
        report["native_performance_adoption_met"] = all(
            times["cached"][clock]["p50"]
            <= 0.95 * times["native"][clock]["p50"]
            for clock in ("wall_us", "event_us")
        )
        report.update(status="measured", performance_measured=True)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "timings",
                        "paired_savings",
                        "projection_prediction_met",
                        "native_performance_adoption_met",
                    )
                }
            ),
            flush=True,
        )
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                try:
                    exchange(process, {"command": "quit"})
                except (BrokenPipeError, RuntimeError):
                    pass
                process.wait(timeout=30)
        for log in logs:
            log.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=ARMS)
    for key in (
        "fixture",
        "native-capture",
        "bank",
        "bank-receipt",
        "original",
        "cached",
        "device-source",
        "sfa-source",
        "custom-library",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--pressure-mib", type=int, default=256)
    args = parser.parse_args()
    if args.samples < 30 or args.warmup < 10 or args.pressure_mib <= 192:
        parser.error("Require >=30 samples, >=10 warmups and >192MiB pressure")
    if args.worker:
        worker(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
