"""Independently validate the saved host-input benchmark using only stdlib."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

ARMS = ("separate", "packet")
WIDTHS = (1, 2, 4, 6, 8, 12, 16, 24, 32)
CLOCKS = ("wall_us", "event_us")
SOURCES = (
    "experiments/host-inputs/bench.py",
    "tilert/models/glm_5_2_ascend/cann_model.py",
    "tilert/models/glm_5_2_ascend/host_inputs.py",
)
CID = "0bb5b86e81c9ce1218a5b94829ab87a1fe1484b681a882d6996a287ba2326376"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_signature(width, number):
    position = 8192 + (number % 32) * 32
    ids = [(number * 113 + index * 127) % 154880 for index in range(width)]
    linear = list(range(position, position + width))
    values = [ids, linear, linear, [position + width], [0, width]]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def statistics(values):
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    mean = math.fsum(ordered) / n
    return {
        "p50": median,
        **{f"p{q}": ordered[round(q / 100 * (n - 1))] for q in (90, 95, 99)},
        "min": ordered[0],
        "max": ordered[-1],
        "std": math.sqrt(math.fsum((value - mean) ** 2 for value in ordered) / n),
    }


def compare_statistics(actual, reported, label):
    require(actual.keys() == reported.keys(), "Statistics fields differ: " + label)
    require(
        all(
            type(reported[key]) in (int, float)
            and math.isfinite(reported[key])
            and math.isclose(value, reported[key], rel_tol=1e-12, abs_tol=1e-9)
            for key, value in actual.items()
        ),
        "Statistics differ: " + label,
    )


def terminal(receipt):
    state = receipt["state"]
    require(
        receipt["id"] == CID
        and receipt["docker_wait_exit"] == "0"
        and state["Status"] == "exited"
        and state["Running"] is False
        and type(state["ExitCode"]) is int
        and state["ExitCode"] == 0
        and state["OOMKilled"] is False,
        "Actual container did not exit cleanly",
    )
    return {
        "id": receipt["id"],
        "docker_wait_exit": receipt["docker_wait_exit"],
        "Status": state["Status"],
        "ExitCode": state["ExitCode"],
    }


def workers(report, source):
    require(set(report["workers"]) == set(ARMS), "Need both worker identities")
    require(
        set(report["worker_exit_codes"]) == set(ARMS)
        and all(
            type(report["worker_exit_codes"][arm]) is int
            and report["worker_exit_codes"][arm] == 0
            for arm in ARMS
        ),
        "A worker exited nonzero",
    )
    baseline = report["workers"]["separate"]["source"]
    require(set(baseline) == set(SOURCES), "Producer source inventory differs")
    actual_source = {name: digest(source / name) for name in SOURCES}
    require(actual_source == baseline, "Provided source bytes differ from producer")
    roots, pids = [], []
    for arm in ARMS:
        worker = report["workers"][arm]
        require(
            worker["ready"] is True
            and worker["source"] == baseline
            and worker["device"] in ("Ascend910_9382", "Ascend910_9392"),
            "Worker source/A3 identity differs",
        )
        require(type(worker["pid"]) is int and worker["pid"] > 0, "Invalid worker pid")
        pids.append(worker["pid"])
        root = PurePosixPath(worker["cache_root"])
        require(
            root.is_absolute() and root.name == arm and ".." not in root.parts,
            "Worker cache root differs",
        )
        roots.append(root)
        final = worker["final_identity"]
        require(
            final["libraries_sha256"]
            and final["packages"].get("torch")
            and final["packages"].get("torch_npu"),
            "Worker lacks loaded runtime identity",
        )
        for name in final["cache_files_sha256"]:
            path = PurePosixPath(name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and path.parts[0]
                in {
                    name.lower()
                    for name in (
                        "ASCEND_CACHE_PATH",
                        "XDG_CACHE_HOME",
                        "TORCHINDUCTOR_CACHE_DIR",
                        "TRITON_CACHE_DIR",
                        "VLLM_CACHE_ROOT",
                    )
                },
                "Recorded cache file escapes the private root",
            )
        graph = worker["graph_checks"]
        require(
            len(graph) == len(WIDTHS)
            and {item["width"] for item in graph} == set(WIDTHS)
            and all(
                type(item["width"]) is int
                and item["queued_replays"] == 32
                and item["consumer_values_exact"] is True
                for item in graph
            ),
            "Nine-width queued graph checks incomplete",
        )
    require(
        len(set(pids)) == 2
        and len(set(roots)) == 2
        and not roots[0].is_relative_to(roots[1])
        and not roots[1].is_relative_to(roots[0]),
        "Workers do not have distinct process/cache roots",
    )
    require(
        report["workers"]["packet"]["final_identity"]["packages"]
        == report["workers"]["separate"]["final_identity"]["packages"],
        "Worker package versions differ",
    )
    return actual_source


def addresses(values, width, arm):
    require(
        len(values) == 5 and all(type(value) is int and value > 0 for value in values),
        "Invalid input addresses",
    )
    sizes, alignment = (4 * width, 8 * width, 4 * width, 4, 8), (4, 8, 4, 4, 4)
    require(
        all(value % align == 0 for value, align in zip(values, alignment, strict=True)),
        "Input address alignment differs",
    )
    regions = sorted(
        (value, value + size) for value, size in zip(values, sizes, strict=True)
    )
    require(
        all(regions[index][1] <= regions[index + 1][0] for index in range(4)),
        "Input fields overlap",
    )
    if arm == "packet":
        position = (4 * width + 7) // 8 * 8
        slots = position + 8 * width
        length = (slots + 4 * width + 7) // 8 * 8
        require(
            values[:4]
            == [values[0] + offset for offset in (0, position, slots, length)],
            "Packet view offsets differ",
        )


def read(report, source):
    require(
        report["status"] == "measured"
        and report["formal_precision_gate_passed"] is False
        and report["TPOT_measured"] is False,
        "Report scope/status differs",
    )
    require(
        report["protocol"]
        == {
            "warmup_pairs": 12,
            "measured_pairs": 60,
            "pressure_bytes": 256 * 1024**2,
            "pressure_outside_both_clocks": True,
            "order": "AB/BA alternating",
            "profiler": False,
            "graph_in_timing": False,
            "graph_correctness_queued_replays": 32,
        },
        "Pre-registered protocol differs",
    )
    identities = workers(report, source)
    require(set(report["widths"]) == {"1", "32"}, "Measured widths differ")
    results, count = {}, 0
    for key, block in report["widths"].items():
        width = int(key)
        fixed, signatures = {}, {}
        for phase, wanted in (
            ("warmup", list(range(-12, 0))),
            ("samples", list(range(60))),
        ):
            rows = block[phase]
            require(
                [row["number"] for row in rows] == wanted,
                "Missing/repeated sample identities",
            )
            for row in rows:
                number = row["number"]
                require(
                    type(number) is int
                    and row["order"] == list(ARMS if number % 2 == 0 else ARMS[::-1]),
                    "AB/BA (flattened ABBA) order differs",
                )
                expected = expected_signature(width, number)
                signatures[str(number)] = expected
                for arm in ARMS:
                    sample = row[arm]
                    require(
                        type(sample["width"]) is int
                        and sample["width"] == width
                        and type(sample["number"]) is int
                        and sample["number"] == number
                        and sample["consumer_signature"] == expected,
                        "Actual consumer signature/identity differs",
                    )
                    require(
                        all(
                            type(sample[clock]) in (int, float)
                            and math.isfinite(sample[clock])
                            and sample[clock] > 0
                            for clock in CLOCKS
                        ),
                        "Clock must be positive and finite",
                    )
                    pointers = sample["input_addresses"]
                    addresses(pointers, width, arm)
                    if arm in fixed:
                        require(
                            pointers == fixed[arm], "Persistent input address changed"
                        )
                    fixed[arm] = pointers
                    count += 1
        measured = block["samples"]
        timings = {
            arm: {
                clock: statistics([row[arm][clock] for row in measured])
                for clock in CLOCKS
            }
            for arm in ARMS
        }
        paired = {
            clock: statistics(
                [row["separate"][clock] - row["packet"][clock] for row in measured]
            )
            for clock in CLOCKS
        }
        for arm in ARMS:
            for clock in CLOCKS:
                compare_statistics(
                    timings[arm][clock],
                    block["timings"][arm][clock],
                    f"width{width}.{arm}.{clock}",
                )
        for clock in CLOCKS:
            compare_statistics(
                paired[clock],
                block["paired_savings_us"][clock],
                f"width{width}.paired.{clock}",
            )
        passed = all(
            timings["packet"][clock]["p50"] <= 0.8 * timings["separate"][clock]["p50"]
            and timings["packet"][clock]["p95"] <= timings["separate"][clock]["p95"]
            for clock in CLOCKS
        )
        require(
            type(block["prediction_met"]) is bool and block["prediction_met"] == passed,
            "Recorded prediction differs",
        )
        results[key] = {
            "warmup_pairs": 12,
            "measured_pairs": 60,
            "device_samples_checked": 144,
            "timings": timings,
            "paired_savings_us": paired,
            "prediction_met": passed,
            "fixed_input_addresses": fixed,
            "reconstructed_consumer_signatures": signatures,
        }
    require(count == 288, "Need all 288 device samples")
    return {
        "source_sha256": identities,
        "widths": results,
        "all_288_sample_signatures_verified": True,
        "all_statistics_recomputed": True,
        "graph_checks": "Producer reports all nine widths x32 exact; raw per-replay snapshots/signatures were not saved",
        "order": "AB/BA pairs alternate; flattened order repeats ABBA",
        "source_commit": report["source_commit"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "exit", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = {
        "schema": "host-inputs-independent-read-v1",
        "scope": "Saved A3 input preparation component; window includes payload generation and return checks, not only _prepare_inputs",
        "raw_report_sha256": digest(args.report),
        "exit_receipt_sha256": digest(args.exit),
        "reader_sha256": digest(Path(__file__)),
        "terminal": terminal(json.loads(args.exit.read_text())),
        **read(report, args.source),
        "formal_precision_gate_passed": False,
        "TPOT_measured": False,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "all_288_samples_verified": True,
                "predictions": {
                    width: item["prediction_met"]
                    for width, item in result["widths"].items()
                },
                "formal_precision_gate_passed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
