"""Audit saved three-rank reports and archive complete logs without device use."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct


ROOT = Path("/tmp/tilert-megakernel-validation-20261021")
PROBE_SHA = "1be5fa375d0d6e7a2a6dcf7f5265cf886317108398aeec940e2b1ce0de04e22f"
HEADER_SHA = "cf9e9a843c82ff77a4a4c2d6e4ff65024107cbd3c45044280c6502d65be8bde7"
MANIFEST_SHA = "fa3b5f146b82ab69374b6c0a701494a56c5ea167b900e850c1fc3bc2d5c713fe"
ALGORITHM = "AllReduceMeshAivFor91093Executor"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16_bits(value):
    word = struct.unpack("<I", struct.pack("<f", value))[0]
    return (word + 0x7FFF + ((word >> 16) & 1)) >> 16


def packed_bf16(values):
    return struct.pack("<3H", *(bf16_bits(x) for x in values))


def fixture(rows, iteration):
    scale = 2.0 ** (iteration - 2)
    epsilon = 2.0 ** -25
    inputs = [[f32(x * scale) for x in triple] for triple in
              ((1, 1, 1), (epsilon, -1, 2), (-1, epsilon, 3))]
    serial = [f32(f32(a + b) + c) for a, b, c in zip(*inputs)]
    alternate = [f32(f32(a + c) + b) for a, b, c in zip(*inputs)]
    assert serial == [0, epsilon * scale, 6 * scale]
    assert alternate == [epsilon * scale, 0, 6 * scale]
    serial_bytes = packed_bf16(serial)
    alternate_bytes = packed_bf16(alternate)
    different_bytes = sum(a != b for a, b in
                          zip(serial_bytes, alternate_bytes)) * rows * 2048
    return {
        "serial": serial, "alternate": alternate,
        "native_vs_hccl": {
            "finite": True, "bitwise_equal": False,
            "bitwise_different_bytes": different_bytes,
            "mismatch_count": rows * 4096,
            "max_absolute": epsilon * scale,
            "mean_absolute": f32(2 * epsilon * scale / 3),
        },
        "reconstructed_cpu_input_sha256": [
            hashlib.sha256(packed_bf16(values) * (rows * 2048)).hexdigest()
            for values in inputs],
    }


def audit_reports(raw):
    cases, identities = [], []
    zero = {"finite": True, "bitwise_equal": True,
            "bitwise_different_bytes": 0, "mismatch_count": 0,
            "max_absolute": 0.0, "mean_absolute": 0.0}
    for rank in range(3):
        path = raw / "communication-order-three-rank-v1" / f"rank-{rank}.json"
        report = json.loads(path.read_text())
        assert (report["rank"], report["ranks"]) == (rank, 3)
        assert report["pass"] and report["order_difference_observed"]
        assert report["library_sha256"] == PROBE_SHA
        assert report["communication_device_header_sha256"] == HEADER_SHA
        assert (report["torch"], report["torch_npu"]) == ("2.12.0+cu130", "2.12.0")
        assert report["device_mapping"] == {
            "logical_to_physical": [4, 6, 7], "logical_rank": rank,
            "direct_peer_access": {str(peer): 1 for peer in range(3) if peer != rank}}
        env = report["environment"]
        assert env["HCCL_OP_EXPANSION_MODE"] == "AIV"
        assert env["HCCL_DETERMINISTIC"] == "true"
        assert env["HCCL_ALGO"] is None and env["HCCL_ALGO_CONFIG"] is None
        assert [shape["rows"] for shape in report["shapes"]] == [1, 12]
        identities.append({"rank": rank, "report_sha256": digest(path),
                           "loaded_libraries": report["loaded_libraries"]})
        for shape in report["shapes"]:
            rows = shape["rows"]
            assert shape["pass"] and shape["hccl_matches_tree"]
            assert shape["order_difference_observed"]
            assert len(shape["checks"]) == 6
            for iteration, check in enumerate(shape["checks"]):
                expected = fixture(rows, iteration)
                assert check["iteration"] == iteration
                assert check["mode"] == ("eager" if iteration < 2 else "graph")
                for name in ("native_vs_serial", "consumer_vs_native", "hccl_vs_tree"):
                    assert check[name] == zero, (rank, rows, iteration, name)
                assert check["native_vs_hccl"] == expected["native_vs_hccl"]
                for name in ("first_three_native", "first_three_serial"):
                    assert check[name] == expected["serial"]
                for name in ("first_three_hccl", "first_three_tree"):
                    assert check[name] == expected["alternate"]
                cases.append({"rank": rank, "rows": rows, "iteration": iteration,
                              "mode": check["mode"], "pass": True, **expected})
    assert len(cases) == 36
    assert all(identity["loaded_libraries"] == identities[0]["loaded_libraries"]
               for identity in identities)
    assert len(identities[0]["loaded_libraries"]) == 9
    shmem = identities[0]["loaded_libraries"]["/shmem/libshmem.so"]
    assert shmem == "cf9b853bb7612ce1e44853b1d16851016c4a2c279bdc9bb32da0640bc411f532"
    return cases, identities


def audit_logs(raw):
    launches, selections, excerpts = [], [], []
    for path in sorted(raw.rglob("*.log")):
        current = None
        for number, line in enumerate(path.read_text().splitlines(), 1):
            selected = "SelectAlgfor91093" in line
            launched = "[INFO]" in line and "[ExecuteKernelLaunchInner]" in line
            if not (selected or launched):
                continue
            location = {"path": str(path.relative_to(raw)), "line": number}
            excerpts.append(f"{location['path']}:{number}\n{line}\n")
            if selected:
                assert f"algName [{ALGORITHM}]" in line
                current = {**location, "algorithm": ALGORITHM,
                           "process": int(re.search(r"HCCL\((\d+),", line)[1])}
                selections.append(current)
                continue
            assert current is not None
            fields = dict(re.findall(r"\b([A-Za-z]\w*)\s*\[([^\]]*)\]", line))
            for key, value in {"rankSize": "3", "dataType": "float32",
                               "reduceOp": "sum", "deterministic": "1"}.items():
                assert fields[key] == value
            assert fields["sendbuff"] == fields["recvbuff"]
            rank, tag, count = (int(fields[k]) for k in ("rank", "tag", "count"))
            rows = 1 if tag <= 6 else 12
            assert 0 <= rank < 3 and 1 <= tag <= 12 and count == rows * 6144
            process = int(re.search(r"HCCL\((\d+),", line)[1])
            assert process == current["process"]
            launches.append({**location, "rank": rank, "tag": tag, "rows": rows,
                             "iteration": (tag - 1) % 6, "process": process,
                             "fields": fields, "selection": current})
    assert len(selections) == 6 and len(launches) == 36
    assert {(x["rank"], x["tag"]) for x in launches} == {
        (rank, tag) for rank in range(3) for tag in range(1, 13)}
    assert Counter((x["rank"], x["rows"]) for x in launches) == {
        (rank, rows): 6 for rank in range(3) for rows in (1, 12)}
    assert set(Counter((x["selection"]["path"], x["selection"]["line"])
                       for x in launches).values()) == {6}
    return {"selections": selections, "launches": launches}, "\n".join(excerpts)


def archive_raw(raw, output):
    records = []
    for path in sorted(raw.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(raw)
        target = output / "raw" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original_sha, original_bytes = digest(path), path.stat().st_size
        if path.suffix == ".log":
            target = target.with_suffix(target.suffix + ".gz")
            with path.open("rb") as source, target.open("wb") as destination:
                with gzip.GzipFile(filename="", mode="wb", mtime=0,
                                   fileobj=destination) as compressed:
                    shutil.copyfileobj(source, compressed)
            with gzip.open(target, "rb") as compressed:
                restored = compressed.read()
            assert len(restored) == original_bytes
            assert hashlib.sha256(restored).hexdigest() == original_sha
        else:
            shutil.copy2(path, target)
            assert digest(target) == original_sha
        records.append({"source": str(relative), "original_bytes": original_bytes,
                        "original_sha256": original_sha,
                        "archive": str(target.relative_to(output)),
                        "archive_bytes": target.stat().st_size,
                        "archive_sha256": digest(target),
                        "compression": "gzip" if path.suffix == ".log" else None})
        assert digest(path) == original_sha
    assert len(records) == 20 and sum(x["original_bytes"] for x in records) == 81185055
    return records


def audit_sources(source, c2, output):
    assert digest(source / "manifest.json") == MANIFEST_SHA
    pins = dict(line.split("  ", 1)[::-1]
                for line in (source / "frozen-inputs.sha256").read_text().splitlines())
    assert len(pins) == 12
    for name, expected in pins.items():
        assert digest(source / name) == expected, name
    assert pins["communication_probe.so"] == PROBE_SHA
    assert pins["communication_device.h"] == HEADER_SHA
    comparison = {}
    for name in ("communication.h", "communication_device.h", "communication_host.cpp"):
        assert ((source / name).read_bytes()
                == (c2 / "communication" / name).read_bytes())
        comparison[name] = digest(source / name)
    target = output / "source"
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix != ".so":
            shutil.copy2(path, target / path.name)
    for name in ("source/kernel.cpp", "manifest.json"):
        destination = output / "c2-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c2 / name, destination)
    return {"frozen_manifest_sha256": MANIFEST_SHA, "frozen_files": pins,
            "c2_identical_communication_files": comparison,
            "c2_kernel_sha256": digest(c2 / "source/kernel.cpp"),
            "c2_manifest_sha256": digest(c2 / "manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=ROOT / "communication-order-three-rank-v1-raw")
    parser.add_argument("--source", type=Path, default=ROOT / "communication-order-v2")
    parser.add_argument("--c2", type=Path, default=ROOT / "layer-tp-chain-build-v1")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, identities = audit_reports(args.raw)
    logs, excerpt = audit_logs(args.raw)
    source = audit_sources(args.source, args.c2, args.output)
    raw = archive_raw(args.raw, args.output)
    for name in ("communication-order-three-rank-v1.command.json",
                 "communication-order-three-rank-v1.launcher.log",
                 "communication-order-three-rank-v1.admission.jsonl",
                 "run-communication-order-three-rank-v1.py"):
        destination = args.output / "launch" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    save(args.output / "case-audit.json", cases)
    save(args.output / "launch-audit.json", logs)
    save(args.output / "source-audit.json", source)
    save(args.output / "raw-manifest.json", raw)
    (args.output / "hccl-selection-and-launch.log").write_text(excerpt)
    summary = {
        "audit_pass": True, "hccl_bitwise_equivalence": False,
        "scope": ("First reduction of the unchanged communication probe; "
                  "synthetic BF16 cancellation inputs"),
        "ranks": 3, "physical_dies": [4, 6, 7], "rows": [1, 12], "blocks": 24,
        "cases": len(cases), "eager_cases": 12, "graph_cases": 24,
        "native_vs_serial_exact": 36, "immediate_consumer_vs_native_exact": 36,
        "graph_consumer_vs_native_exact": 24,
        "hccl_vs_alternative_grouping_exact": 36,
        "native_vs_hccl_different": 36, "different_elements_per_row": 4096,
        "max_absolute_range": [2.0 ** -27, 2.0 ** -22],
        "hccl_arithmetic": "BF16.float -> installed HCCL FP32 SUM -> BF16",
        "algorithm": ALGORITHM, "selector_lines": 6, "info_launch_lines": 36,
        "deterministic": True, "hccl_algo_override": None,
        "probe_library_sha256": PROBE_SHA, "communication_header_sha256": HEADER_SHA,
        "identity_by_rank": identities,
        "raw_files": 20, "raw_bytes": sum(x["original_bytes"] for x in raw),
        "gzip_files": sum(x["compression"] == "gzip" for x in raw),
        "archived_bytes": sum(x["archive_bytes"] for x in raw),
        "raw_manifest_sha256": digest(args.output / "raw-manifest.json"),
        "full_selector_launch_excerpt_sha256": digest(
            args.output / "hccl-selection-and-launch.log"),
        "limits": [
            ("Full device tensor dumps were not retained; all recorded "
             "raw-byte metrics and first-three values are audited."),
            ("Input hashes in case-audit are independently reconstructed "
             "CPU fixture bytes, not measured device hashes."),
            ("The fixture identifies an incompatible accumulation result, "
             "not a unique complete HCCL tree."),
            ("The probe executes a projection and second reduction, but "
             "this diagnostic compares only its first reduction."),
            ("This is neither a TP16 result nor a whole-layer "
             "or performance measurement."),
        ],
    }
    save(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "audit_pass", "cases", "selector_lines", "info_launch_lines", "raw_files",
        "raw_bytes", "gzip_files", "archived_bytes")}, indent=2))


if __name__ == "__main__":
    main()
