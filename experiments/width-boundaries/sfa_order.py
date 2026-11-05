"""Replay three actual index orders at the fixed rank12 M1 SFA boundary."""

import argparse
import ast
import hashlib
import json
import os
import traceback
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch

RANK, ROW, END = 12, 13, 8206
PREFIX = "layer68.sfa.call0."
SUMMARY_SHA = "a0d9ab0eb6317bbc41d2fb18c138cc96e76c051b6e7fe45c926671b945762df3"
REPORT_SHA = "f2e1ddea3571d7c45c36f953c123ba6768ce18fe52875a648bd1146a07b11b8d"
REFERENCE_SHA = "1f3b3d62119d4e887dff91fbabefd397391d787a04414cbe7601e2a290af76fe"
DEVICE_SHA = "b7062ab6757b52a1ce95844dd2c190b7154e5dbb11cc1b31545a13c1628abbb8"
LIBRARY_SHA = "ec36626f56ae4c75b5d2fe824d85d55d09c070eeeaebc31eb4eb7b9a71be7cc5"
RAW_SHA = {
    "original-serial.pt": "732c1af2dea25ead39a82acf82ea437990851f5d5c27850485cf40b40dfc6b06",
    "original-wide.pt": "006dfa411fde95530badf4c1aeee46541be00abff359b83626761102c97684cb",
    "sorted-serial.pt": "8923d44a53d028801763355faa46ce53eddadb265326ebaf61766ab7fb2ee008",
}
SOURCES = {
    "/width-boundaries-v1/run.py": "7e5028edb20bd552b8a0861eb8ff7cca049ac0793126344afd262de8a21ce216",
    "/tilert/models/glm_5_2_ascend/cann_model.py": "0c59db724306ec82db21f2655879874193f69dd1425872180511a6e4d434994d",
    "/vllm_ascend/device/device_op.py": DEVICE_SHA,
}
OBSERVED_SOURCES = {
    "/observe.py": "0c2f0ccff671b19a9625759c0cb1e6057dbad549512740fa4a040dff9cd49d5e",
    "/routes.py": "8a0a3d8b863145a86bd2f75a04a114b07f69db1f3565317852aca6ad7bbed334",
    "/tests/glm52/run_cann_model.py": "c5aa6a009f3a8b7d0034ca78b658b600c94f45c27e4d5ef8507e0999eeb9e03e",
    "/vllm_ascend/attention/sfa_v1.py": "b5ad39a05b1323ac5048962f137969158298e4f637b5bd3e401cb4c4248ecfa0",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path, expected):
    require(digest(path) == expected, "Frozen bytes differ: " + str(path))
    return {
        "path": str(path.resolve()),
        "sha256": expected,
        "bytes": path.stat().st_size,
    }


def sources(actual, expected):
    for suffix, sha in expected.items():
        matches = [value for path, value in actual.items() if path.endswith(suffix)]
        require(
            matches and all(value == sha for value in matches),
            "Source differs: " + suffix,
        )


def raw_sha(value):
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def exact(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and raw_sha(left) == raw_sha(right)
    )


def length(frame, name):
    """Keep tensor metadata distinct from scalar sequences captured as slots."""
    key = PREFIX + name
    observed = frame["observed"]
    tensor = observed["tensors"].get(key)
    if isinstance(tensor, torch.Tensor):
        require(
            tensor.dtype in (torch.int32, torch.int64), "Length tensor is not integer"
        )
        return {
            "kind": "tensor",
            "value": tensor.clone(),
            "original_device": observed["metadata"]["tensor_layouts"][key]["device"],
        }
    slots = observed["metadata"]["scalar_values"]
    selected = {
        int(tag[len(key) + 1 :]): value
        for tag, value in slots.items()
        if tag.startswith(key + ".")
    }
    require(
        selected
        and set(selected) == set(range(len(selected)))
        and all(type(value) is int for value in selected.values()),
        "Length is neither a tensor nor an integer sequence",
    )
    return {
        "kind": "python_list",
        "value": [selected[index] for index in range(len(selected))],
        "container_type_recorded": False,
        "note": "Observer records scalar sequence slots; replay preserves these as a Python list",
    }


def length_values(descriptor):
    value = descriptor["value"]
    return value.flatten().tolist() if descriptor["kind"] == "tensor" else value


def length_identity(descriptor):
    result = {key: value for key, value in descriptor.items() if key != "value"}
    result["values"] = length_values(descriptor)
    if descriptor["kind"] == "tensor":
        result.update(
            dtype=str(descriptor["value"].dtype),
            shape=list(descriptor["value"].shape),
            sha256=raw_sha(descriptor["value"]),
        )
    return result


def runtime_length(descriptor, device):
    if descriptor["kind"] == "python_list":
        return list(descriptor["value"])
    value = descriptor["value"]
    return (
        value.to(device)
        if descriptor["original_device"].startswith("npu:")
        else value.clone()
    )


def logical_cache(value, table):
    require(
        value.ndim == 4 and value.shape[1:3] == (128, 1),
        "Unexpected paged cache layout",
    )
    positions = torch.arange(END)
    pages = table[0].long()[positions // 128]
    require(
        bool(((pages >= 0) & (pages < value.shape[0])).all()),
        "Page table exceeds cache",
    )
    return value.reshape(-1, value.shape[-1])[pages * 128 + positions % 128]


def extract(frame, wide=False):
    tensors = frame["observed"]["tensors"]
    row = ROW if wide else 0
    width = 32 if wide else 1
    start = 8192 if wide else END - 1
    tokens = frame["tokens"] if wide else [frame["token"]]
    require(
        len(tokens) == width
        and tokens[row] == 24187
        and frame["position"] == start
        and frame["committed_position"] == start + width,
        "Teacher row/position identity differs",
    )
    expected_inputs = {
        "input_ids": tokens,
        "positions": list(range(start, start + width)),
        "slots": list(range(start, start + width)),
        "query_cpu": [0, width],
        "seq_lens": [start + width],
    }
    require(
        all(
            frame["model_inputs"][name].tolist() == value
            for name, value in expected_inputs.items()
        ),
        "Actual model input identity differs",
    )
    values = {
        name: tensors[PREFIX + name].clone()
        for name in (
            "ql_nope",
            "q_pe",
            "block_table",
            "kv_cache.0",
            "kv_cache.1",
            "topk_indices",
        )
    }
    for name, size in (("ql_nope", 512), ("q_pe", 64)):
        require(
            values[name].shape == (width, 4, size)
            and values[name].dtype == torch.bfloat16,
            "Actual query layout differs",
        )
        values[name] = values[name][row : row + 1].clone()
    for number, size in ((0, 512), (1, 64)):
        cache = values[f"kv_cache.{number}"]
        require(
            cache.shape == (73, 128, 1, size) and cache.dtype == torch.bfloat16,
            "This experiment requires the frozen BF16 two-cache path",
        )
    require(
        values["block_table"].shape == (1, 73)
        and values["block_table"].dtype in (torch.int32, torch.int64),
        "Block table layout differs",
    )
    indices = values["topk_indices"].reshape(width, 1, 2048)[row : row + 1].clone()
    require(
        indices.dtype in (torch.int32, torch.int64)
        and indices.unique().numel() == 2048
        and bool(((indices >= 0) & (indices < END)).all()),
        "Invalid row13 causal selection",
    )
    values["topk_indices"] = indices
    lengths = {
        name: length(frame, name)
        for name in ("actual_seq_lengths_query", "actual_seq_lengths_key")
    }
    require(
        length_values(lengths["actual_seq_lengths_query"]) == [width]
        and length_values(lengths["actual_seq_lengths_key"]) == [8224 if wide else END],
        "Actual SFA lengths differ",
    )
    calls = frame["observed"]["metadata"]["sfa_calls"]
    require(
        len(calls) == 1 and calls[0]["scale"] == 0.0625,
        "Actual SFA scale/call count differs",
    )
    output = tensors[PREFIX + "out"]
    require(
        output.shape == (width, 4, 512) and output.dtype == torch.bfloat16,
        "SFA output layout differs",
    )
    return {
        "inputs": values,
        "lengths": lengths,
        "scale": calls[0]["scale"],
        "expected": output[row : row + 1].clone(),
    }


def admit(capture):
    identities = {
        "summary": bind(capture / "summary.json", SUMMARY_SHA),
        "rank12": bind(capture / "rank-12.json", REPORT_SHA),
        "raw": {},
    }
    summary = json.loads((capture / "summary.json").read_text())
    rank = json.loads((capture / "rank-12.json").read_text())
    require(
        summary["status"] == "captured"
        and summary["all_rank_global_states_agree"] is True
        and next(item for item in summary["ranks"] if item["rank"] == RANK)[
            "report_sha256"
        ]
        == REPORT_SHA,
        "Summary does not bind the completed rank12 capture",
    )
    require(
        rank["rank"] == RANK
        and rank["status"] == "captured"
        and rank["reference_sha256"] == REFERENCE_SHA
        and rank["teacher_sha256"]
        == "82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655"
        and rank["checks"]
        and all(check["passed"] is True for check in rank["checks"]),
        "Original execution/reference control did not pass",
    )
    sources(rank["source_sha256"], SOURCES)
    captures, frames = {}, {}
    for name, sha in RAW_SHA.items():
        require(rank["trace_sha256"][name] == sha, "Rank report raw identity differs")
        path = capture / "rank-12" / name
        identities["raw"][name] = bind(path, sha)
        frame = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        sources(frame["observed"]["metadata"]["source_sha256"], OBSERVED_SOURCES)
        require(
            all(frame["observed"]["coverage"].values()), "Observation coverage failed"
        )
        captures[name] = extract(frame, wide=name == "original-wide.pt")
        frames[name] = frame
    original, wide, sorted_control = (captures[name] for name in RAW_SHA)
    for other in (wide, sorted_control):
        for name in ("ql_nope", "q_pe"):
            require(
                exact(original["inputs"][name], other["inputs"][name]),
                "Query differs across controls: " + name,
            )
        for number in (0, 1):
            a = logical_cache(
                original["inputs"][f"kv_cache.{number}"],
                original["inputs"]["block_table"],
            )
            b = logical_cache(
                other["inputs"][f"kv_cache.{number}"], other["inputs"]["block_table"]
            )
            require(exact(a, b), "Effective KV differs across controls")
    require(
        all(
            length_identity(original["lengths"][name])
            == length_identity(sorted_control["lengths"][name])
            for name in original["lengths"]
        ),
        "M1 sorted control length type/values differ",
    )
    first = original["inputs"]["topk_indices"]
    second = wide["inputs"]["topk_indices"]
    require(
        first.dtype == second.dtype
        and exact(first.sort(-1).values, second.sort(-1).values),
        "M32 changes the selected set",
    )
    swapped = (first != second).flatten().nonzero().flatten().tolist()
    require(
        len(swapped) == 2
        and first.flatten()[swapped[0]] == second.flatten()[swapped[1]]
        and first.flatten()[swapped[1]] == second.flatten()[swapped[0]],
        "Expected exactly the two actual index swaps",
    )
    require(
        exact(first.sort(-1).values, sorted_control["inputs"]["topk_indices"])
        and exact(
            first,
            frames["sorted-serial.pt"]["observed"]["tensors"][
                "intervention.original_indices"
            ],
        ),
        "Sorted serial changes more than the input index order",
    )
    return {
        "inputs": original["inputs"],
        "lengths": original["lengths"],
        "scale": original["scale"],
        "orders": {
            "original": first,
            "m32_order": second,
            "sorted": first.sort(-1).values,
        },
        "expected": {
            "original": original["expected"],
            "m32_order": wide["expected"],
            "sorted": sorted_control["expected"],
        },
        "swapped_flat_positions": swapped,
        "m32_lengths_not_replayed": {
            key: length_identity(value) for key, value in wide["lengths"].items()
        },
        "capture_inputs": rank["inputs"],
        "checkpoint_sha256": rank["checkpoint_sha256"],
        "reference_sha256": REFERENCE_SHA,
        "identities": identities,
    }


def method(path):
    source = path.read_text()
    matches = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_sparse_flash_attention_process"
    ]
    require(len(matches) == 1, "Expected one frozen DeviceOperator method")
    node = matches[0]
    segment_sha = hashlib.sha256(
        ast.get_source_segment(source, node).encode()
    ).hexdigest()
    node.decorator_list = []
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102 - exact SHA-pinned AST, unchanged arithmetic.
    return namespace[node.name], segment_sha


def mapped_libraries(library):
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        name = line.split()[-1]
        if (
            name.startswith("/")
            and ".so" in name
            and any(
                word in name.lower()
                for word in ("ascend", "torch_npu", "vllm", "custom")
            )
            and Path(name).is_file()
        ):
            paths.add(Path(name).resolve())
    require(
        library.resolve() in paths,
        "Requested vllm_ascend_C library is not actually mapped",
    )
    return {str(path): digest(path) for path in sorted(paths)}


def differences(actual, expected):
    require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "Output layout differs",
    )
    changed = actual.contiguous().view(torch.uint8).reshape(
        -1, actual.element_size()
    ) != expected.contiguous().view(torch.uint8).reshape(-1, expected.element_size())
    flat = changed.any(-1).nonzero().flatten()
    return {
        "bit_exact": not bool(flat.numel()),
        "unequal_elements": int(flat.numel()),
        "first_flat_index": int(flat[0]) if flat.numel() else None,
        "max_abs": float((actual.float() - expected.float()).abs().max()),
        "sha256": raw_sha(actual),
        "expected_sha256": raw_sha(expected),
    }


def replay(fixture, function, device):
    inputs = {
        name: value.to(device)
        for name, value in fixture["inputs"].items()
        if name != "topk_indices"
    }
    orders = {name: value.to(device) for name, value in fixture["orders"].items()}
    lengths = {
        name: runtime_length(value, device)
        for name, value in fixture["lengths"].items()
    }
    metadata, impl = (
        SimpleNamespace(block_table=inputs["block_table"]),
        SimpleNamespace(scale=fixture["scale"]),
    )

    def fingerprints():
        return {
            **{name: raw_sha(value) for name, value in inputs.items()},
            **{"order." + name: raw_sha(value) for name, value in orders.items()},
            **{
                "length." + name: raw_sha(value)
                if isinstance(value, torch.Tensor)
                else list(value)
                for name, value in lengths.items()
            },
        }

    before, outputs, records = fingerprints(), {}, {}
    with torch.inference_mode():
        for name, indices in orders.items():
            values, hashes, matches = [], [], []
            for _ in range(32):
                output = function(
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
                torch.npu.synchronize()
                value = output.detach().cpu().clone()
                require(
                    value.shape == (1, 4, 512)
                    and value.dtype == torch.bfloat16
                    and bool(torch.isfinite(value).all()),
                    "Nonfinite or malformed SFA output",
                )
                values.append(value)
                hashes.append(raw_sha(value))
                matches.append(exact(value, fixture["expected"][name]))
            outputs[name] = torch.stack(values)
            records[name] = {
                "repeats": 32,
                "all_raw_sha256": hashes,
                "stable": len(set(hashes)) == 1,
                "all_expected_bit_exact": all(matches),
                "per_call_expected_bit_exact": matches,
                "vs_frozen_output": differences(values[0], fixture["expected"][name]),
                "inputs_unchanged_after_arm": fingerprints() == before,
            }
    unchanged = fingerprints() == before
    controlled = unchanged and all(
        row["stable"] and row["inputs_unchanged_after_arm"] for row in records.values()
    )
    controlled = controlled and all(
        records[name]["all_expected_bit_exact"] for name in ("original", "sorted")
    )
    return outputs, {
        "arms": records,
        "all_inputs_unchanged": unchanged,
        "controls_and_repeatability_passed": controlled,
        "m32_order_reproduces_frozen_wide": records["m32_order"][
            "all_expected_bit_exact"
        ],
        "order_sufficiency_met": controlled
        and records["m32_order"]["all_expected_bit_exact"],
        "frozen_original_vs_wide": differences(
            fixture["expected"]["original"], fixture["expected"]["m32_order"]
        ),
        "fixed_M1_original_vs_m32_order": differences(
            outputs["original"][0], outputs["m32_order"][0]
        ),
        "input_fingerprints": before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "device-source", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "sfa-index-order-replay-v1",
        "status": "preparing",
        "npu_executed": False,
        "scope": "Rank12 layer68 fixed M1 shape, three actual orders, 32 complete SFA calls each",
        "reader_sha256": digest(Path(__file__)),
        "formal_precision_gate_passed": False,
        "precision_verified": False,
        "performance_measured": False,
    }
    try:
        torch.set_num_threads(2)
        fixture = admit(args.capture)
        report.update(
            identities=fixture["identities"],
            reference_sha256=REFERENCE_SHA,
            device_source=bind(args.device_source, DEVICE_SHA),
            library=bind(args.library, LIBRARY_SHA),
            metadata_lengths={
                name: length_identity(value)
                for name, value in fixture["lengths"].items()
            },
            m32_lengths_not_replayed=fixture["m32_lengths_not_replayed"],
            swapped_flat_positions=fixture["swapped_flat_positions"],
        )
        require(args.library.stat().st_size == 934816, "Frozen library size differs")
        torch.save(fixture, args.output / "inputs.pt")
        report["inputs_file_sha256"] = digest(args.output / "inputs.pt")
        function, method_sha = method(args.device_source)
        report["unmodified_method_source_sha256"] = method_sha
        import torch_npu

        torch_npu.npu.set_device(0)
        torch.npu.matmul.allow_hf32 = False
        name = torch.npu.get_device_name(0)
        require(any(sku in name for sku in ("9382", "9392")), "Actual A3 required")
        torch.ops.load_library(str(args.library.resolve()))
        report.update(
            npu_executed=True,
            device=name,
            logical_device=0,
            packages={"torch": torch.__version__, "torch-npu": version("torch-npu")},
        )
        outputs, result = replay(fixture, function, "npu:0")
        report.update(result)
        torch.save(outputs, args.output / "outputs.pt")
        report["outputs_file_sha256"] = digest(args.output / "outputs.pt")
        report["mapped_libraries_sha256"] = mapped_libraries(args.library)
        require(
            report["mapped_libraries_sha256"][str(args.library.resolve())]
            == LIBRARY_SHA,
            "Actually mapped custom library differs",
        )
        report["status"] = (
            "replayed" if result["controls_and_repeatability_passed"] else "fail"
        )
    except Exception:  # noqa: BLE001 - persist failure receipt and return exit1 below.
        report.update(status="fail", error=traceback.format_exc())
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "controls_and_repeatability_passed",
                    "m32_order_reproduces_frozen_wide",
                    "order_sufficiency_met",
                )
            }
        )
    )
    return 0 if report["status"] == "replayed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
