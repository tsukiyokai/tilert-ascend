"""CPU-only row-13 diagnostics for the frozen full-fixed-v3 native capture.

This reader accepts the failed numerical run as diagnostic evidence, never as
a precision receipt. It imports neither the capture entry nor HF/vLLM code.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
from torch.nn import functional

INITIAL = 8192
WIDTH = 32
ROW = 13
LAYERS = 78
FROZEN_SHA256 = {
    "summary.json": "9c8b7eb72ee618a4e2e24ae87ce4bedd90e88c66d8c2b197d77ca04bf5f55e03",
    "rank-0.json": "93ad8a7e873bae615e247c75fe74c06c6fa7b9d4775c9896f912c52c35d61265",
    "serial-13.pt": "c68e614337b810b07e957773e63b4dce6e05b6d7cd7ab0fa1258e4c486dd94d8",
    "wide.pt": "dd81802b7f6b64bc44a8fe0114fae2eebdad1d0db27ffe319efbb951ce607a43",
}
TEACHER_SHA256 = "82d011962a288b30b777fc4964d5c96c6b9784cb13b97f509322f603b263b655"
SOURCE_BINDINGS = {
    "experiments/cann-precision/run.py": "c832fb86cc58b5f9b5092a4e4a2d6e4fc463c554497172f8f5bba6d25aa963dc",
    "experiments/graph-evidence/producer_probe.py": "a817d0b5dbe2a969b483b0b6916625311497fd6731bf134789aee533b7921f67",
    "tests/glm52/run_cann_verify.py": "ef5c59502d8d7ca8e07a1c6ffa1fa168f802fb3af61c6635be6c190625ba8ac6",
    "tests/glm52/run_native_decode.py": "6f4647258d49f2056810e7dc421bc64ffcfb9f84a99ca6d0acdbcad661a9ed79",
    "tests/glm52/run_cann_model.py": "c5aa6a009f3a8b7d0034ca78b658b600c94f45c27e4d5ef8507e0999eeb9e03e",
    "tilert/models/glm_5_2_ascend/cann_trace.py": "4e51c1c0c6278b412f230c3167a536cf004f80faf551c1b72b88ecc158869776",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path, expected):
    actual = digest(path)
    require(actual == expected, f"SHA256 mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def receipts(directory, teacher_path):
    paths = {name: directory / name for name in ("summary.json", "rank-0.json")}
    identities = {name: bind(path, FROZEN_SHA256[name]) for name, path in paths.items()}
    summary, report = (json.loads(paths[name].read_text()) for name in paths)
    identities["teacher"] = bind(teacher_path, TEACHER_SHA256)
    teacher = json.loads(teacher_path.read_text())
    tokens = teacher["teacher_tokens"]
    require(
        len(tokens) == WIDTH
        and all(type(token) is int and 0 <= token < 154880 for token in tokens)
        and tokens[0] == teacher["first_token_id"]
        and tokens[1:] == teacher["expected_next_tokens"][:-1],
        "Incomplete or inconsistent teacher trajectory",
    )
    require(
        report["rank"] == 0
        and report["world_size"] == 16
        and report["capture_complete"] is True
        and report["mode"] == "strict_hf"
        and report["status"] == "fail"
        and report["source_commit"] == "3fcfd8b4904bc609888150d83d4b9137f4bcae63",
        "This reader is restricted to the frozen completed failed rank0 run",
    )
    receipt = next(item for item in summary["ranks"] if item["rank"] == 0)
    require(
        receipt["report_sha256"] == identities["rank-0.json"]["sha256"]
        and summary["all_rank_state_agreement"] is True,
        "Summary does not bind the rank report or agreed execution states",
    )
    require(
        report["teacher"]["sha256"] == TEACHER_SHA256
        and report["teacher"]["teacher_tokens"] == tokens
        and report["inputs"]["initial_position"] == INITIAL
        and report["inputs"]["first_token_id"] == tokens[0],
        "Report teacher/prefix identity differs",
    )
    for name, expected in SOURCE_BINDINGS.items():
        require(
            report["source_sha256"][name] == expected,
            "Captured observer/comparator source differs: " + name,
        )
    for name in ("serial-13.pt", "wide.pt"):
        require(
            report["trace_sha256"][name] == FROZEN_SHA256[name],
            "Report raw SHA differs: " + name,
        )
        identities[name] = bind(directory / "rank-0" / name, FROZEN_SHA256[name])
    return report, tokens, identities


def tensor(value, shape, dtypes, label):
    require(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and tuple(value.shape) == shape
        and value.dtype in dtypes,
        "Unexpected CPU tensor layout: " + label,
    )


def validate_frame(frame, tokens, position):
    width = len(tokens)
    require(
        frame["token"] == tokens
        and all(type(token) is int for token in frame["token"])
        and type(frame["position"]) is int
        and frame["position"] == position
        and type(frame["committed_position"]) is int
        and frame["committed_position"] == position + width,
        "Frame token/position identity differs",
    )
    for name, size, dtype in (
        ("token_ids", None, torch.int64),
        ("logits", 154880, torch.float32),
        ("last_hidden", 6144, torch.bfloat16),
    ):
        shape = (width,) if size is None else (width, size)
        tensor(frame[name], shape, (dtype,), name)
    for field, count, size, dtypes in (
        ("hidden_states", 79, 6144, (torch.bfloat16,)),
        ("layer_logits", 78, 154880, (torch.float32,)),
        ("indices", 78, 2048, (torch.int32, torch.int64)),
        ("routes", 78, 8, (torch.int32, torch.int64)),
    ):
        require(len(frame[field]) == count, "Incomplete " + field)
        for layer, value in enumerate(frame[field]):
            if field == "routes" and layer < 3:
                require(value is None, "Dense layer has routed choices")
            else:
                tensor(value, (width, size), dtypes, f"{field}.{layer}")
    require(len(frame["cache_rows"]) == LAYERS, "Incomplete new-KV rows")
    for layer, cache in enumerate(frame["cache_rows"]):
        require(
            set(cache) == {"latent", "rope", "index", "index_scale"},
            "Unexpected cache fields",
        )
        for field, size in (("latent", 512), ("rope", 64)):
            tensor(cache[field], (width, size), (torch.bfloat16,), field)
        full = layer < 3 or (layer - 2) % 4 == 0
        quantized = full and 6 <= layer <= 66
        if full:
            dtype = torch.int8 if quantized else torch.bfloat16
            tensor(cache["index"], (width, 128), (dtype,), "index")
        else:
            require(cache["index"] is None, "Shared index layer writes index KV")
        scale = cache["index_scale"]
        if quantized:
            require(
                isinstance(scale, torch.Tensor)
                and tuple(scale.shape) in ((width,), (width, 1)),
                "Index scale must store exactly one scalar per row",
            )
            tensor(scale.reshape(width), (width,), (torch.float16,), "index_scale")
        else:
            require(scale is None, "Unexpected index scale")


def compare(actual, expected, *, exact=False, logits=False):
    """Same FP32 metrics/thresholds as frozen run_native_decode.compare_tensor.

    Actual is M32; expected is M1, preserving the original NRMSE denominator.
    Raw byte equality is reported separately from numeric exact equality.
    """
    if actual is None or expected is None:
        same = actual is None and expected is None
        return {"bit_equal": same, "original_passed": same, "absent": same}
    require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "Comparison layout differs",
    )
    actual, expected = actual.contiguous(), expected.contiguous()
    byte_changes = actual.reshape(-1).view(torch.uint8).reshape(
        -1, actual.element_size()
    ) != (expected.reshape(-1).view(torch.uint8).reshape(-1, expected.element_size()))
    changes = byte_changes.any(-1)
    changed = changes.nonzero().flatten()
    result = {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "bit_equal": not bool(changed.numel()),
        "unequal_elements": int(changed.numel()),
        "first_flat_index": int(changed[0]) if changed.numel() else None,
        "original_exact_required": exact,
        "original_passed": False,
    }
    if changed.numel():
        index = int(changed[0])
        result["first_wide_value"] = str(actual.flatten()[index].item())
        result["first_serial_value"] = str(expected.flatten()[index].item())
    left, right = actual.float(), expected.float()
    result["finite"] = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not result["finite"]:
        return result
    norm = right.norm().item()
    error = left - right
    nrmse = error.norm().item() / max(norm, 1e-12)
    cosine = (
        functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()
        if norm
        else float(torch.count_nonzero(left) == 0)
    )
    max_abs = error.abs().max().item()
    passed = (
        torch.equal(actual, expected) if exact else nrmse <= 0.01 and cosine >= 0.999
    )
    if logits and not exact:
        passed = (
            passed
            and max_abs <= 0.1
            and torch.equal(actual.argmax(-1), expected.argmax(-1))
        )
    result.update(
        nrmse=nrmse, cosine=cosine, max_abs=max_abs, original_passed=bool(passed)
    )
    return result


def choices(actual, expected, field, position):
    result = compare(actual, expected, exact=True)
    if actual is None:
        return result
    left, right = actual.flatten().long(), expected.flatten().long()
    if field == "indices":
        valid_range = all(
            bool(torch.all((row == -1) | ((row >= 0) & (row <= position))))
            for row in (left, right)
        )
        left, right = left[left >= 0], right[right >= 0]
        count = min(position + 1, 2048)
    else:
        valid_range = all(
            bool(torch.all((row >= 0) & (row < 256))) for row in (left, right)
        )
        count = 8
    selection_valid = valid_range and all(
        row.numel() == count and row.unique().numel() == count for row in (left, right)
    )
    left_set, right_set = set(left.tolist()), set(right.tolist())
    same_set = left_set == right_set
    result.update(
        selection_valid=selection_valid,
        sets_equal=same_set,
        order_equal=torch.equal(left, right),
        overlap=len(left_set & right_set),
        wide_only=sorted(left_set - right_set),
        serial_only=sorted(right_set - left_set),
        original_passed=selection_valid and same_set,
    )
    return result


def first_boundary(embedding, layers, key):
    if not embedding[key]:
        return {"boundary": "embedding", "fields": ["hidden_states.0"]}
    for item in layers:
        changed = [
            field for field, metrics in item["fields"].items() if not metrics[key]
        ]
        if changed:
            layer = item["layer"]
            return {
                "layer": layer,
                "fields": changed,
                "input_hidden": f"hidden_states.{layer}",
                "output_hidden": f"hidden_states.{layer + 1}",
                "internal_field_order": "not observed",
                "last_layer_output_includes_final_norm": layer == 77,
            }
    return None


def diagnose(serial, wide):
    def row(frame, field, layer, index):
        value = frame[field][layer]
        return None if value is None else value[index : index + 1]

    embedding = compare(
        wide["hidden_states"][0][ROW : ROW + 1], serial["hidden_states"][0]
    )
    layers = []
    for layer in range(LAYERS):
        fields = {}
        for field in ("latent", "rope", "index", "index_scale"):
            left, right = (
                frame["cache_rows"][layer][field] for frame in (wide, serial)
            )
            if left is not None:
                left, right = left[ROW : ROW + 1], right[:1]
                if field == "index_scale":
                    left, right = left.reshape(1), right.reshape(1)
            exact = field == "index_scale" or (
                left is not None and left.dtype == torch.int8
            )
            fields["cache." + field] = compare(left, right, exact=exact)
        for field in ("indices", "routes"):
            fields[field] = choices(
                row(wide, field, layer, ROW),
                row(serial, field, layer, 0),
                field,
                INITIAL + ROW,
            )
        fields["output_hidden"] = compare(
            row(wide, "hidden_states", layer + 1, ROW),
            row(serial, "hidden_states", layer + 1, 0),
        )
        layers.append(
            {
                "layer": layer,
                "fields": fields,
                "diagnostic_layer_logits": compare(
                    row(wide, "layer_logits", layer, ROW),
                    row(serial, "layer_logits", layer, 0),
                    logits=True,
                ),
            }
        )
    return {
        "embedding": embedding,
        "layers": layers,
        "first_bit_difference": first_boundary(embedding, layers, "bit_equal"),
        "first_original_threshold_failure": first_boundary(
            embedding, layers, "original_passed"
        ),
        "final_logits": compare(
            wide["logits"][ROW : ROW + 1], serial["logits"], logits=True
        ),
        "last_hidden": compare(
            wide["last_hidden"][ROW : ROW + 1], serial["last_hidden"]
        ),
        "returned_ids": compare(
            wide["token_ids"][ROW : ROW + 1], serial["token_ids"], exact=True
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report, tokens, identities = receipts(args.capture_dir, args.teacher)
    frames = [
        torch.load(
            args.capture_dir / "rank-0" / name,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        for name in ("serial-13.pt", "wide.pt")
    ]
    serial, wide = frames
    validate_frame(serial, [tokens[ROW]], INITIAL + ROW)
    validate_frame(wide, tokens, INITIAL)
    expected_inputs = {
        "input_ids": [tokens[ROW]],
        "positions": [INITIAL + ROW],
        "slots": [INITIAL + ROW],
        "query_cpu": [0, 1],
        "seq_lens": [INITIAL + ROW + 1],
    }
    require(
        all(
            serial["model_inputs"][name].tolist() == values
            for name, values in expected_inputs.items()
        ),
        "Actual serial model inputs differ",
    )
    result = {
        "schema": "cann-native-width-first-difference-v1",
        "rank": 0,
        "row_zero_based": ROW,
        "token": tokens[ROW],
        "position": INITIAL + ROW,
        "committed_position_for_row": INITIAL + ROW + 1,
        "whole_wide_committed_position": wide["committed_position"],
        "scope": "Selected row only; not the earliest bit-different row of all 32",
        "wide_metadata_evidence": "tokens/positions checked by frozen observer; raw slots not saved",
        "hidden_semantics": "h0 embedding; h1..77 preceding merged residual; h78 final normalized",
        "earliest_boundary_order": "embedding, then layer number; no internal ordering inferred",
        "diagnostic_logits_excluded_from_boundary_order": True,
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "identities": identities,
        "reader_sha256": digest(Path(__file__)),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "inputs": report["inputs"],
        "captured_source_sha256": report["source_sha256"],
        **diagnose(serial, wide),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "row_zero_based",
                    "token",
                    "position",
                    "first_bit_difference",
                    "first_original_threshold_failure",
                    "formal_precision_gate_passed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
