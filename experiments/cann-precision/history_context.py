"""Read only layer68, rows0..13, from the frozen native TP16 rank0 capture."""

import argparse
import json
from pathlib import Path

import first_difference as frozen

torch = frozen.torch
LAYER = 68
HELPER_SHA256 = "30ae41b719eed1adefd4ea53d33180c8058c3fee5c38ba4fd8127b44e4504043"


def selected_frame(frame, tokens, position):
    """Validate selected-layer layouts without touching other tensor payloads."""
    width = len(tokens)
    frozen.require(
        frame["token"] == tokens
        and all(type(token) is int for token in frame["token"])
        and type(frame["position"]) is int
        and frame["position"] == position
        and type(frame["committed_position"]) is int
        and frame["committed_position"] == position + width,
        "Frame token/position identity differs",
    )
    for field, size, dtypes in (
        ("hidden_states", 6144, (torch.bfloat16,)),
        ("indices", 2048, (torch.int32, torch.int64)),
        ("routes", 8, (torch.int32, torch.int64)),
    ):
        frozen.tensor(frame[field][LAYER], (width, size), dtypes, field)
    cache = frame["cache_rows"][LAYER]
    frozen.require(
        set(cache) == {"latent", "rope", "index", "index_scale"}
        and cache["index"] is None
        and cache["index_scale"] is None,
        "Layer68 must retain latent/rope and share index selection",
    )
    for field, size in (("latent", 512), ("rope", 64)):
        frozen.tensor(cache[field], (width, size), (torch.bfloat16,), field)


def compare_row(serial, wide, row):
    fields = {}
    fields["h68"] = frozen.compare(
        wide["hidden_states"][LAYER][row : row + 1], serial["hidden_states"][LAYER]
    )
    for field in ("latent", "rope"):
        fields["cache." + field] = frozen.compare(
            wide["cache_rows"][LAYER][field][row : row + 1],
            serial["cache_rows"][LAYER][field],
        )
    for field in ("indices", "routes"):
        fields[field] = frozen.choices(
            wide[field][LAYER][row : row + 1],
            serial[field][LAYER],
            field,
            frozen.INITIAL + row,
        )
    return {
        "row_zero_based": row,
        "position": frozen.INITIAL + row,
        "token": serial["token"][0],
        "fields": fields,
    }


def history_selection(rows, serial_indices, wide_indices):
    changed = {
        item["position"]
        for item in rows
        if any(
            not item["fields"]["cache." + field]["bit_equal"]
            for field in ("latent", "rope")
        )
    }
    history_end = frozen.INITIAL + frozen.ROW
    history = changed - {history_end}
    selected = {}
    for name, indices in (("serial", serial_indices), ("wide", wide_indices)):
        positions = set(indices.flatten().tolist())
        selected[name] = {
            "selected_new_positions": sorted(
                position
                for position in positions
                if frozen.INITIAL <= position <= history_end
            ),
            "selected_changed_history_positions": sorted(positions & history),
            "selected_changed_new_positions_including_current": sorted(
                positions & changed
            ),
        }
    return {
        "all_14_new_kv_rows_bit_equal": not bool(changed),
        "changed_new_kv_positions": sorted(changed),
        "changed_history_positions": sorted(history),
        "row13_selection": selected,
        "interpretation": (
            "Selected differing KV permits a candidate explanation, not causal attribution. "
            "Equal saved rows do not establish equality of QL, norm inputs, or metadata."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    helper = Path(frozen.__file__).resolve()
    frozen.require(
        frozen.digest(helper) == HELPER_SHA256, "Diagnostic helper source changed"
    )
    report, tokens, identities = frozen.receipts(args.capture_dir, args.teacher)
    wide = torch.load(
        args.capture_dir / "rank-0/wide.pt",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    selected_frame(wide, tokens, frozen.INITIAL)
    rows = []
    for row in range(frozen.ROW + 1):
        name = f"serial-{row:02d}.pt"
        path = args.capture_dir / "rank-0" / name
        if name not in identities:
            identities[name] = frozen.bind(path, report["trace_sha256"][name])
        serial = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        position = frozen.INITIAL + row
        selected_frame(serial, [tokens[row]], position)
        expected = {
            "input_ids": [tokens[row]],
            "positions": [position],
            "slots": [position],
            "query_cpu": [0, 1],
            "seq_lens": [position + 1],
        }
        frozen.require(
            all(
                serial["model_inputs"][key].tolist() == values
                for key, values in expected.items()
            ),
            "Actual serial model inputs differ: " + name,
        )
        rows.append(compare_row(serial, wide, row))
        if row == frozen.ROW:
            serial_indices = serial["indices"][LAYER].clone()
        del serial
    prefix_checks = [
        check
        for check in report["checks"]
        if check["name"].startswith(("serial.prefix.", "wide.prefix."))
    ]
    failed_prefix = [
        check["name"] for check in prefix_checks if check["passed"] is not True
    ]
    result = {
        "schema": "cann-native-layer68-history-context-v1",
        "rank": 0,
        "layer": LAYER,
        "scope": "Original capture CPU diagnostics; rows0..13 only, no new device execution",
        "h68_semantics": "Merged residual observed at layer68 input norm; not its two actual inputs",
        "case_precision_passed": False,
        "precision_verified": False,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
        "reader_sha256": frozen.digest(Path(__file__)),
        "helper_sha256": HELPER_SHA256,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "identities": identities,
        "inputs": report["inputs"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "captured_source_sha256": report["source_sha256"],
        "frozen_prefix_checks": {
            "count": len(prefix_checks),
            "all_passed": bool(prefix_checks) and not failed_prefix,
            "failed": failed_prefix,
        },
        "rows": rows,
        **history_selection(rows, serial_indices, wide["indices"][LAYER][frozen.ROW]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "all_14_new_kv_rows_bit_equal",
                    "changed_new_kv_positions",
                    "row13_selection",
                    "formal_precision_gate_passed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
