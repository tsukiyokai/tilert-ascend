"""Prepare small immutable controls from the saved TP16 M1/M32 captures."""

import argparse
import json
import os
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch
from first_difference import bind, digest, receipts, validate_frame

FIRST_SERIAL_SHA = (
    "2d3bd58b42339aceee22a570c69993ed38a7ac13fada77bf02f02193198ad29f"
)
HELPER_SHA = "30ae41b719eed1adefd4ea53d33180c8058c3fee5c38ba4fd8127b44e4504043"


def view(frame):
    result = {
        name: frame[name].clone()
        for name in ("logits", "last_hidden", "token_ids")
    }
    result.update(
        h68=frame["hidden_states"][68].clone(),
        h69=frame["hidden_states"][69].clone(),
        indices=frame["indices"][68].clone(),
        routes=frame["routes"][68].clone(),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    import first_difference

    assert digest(Path(first_difference.__file__)) == HELPER_SHA
    report, teacher, identities = receipts(args.capture_dir, args.teacher)
    path = args.capture_dir / "rank-0/serial-00.pt"
    identities["serial-00.pt"] = bind(path, FIRST_SERIAL_SHA)
    first = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    serial = torch.load(
        args.capture_dir / "rank-0/serial-13.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    wide = torch.load(
        args.capture_dir / "rank-0/wide.pt",
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    validate_frame(first, [teacher[0]], 8192)
    validate_frame(serial, [teacher[13]], 8205)
    validate_frame(wide, teacher, 8192)
    args.output.mkdir(parents=True, exist_ok=False)
    values = {
        "serial_first": view(first),
        "serial_target": view(serial),
        "wide": view(wide),
        "teacher_tokens": teacher,
        "serial_predictions": next(
            row["predictions"]
            for row in report["states"]
            if row["phase"] == "serial"
        ),
        "checkpoint_sha256": report["checkpoint_sha256"],
    }
    path = args.output / "reference.pt"
    with path.open("xb") as stream:
        torch.save(values, stream)
    receipt = {
        "schema": "width-boundaries-controls-v1",
        "scope": "Views of saved actual A3 outputs; no new model execution",
        "identities": identities,
        "source_sha256": digest(Path(__file__)),
        "helper_sha256": HELPER_SHA,
        "reference_sha256": digest(path),
        "reference_bytes": path.stat().st_size,
        "formal_precision_gate_passed": False,
        "performance_measured": False,
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
