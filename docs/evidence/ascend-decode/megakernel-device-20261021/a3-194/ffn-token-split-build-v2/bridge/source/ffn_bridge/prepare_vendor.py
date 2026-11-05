"""Freeze the pinned top-k arithmetic with explicit full-copy capacities."""

import argparse
import hashlib
import json
from pathlib import Path


TOPK = "csrc/moe/moe_gating_top_k/op_kernel"
NORM = "csrc/moe/add_rms_norm_bias/op_kernel"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    names = [(TOPK, "common.h"), (TOPK, "moe_gating_top_k_without_group.h")]
    names += [(NORM, name) for name in (
        "add_rms_norm_bias_single_n.h", "rms_norm_base.h", "reduce_common.h")]
    for relative, name in names:
        path = source / relative / name
        original = path.read_bytes()
        data = original
        if name == "moe_gating_top_k_without_group.h":
            replacements = {
                "pipe_->InitBuffer(expertIdxOutQueue_, 1, "
                "Align(k_, sizeof(float)) * sizeof(int32_t));":
                    "pipe_->InitBuffer(expertIdxOutQueue_, 1, "
                    "expertCountAlign_ * sizeof(int32_t));",
                "pipe_->InitBuffer(topKExpertIdBuf_, "
                "Align(k_, sizeof(float)) * sizeof(int32_t));":
                    "pipe_->InitBuffer(topKExpertIdBuf_, "
                    "expertCountAlign_ * sizeof(int32_t));",
            }
            text = data.decode()
            for old, new in replacements.items():
                if text.count(old) != 1:
                    raise ValueError(f"Pinned top-k capacity expression changed: {old}")
                text = text.replace(old, new)
            data = text.encode()
        (destination / name).write_bytes(data)
        manifest.append({"source": str(path), "name": name,
                         "original_sha256": digest(original),
                         "compiled_sha256": digest(data),
                         "changed": data != original})
    (destination / "manifest.json").write_text(
        json.dumps({"scope": "Two UB capacities; arithmetic and ordering unchanged",
                    "files": manifest}, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
