"""Inspect restored prefix identities and tensor layouts on CPU, without a model.

Payload SHA-256 is checked by prefix_transfer.py. Main cache tensors are mapped
read-only for their descriptors; only the small parent-hidden tensors are read.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


MANIFEST_SHA = "b01cceb2d2e30564807ce9efc441ad535999b56b73d0c333e93bb18694ccda74"


def expected_layer(number, rows):
    result = {"latent": ([rows, 512], "torch.bfloat16"),
              "rope": ([rows, 64], "torch.bfloat16")}
    if number in (0, 1, 2, *range(6, 79, 4)):
        quantized = number in range(6, 67, 4)
        result["index_keys"] = ([rows, 128],
                                "torch.int8" if quantized else "torch.bfloat16")
        if quantized:
            result["index_scales"] = ([rows], "torch.float16")
    return result


def check_layer(values, number, rows):
    expected = expected_layer(number, rows)
    observed = {name: (list(value.shape), str(value.dtype))
                for name, value in values.items()}
    if observed != expected or any(value.device.type != "cpu"
                                   for value in values.values()):
        raise ValueError(f"Layer {number} prefix has a different tensor layout")
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if list(Path("/dev").glob("davinci*")):
        raise RuntimeError("Use a CPU-only container without accelerator devices")
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if "torch_npu" in sys.modules:
        raise RuntimeError("Disable device backend autoload for prefix inspection")
    raw = (args.case / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("The restored capture manifest changed")
    manifest = json.loads(raw)
    checkpoint = manifest["runtime"]["checkpoint_sha256"]
    contract = manifest["mtp_parent_hidden"]
    parent_identity = {**contract, "request_id": manifest["request_id"],
                       "generation": manifest["generation"]}
    if contract != {"position": 8191, "normalized": True, "rotated": False,
                    "dtype": "bfloat16", "shape": [1, 6144]}:
        raise ValueError("The captured parent-hidden contract changed")
    reports = []
    for rank in range(16):
        main_prefix = torch.load(args.case / f"main-prefix/rank-{rank}.pt",
                                 map_location="cpu", weights_only=True, mmap=True)
        if (main_prefix["metadata_sha256"] != checkpoint
                or main_prefix["prompt_token_ids"] != manifest["prompt_token_ids"]
                or len(main_prefix["layers"]) != 78):
            raise ValueError(f"Main prefix identity changed at rank {rank}")
        main_shapes = [check_layer(layer, number, 8192)
                       for number, layer in enumerate(main_prefix["layers"])]
        mtp = torch.load(args.case / f"mtp-prefix/rank-{rank}.pt",
                         map_location="cpu", weights_only=True, mmap=True)
        parent = mtp["parent_hidden"]
        if (mtp["metadata_sha256"] != checkpoint
                or {key: value for key, value in parent.items() if key != "tensor"}
                != parent_identity):
            raise ValueError(f"MTP prefix identity changed at rank {rank}")
        value = parent["tensor"]
        if (value.shape != (1, 6144) or value.dtype != torch.bfloat16
                or value.device.type != "cpu" or not torch.isfinite(value).all()):
            raise ValueError(f"MTP parent tensor differs at rank {rank}")
        reports.append({
            "rank": rank, "main_layers": main_shapes,
            "mtp_layer": check_layer(mtp["layer"], 78, 8191),
            "parent_identity": parent_identity,
            "parent_tensor_sha256": hashlib.sha256(
                value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "parent_finite": True, "pass": True,
        })
        del main_prefix, mtp, parent, value
    result = {"manifest_sha256": MANIFEST_SHA, "torch": torch.__version__,
              "checkpoint_metadata": checkpoint, "ranks": reports,
              "payload_file_hash_scope": "separate transfer report",
              "main_cache_contents_read": False,
              "parent_tensor_bytes_read": 16 * 6144 * 2,
              "model_started": False, "npu_backend_imported": False, "pass": True}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"ranks": 16, "main_layers_per_rank": 78,
                      "parent_tensors": 16, "pass": True}))


if __name__ == "__main__":
    main()
