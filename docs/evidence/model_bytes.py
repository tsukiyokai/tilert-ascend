"""Audit checkpoint storage from safetensors headers; never read tensor payloads."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct


def category(name):
    for pattern, label in (
        (".mlp.experts.", "routed_experts"),
        (".mlp.shared_experts.", "shared_expert"),
        (".mlp.gate.", "router"),
        (".mlp.", "dense_ffn"),
        (".indexer.", "indexer"),
        (".self_attn.", "attention"),
    ):
        if pattern in name:
            return label
    return {
        "model.embed_tokens.weight": "embedding",
        "lm_head.weight": "lm_head",
        "rot.weight": "rotation",
    }.get(name, "other")


def audit(root):
    inputs = {}
    hashes = {}
    for name in (
        "config.json",
        "quant_model_description.json",
        "quant_model_weights.safetensors.index.json",
    ):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        inputs[name] = json.loads(raw)
    index = inputs["quant_model_weights.safetensors.index.json"]
    quant = inputs["quant_model_description.json"]
    headers = {}
    for name in sorted(set(index["weight_map"].values())):
        with (root / name).open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            headers[name] = json.loads(stream.read(length))

    groups = defaultdict(Counter)
    examples = {}
    for name, filename in index["weight_map"].items():
        tensor = headers[filename][name]
        size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
        match = re.search(r"model\.layers\.(\d+)\.", name)
        prefix = "mtp/" if match and int(match[1]) == 78 else "main/"
        group = groups[prefix + category(name)]
        group["tensors"] += 1
        group["bytes"] += size
        group["dtype_" + tensor["dtype"]] += size
        if name.startswith(("model.layers.0.", "model.layers.3.mlp.experts.0.")):
            examples[name] = {
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "bytes": size,
                "quant": quant.get(name),
            }

    return {
        "model_path": str(root),
        "input_sha256": hashes,
        "config": inputs["config.json"],
        "quantization": {
            key: quant.get(key)
            for key in (
                "model_quant_type", "group_size", "is_rot_used",
                "indexer_quant_type", "fa_quant_type",
            )
        },
        "indexer_layer_quantization": {
            key: value for key, value in quant.items()
            if key.endswith(".indexer.quant_type")
        },
        "metadata": index["metadata"],
        "indexed_payload_bytes": sum(group["bytes"] for group in groups.values()),
        "tensor_count": len(index["weight_map"]),
        "shard_count": len(headers),
        "groups": dict(groups),
        "examples": examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.model), indent=2, sort_keys=True))
