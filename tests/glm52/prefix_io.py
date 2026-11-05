"""Prompt identities and checkpoint-bound layer-78 prefix bundles."""

import hashlib
import json


def prompt_identity(tokens):
    if (not isinstance(tokens, list) or not tokens
            or any(type(token) is not int or not 0 <= token < 154880
                   for token in tokens)):
        raise ValueError("Prompt requires vocabulary token IDs")
    encoded = json.dumps(tokens, separators=(",", ":")).encode("ascii")
    return {"prompt_length": len(tokens),
            "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
            "last_prompt_token": tokens[-1]}


def prefix_prompt_identity(saved, length):
    identity = prompt_identity(saved.get("prompt_token_ids"))
    if identity["prompt_length"] != length:
        raise ValueError("Prompt IDs and frozen backbone prefix differ")
    return identity


def mtp_prefix_layer(saved, checkpoint_hashes):
    if saved.get("metadata_sha256") != checkpoint_hashes:
        raise ValueError("MTP prefix checkpoint differs from decode")
    layer = saved.get("layer")
    if not isinstance(layer, dict):
        raise ValueError("MTP prefix bundle requires a layer dictionary")
    layer = {name: value for name, value in layer.items() if value is not None}
    if set(layer) != {"latent", "rope", "index_keys"}:
        raise ValueError("MTP prefix needs latent, rope and BF16 index keys")
    return layer
