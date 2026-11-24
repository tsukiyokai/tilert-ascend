"""Check the complete TCubeTiling produced by the accepted N1024 host tiler."""

import hashlib


EXPECTED_HEX = '0100000040000000001800008000000080000000400000000018000080000000200000000004000040000000040000000c00000002000000060000000000000000000000000000000000000000100600000002000000000001000000010000000100000001000000020000000200000000000000000000000200000002000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
EXPECTED_SHA = 'c207230aafb11e2c2f825cd9a2ea70147ee7733046b8b2957593c9c2bc3e8009'


def check_down_tiling(torch, value):
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 200:
        raise ValueError("Down tiling requires at least 200 explicit bytes")
    payload = bytes(value[:200].detach().cpu().tolist())
    if payload.hex() != EXPECTED_HEX:
        raise RuntimeError("Prepared Down tiling differs from the N1024 host tiler")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != EXPECTED_SHA:
        raise RuntimeError("The Down tiling byte oracle is inconsistent")
    return {"address": value.data_ptr(), "bytes": 200,
            "sha256": actual, "pass": True}
