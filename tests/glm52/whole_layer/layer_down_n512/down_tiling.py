"""Check the complete TCubeTiling produced by the accepted N512 host tiler."""

import hashlib


EXPECTED_HEX = (
    '0100000040000000001800008000000080000000400000000018000080000000'
    '400000000002000080000000010000000c000000010000000c00000000000000'
    '0000000000000000000000000010060000000200000000000100000001000000'
    '0100000001000000010000000100000000000000000000000200000002000000'
    '0100000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    '0000000000000000'
)
EXPECTED_SHA = "bb8ef4481b8c63b00dc43940975ee5b172818bcd228e8bc553b22234cbb2a244"


def check_down_tiling(torch, value):
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 200:
        raise ValueError("Down tiling requires at least 200 explicit bytes")
    payload = bytes(value[:200].detach().cpu().tolist())
    if payload.hex() != EXPECTED_HEX:
        raise RuntimeError("Prepared Down tiling differs from the N512 host tiler")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != EXPECTED_SHA:
        raise RuntimeError("The Down tiling byte oracle is inconsistent")
    return {"address": value.data_ptr(), "bytes": 200,
            "sha256": actual, "pass": True}
