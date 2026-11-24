"""Check the complete TCubeTiling produced by the accepted K128 host tiler."""

import hashlib


EXPECTED_HEX = '0100000040000000001800008000000080000000400000000018000080000000200000000004000080000000020000000600000002000000060000000000000000000000000000000000000000100600000002000000000001000000010000000100000001000000010000000100000000000000000000000200000001000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
EXPECTED_SHA = 'c2b73af9d605ad8e64ea551f288d188a0cf636828a5a0471ea7c63d616f50999'


def check_down_tiling(torch, value):
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() < 200:
        raise ValueError("Down tiling requires at least 200 explicit bytes")
    payload = bytes(value[:200].detach().cpu().tolist())
    if payload.hex() != EXPECTED_HEX:
        raise RuntimeError("Prepared Down tiling differs from the K128 host tiler")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != EXPECTED_SHA:
        raise RuntimeError("The Down tiling byte oracle is inconsistent")
    return {"address": value.data_ptr(), "bytes": 200,
            "sha256": actual, "pass": True}
