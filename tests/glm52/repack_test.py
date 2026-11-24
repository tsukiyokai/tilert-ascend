"""Verify native non-transposed B[K,N] preserves checkpoint INT4 codes."""
import pathlib
import sys

import torch

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from tilert.models.glm_5_2_ascend.routed_expert import _repack


def signed(nibble):
    return torch.where(nibble > 7, nibble.int() - 16, nibble.int())


def unpack_along_outputs(packed, columns):
    """The checkpoint's reading: byte holds outputs 2n and 2n+1 for one K."""
    experts, halved, inner = packed.shape
    low = signed(packed & 0x0F)
    high = signed((packed >> 4) & 0x0F)
    stacked = torch.stack((low, high), dim=2)
    return stacked.reshape(experts, 2 * halved, inner)


def unpack_native_b(packed):
    """Native B[E,K,N/2] pairs adjacent output columns."""
    low = signed(packed & 0x0F)
    high = signed((packed >> 4) & 0x0F)
    experts, inner, halved = packed.shape
    stacked = torch.stack((low, high), dim=3)
    return stacked.reshape(experts, inner, 2 * halved).transpose(1, 2)


def main():
    torch.manual_seed(20261020)
    for experts, columns, inner in ((2, 8, 16), (3, 256, 64), (1, 64, 128)):
        original = torch.randint(0, 256, (experts, columns // 2, inner),
                                 dtype=torch.uint8).to(torch.int8)
        expected = unpack_along_outputs(original, columns)
        actual = unpack_native_b(_repack(original, columns))
        same = bool(torch.equal(actual, expected))
        print(f"{experts}x{columns}x{inner}: codes preserved {same}  "
              f"unequal {int((actual != expected).sum())}/{actual.numel()}")
        if not same:
            raise SystemExit("repack does not preserve the codes")


if __name__ == "__main__":
    main()
