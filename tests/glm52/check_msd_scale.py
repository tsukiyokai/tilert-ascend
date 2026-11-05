"""Check the A3 Fixpipe scale contract without a device or checkpoint."""

import torch

from tilert.models.glm_5_2_ascend.reference_weights import w4_msd_product


def main():
    torch.set_num_threads(1)
    codes = torch.tensor([[16], [-16]], dtype=torch.int32)
    integer = torch.ones((2, 1), dtype=torch.int32)
    # Noncontiguous input includes discarded bits and a retained bit.
    scales = torch.tensor([1 + 3 / 4096, 0, 1 + 1 / 1024, 0])[::2]
    frozen = scales.clone()
    output = w4_msd_product(
        codes, integer, scales, torch.zeros(2), torch.ones((2, 1)),
        output_dtype=torch.float32,
    )
    expected = torch.tensor([[8, 8 + 1 / 128], [-24, -24 - 3 / 128]])
    assert torch.equal(output, expected), (output, expected)
    assert torch.equal(scales, frozen)
    print("A3 MSD scale: 4 exact outputs; input unchanged")


if __name__ == "__main__":
    main()
