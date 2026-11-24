"""How much over-coverage an 8-token sparse block would cost.

`sparse_flash_attention_tiling.cpp:282-286` takes `V_TEMPLATE` when
`sparseBlockSize <= 4` and `C_TEMPLATE` otherwise, and `C_TEMPLATE` has no
`kvMergeGm_` round trip -- 36.7 microseconds a call, 2.86 ms a verify. The
indexer selects at pool granularity 4, so the attended set is already aligned
four-blocks, one short of the gate.

Passing 8 makes each four-block an eight-block, which attends to more
positions than were selected, and the extra positions cost 17.58 microseconds
a thousand in the graph. Whether the route pays therefore turns on one number:
how often both halves of an eight-slot are already selected. Random placement
at the measured 25% density gives 75% over-coverage and the route still wins
by 9.7 microseconds a call; perfect clustering gives none and it wins by 36.7;
every block alone gives 100% and it loses by 0.7.

This reads the indices the kernel actually received and reports which it is.
"""
import argparse
import json
import pathlib

PICK_SLOPE_US_PER_THOUSAND = 17.58
MERGE_US_PER_CALL = 36.7


def coverage(positions, block):
    """Selected positions, and how many an aligned block cover touches."""
    selected = {value for value in positions if value >= 0}
    slots = {value // block for value in selected}
    return len(selected), len(slots) * block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", type=pathlib.Path,
                        help="a .pt of the indexer's topk_indices")
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    import torch

    tensor = torch.load(args.indices, map_location="cpu")
    positions = [int(value) for value in tensor.flatten().tolist()]
    picked, covered = coverage(positions, args.block)
    extra = covered - picked
    added = PICK_SLOPE_US_PER_THOUSAND * extra / 1000
    net = added - MERGE_US_PER_CALL

    report = {
        "block": args.block,
        "picked": picked,
        "covered": covered,
        "over_coverage": covered / picked - 1,
        "picks_added_us": added,
        "merge_saved_us": MERGE_US_PER_CALL,
        "net_us_per_call": net,
        "net_ms_per_verify": net * 78 / 1000,
        "verdict": "pays" if net < 0 else "does not pay",
    }
    print(json.dumps(report, indent=1))
    if args.output:
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
