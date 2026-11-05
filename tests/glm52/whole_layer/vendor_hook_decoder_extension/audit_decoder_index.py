"""Locate index-cache differences in a frozen complete-decoder comparison."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(torch, fixture):
    original, native = fixture["original"], fixture["native"]
    plan = fixture["metadata"]
    before, after = original["layer6.index_cache"], native["layer6.index_cache"]
    cache = []
    for coordinate in (before != after).nonzero().tolist():
        page, offset, head, channel = coordinate
        physical_slot = page * 128 + offset
        tokens = (plan["index_slots"] == physical_slot).nonzero().flatten().tolist()
        logical_pages = (plan["index_table"][0] == page).nonzero().flatten().tolist()
        cache.append({"coordinate": coordinate,
                      "original": before[tuple(coordinate)].item(),
                      "native": after[tuple(coordinate)].item(),
                      "physical_slot": physical_slot, "input_token_rows": tokens,
                      "logical_positions": [index * 128 + offset for index in logical_pages],
                      "input_positions": [plan["positions"][index].item() for index in tokens]})
    topk = {}
    for layer in (6, 7):
        name = f"layer{layer}.topk"
        left, right = original[name], native[name]
        rows = []
        for row in (left != right).any(1).nonzero().flatten().tolist():
            ranks = (left[row] != right[row]).nonzero().flatten()
            rows.append({"row": row, "ranks": ranks.tolist(),
                         "original": left[row, ranks].tolist(),
                         "native": right[row, ranks].tolist(),
                         "whole_row_multiset_equal": bool(torch.equal(
                             left[row].sort().values, right[row].sort().values))})
        topk[name] = rows
    equality = {name: bool(torch.equal(value.contiguous().view(torch.uint8),
                                     native[name].contiguous().view(torch.uint8)))
                for name, value in original.items()}
    return {"scope": "Saved device outputs and metadata; no inferred device score",
            "rows": fixture["hidden"].shape[0], "cache_differences": cache,
            "topk_differences": topk, "boundary_byte_equality": equality,
            "scores_available": False,
            "missing_score_inputs": "The original dump does not contain index query codes/scales or head weights"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    result = audit(torch, torch.load(args.fixture, map_location="cpu", weights_only=True))
    result.update(fixture_path=str(args.fixture),
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
