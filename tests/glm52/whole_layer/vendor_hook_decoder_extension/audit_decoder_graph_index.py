"""Locate a saved decoder graph cache difference and verify its live producers."""

import argparse
import hashlib
import json
from pathlib import Path


def tensor_hash(torch, value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def audit(torch, fixture, report):
    original, native, plan = (fixture[name] for name in ("expected", "actual", "plan"))
    case = report["cases"][-1]
    rows, iteration = case["rows"], case["iteration"]
    differences = []
    left, right = original["layer6.index_cache"], native["layer6.index_cache"]
    for coordinate in (left != right).nonzero().tolist():
        page, offset, head, channel = coordinate
        slot = page * 128 + offset
        tokens = (plan["index_slots"] == slot).nonzero().flatten().tolist()
        logical_pages = (plan["index_table"][0] == page).nonzero().flatten().tolist()
        differences.append({
            "coordinate": coordinate, "original": left[tuple(coordinate)].item(),
            "native": right[tuple(coordinate)].item(), "physical_slot": slot,
            "input_token_rows": tokens,
            "logical_positions": [index * 128 + offset for index in logical_pages],
            "input_positions": [plan["positions"][index].item() for index in tokens],
            "original_fp16_scale": original["layer6.index_scale"][page, offset, head, 0].item(),
            "native_fp16_scale": native["layer6.index_scale"][page, offset, head, 0].item(),
        })
    generator = torch.Generator().manual_seed(42023 + rows * 100 + iteration)
    producers = {"hidden": torch.randn(rows, 6144, generator=generator).bfloat16()}
    if case["initial_residual"] == "provided":
        producers["residual"] = torch.randn(rows, 6144, generator=generator).bfloat16()
    for index in range(2):
        producers[f"post_ar1:{index}"] = torch.randn(rows, 6144, generator=generator).bfloat16()
    hashes = {name: {"regenerated_sha256": tensor_hash(torch, value),
                     "recorded_sha256": case["inputs"]["tensors"][name]["sha256"]}
              for name, value in producers.items()}
    for value in hashes.values():
        value["equal"] = value["regenerated_sha256"] == value["recorded_sha256"]
    checks = {name: bool(torch.equal(value.contiguous().view(torch.uint8),
                                    native[name].contiguous().view(torch.uint8)))
              for name, value in original.items()}
    return {
        "scope": "Saved graph outputs and live metadata; no front-end score or arithmetic reconstruction",
        "case": {name: case[name] for name in ("rows", "initial_residual", "iteration", "phase")},
        "cache_differences": differences, "boundary_byte_equality": checks,
        "regenerated_producer_hashes": hashes,
        "cpu_reconstruction": {
            "torch_version": torch.__version__,
            "all_producers_byte_equal": all(value["equal"] for value in hashes.values()),
            "scope": "Auxiliary CPU RNG reconstruction; the device reproduction gate compares the original live-input hashes directly",
        },
        "input_norm_byte_equal": checks["layer6.input_norm.0"],
        "ordered_topk_byte_equal": checks["layer6.topk"] and checks["layer7.topk"],
        "current_slot_only": all(value["input_token_rows"] for value in differences),
        "missing_observations": ["WK projection", "key LayerNorm", "RoPE", "Hadamard", "dynamic quantization"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    result = audit(torch, torch.load(args.fixture, map_location="cpu", weights_only=True),
                   json.loads(args.report.read_text()))
    result.update(fixture_path=str(args.fixture),
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
