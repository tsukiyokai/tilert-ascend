"""The strict device gate distinguishes numerical equality from raw bytes."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "routed_check", Path(__file__).with_name("check.py"),
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_signed_zero_requires_raw_equality():
    result = CHECK.compare(torch, torch.tensor([-0.0], dtype=torch.bfloat16),
                           torch.tensor([0.0], dtype=torch.bfloat16), 0, 0)
    assert result["pass"] and result["exact"]
    assert result["bitwise_equal"] is False
    assert not CHECK.accepts_comparison(result, True)
    assert CHECK.accepts_comparison(result, False)


def test_strict_gate_requires_a_comparable_representation():
    result = CHECK.compare(torch, torch.tensor([1.0], dtype=torch.bfloat16),
                           torch.tensor([1.0], dtype=torch.float32), 0, 0)
    assert result["pass"] and result["bitwise_equal"] is None
    assert not CHECK.accepts_comparison(result, True)


def test_mdl_selection_keeps_an_isolated_library(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "check.py", "--device", "npu:0", "--output", "/tmp/check.json",
        "--native-schedule", "row", "--native-cube", "gu-mdl",
    ])
    args = CHECK.arguments()
    assert args.library.parent.name == "build-gu-mdl"
    assert args.native_api == "isolated"


def test_mdl_rejects_the_engine_namespace(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "check.py", "--device", "npu:0", "--output", "/tmp/check.json",
        "--native-schedule", "row", "--native-cube", "gu-mdl",
        "--native-api", "tilert",
    ])
    with pytest.raises(SystemExit):
        CHECK.arguments()


def test_padded_integer_boundary_uses_only_valid_original_slots():
    counts = torch.zeros(256, dtype=torch.int32)
    counts[2], counts[9], counts[20:33] = 1, 2, 1
    starts = counts.cumsum(0).to(torch.int32) - counts
    original = torch.tensor([1, 0, 8, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])
    metadata = (counts, starts, original // 8, original % 8)
    compact = torch.empty(32, 2, dtype=torch.int32)
    padded = torch.full((16 * 16, 2), 999, dtype=torch.int32)
    for expert in counts.nonzero().flatten().tolist():
        count, begin = int(counts[expert]), int(starts[expert])
        for part in range(2):
            values = (original[begin:begin + count] + 100 * part).to(torch.int32)
            values = torch.stack((values, -values), dim=-1)
            compact[2 * begin + part * count:2 * begin + (part + 1) * count] = values
            padded[16 * begin + part * count:16 * begin + (part + 1) * count] = values
    expected = CHECK.ordered_partials(torch, compact, metadata, 2, 2)
    actual = CHECK.ordered_partials(torch, padded, metadata, 2, 16)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[0, :, 0], torch.arange(16, dtype=torch.int32))
