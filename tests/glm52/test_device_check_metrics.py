"""Distinguish numerical equality from raw representation in device reports."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(params=[("whole_layer/attention_check.py", "metric"),
                        ("mla_tail/check.py", "metrics"),
                        ("routed_int4/check.py", "compare")])
def compare(request):
    relative_path, name = request.param
    path = Path(__file__).parent / relative_path
    spec = importlib.util.spec_from_file_location("device_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def test_signed_zero_is_numerically_equal_but_has_different_bytes(compare):
    actual = torch.tensor([0.0, -0.0], dtype=torch.bfloat16)
    expected = torch.zeros(2, dtype=torch.bfloat16)
    result = compare(torch, actual, expected, 0.0, 0.0)
    assert result["pass"] and result["exact"]
    assert result["bitwise_equal"] is False
    assert result["bitwise_different_bytes"] == 1
    assert compare(torch, actual, actual.clone(), 0.0, 0.0)["bitwise_equal"]


def test_different_dtypes_have_no_raw_byte_comparison(compare):
    actual = torch.tensor([1.0], dtype=torch.bfloat16)
    result = compare(torch, actual, actual.float(), 0.0, 0.0)
    assert result["pass"] and result["exact"]
    assert result["bitwise_equal"] is None
    assert result["bitwise_different_bytes"] is None


def test_nonfinite_values_fail_even_when_bytes_match(compare):
    actual = torch.tensor([float("nan")], dtype=torch.bfloat16)
    result = compare(torch, actual, actual.clone(), 0.0, 0.0)
    assert not result["pass"]
    assert result["bitwise_equal"] is True


def test_singleton_strided_scale_supports_raw_byte_comparison(compare):
    actual = torch.zeros(1, 8, dtype=torch.float32)[:, 0]
    assert actual.stride() == (8,) and actual.is_contiguous()
    expected = torch.zeros(1, dtype=torch.float32)
    assert compare(torch, actual, expected, 0, 0)["bitwise_equal"]
    actual[0] = -0.0
    result = compare(torch, actual, expected, 0, 0)
    assert result["pass"] and result["bitwise_different_bytes"] == 1
