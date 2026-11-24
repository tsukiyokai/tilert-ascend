"""CPU checks for exact failure reproduction and transparent shared execution."""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from audit_decoder_index import audit
from decoder_index_diagnostic import observe_full_indexer, reproduction


def fixture():
    return {
        "hidden": torch.ones(1, 2, dtype=torch.bfloat16), "residual": None,
        "post_ar1": [torch.zeros(1, 2, dtype=torch.bfloat16) for _ in range(2)],
        "metadata": {"slots": torch.tensor([1], dtype=torch.int32)},
        "original": {"output": torch.zeros(1, 2, dtype=torch.bfloat16)},
        "native": {"output": torch.ones(1, 2, dtype=torch.bfloat16)},
    }


class DecoderIndexDiagnosticTests(unittest.TestCase):
    def test_shared_call_reaches_original_once_and_does_not_replace_full_tap(self):
        calls = []

        def original(value, *, mode):
            calls.append(("original", value, mode))
            return value + 1

        adapter = SimpleNamespace(run_body=original)

        class Taps:
            @contextmanager
            def installed(self):
                def indexed(value, *, mode):
                    calls.append(("tap", value, mode))
                    return original(value, mode=mode)
                adapter.run_body = indexed
                try:
                    yield
                finally:
                    adapter.run_body = original

        with observe_full_indexer(Taps(), adapter):
            self.assertEqual(adapter.run_body(2, mode="int8"), 3)
            self.assertEqual(adapter.run_body(4, mode="shared"), 5)
        self.assertEqual(calls, [("tap", 2, "int8"), ("original", 2, "int8"),
                                 ("original", 4, "shared")])
        self.assertIs(adapter.run_body, original)

    def test_reproduction_rejects_changed_tp_input_signed_zero_or_metadata(self):
        self.assertTrue(reproduction(torch, fixture(), fixture())["pass"])
        for group, name in (("post_ar1", 1), ("original", "output"),
                            ("metadata", "slots")):
            current = fixture()
            value = current[group][name]
            if value.is_floating_point():
                value.flatten()[0] = -0.0
            else:
                value.flatten()[0] = 2
            with self.subTest(group=group):
                self.assertFalse(reproduction(torch, fixture(), current)["pass"])

    def test_reproduction_rejects_residual_branch_and_missing_output(self):
        current = fixture()
        current["residual"] = current["hidden"].clone()
        self.assertFalse(reproduction(torch, fixture(), current)["pass"])
        current = fixture()
        current["native"].clear()
        self.assertFalse(reproduction(torch, fixture(), current)["pass"])

    def test_cache_difference_maps_physical_page_to_live_input_and_order(self):
        cache = torch.zeros(6, 128, 1, 128, dtype=torch.int8)
        changed = cache.clone()
        changed[5, 121, 0, 31] = 1
        topk = torch.tensor([[8185, 4], [3, 2]], dtype=torch.int32)
        reordered = topk.clone()
        reordered[0] = topk[0].flip(0)
        values = {
            "hidden": torch.zeros(2, 6144, dtype=torch.bfloat16),
            "metadata": {"positions": torch.tensor([8185, 8186]),
                         "index_slots": torch.tensor([761, 762]),
                         "index_table": torch.tensor([[0] * 63 + [5]])},
            "original": {"layer6.index_cache": cache, "layer6.topk": topk,
                         "layer7.topk": topk.clone()},
            "native": {"layer6.index_cache": changed, "layer6.topk": reordered,
                       "layer7.topk": reordered.clone()},
        }
        result = audit(torch, values)
        difference = result["cache_differences"][0]
        self.assertEqual(difference["input_token_rows"], [0])
        self.assertEqual(difference["logical_positions"], [8185])
        self.assertTrue(result["topk_differences"]["layer6.topk"][0][
            "whole_row_multiset_equal"])


if __name__ == "__main__":
    unittest.main()
