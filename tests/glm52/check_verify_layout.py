"""Check causal verify layouts and cache rollback without Torch or NPU."""

import argparse
from dataclasses import FrozenInstanceError
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tilert.models.glm_5_2_ascend.ops.indexer import (  # noqa: E402
    NativeIndexer,
    TOP_K,
    VerifyLayout,
    cache_append_range,
    validate_truncate,
)
from tilert.models.glm_5_2_ascend.ops.mla import NativeMla  # noqa: E402


PREFIXES = (0, 1, 7, 8, 124, 125, 126, 127, 128, 129,
            2044, 2047, 2048, 2049, 8188, 8191, 8192)


def layouts():
    for rows in (1, 2, 4):
        for prefix in PREFIXES:
            yield prefix, VerifyLayout(rows, prefix + rows)


def selected_rows(layout):
    """Choose distinct, noncontiguous positions without score assumptions."""
    random_source = random.Random(2701 + layout.cache_length)
    result = []
    for bound, count in zip(layout.key_bounds, layout.valid_counts):
        selected = random_source.sample(range(bound), count)
        result.append(selected + [-1] * (TOP_K - count))
    return result


class LengthRecorder:
    """Record metadata writes made by the actual truncate/reset methods."""

    def __init__(self, length):
        self.value = length
        self.writes = []

    def fill_(self, value):
        self.value = value
        self.writes.append(value)
        return self


def cache_metadata(cache_type, length):
    """Create no tensors and invoke no operator or cache constructor."""
    cache = cache_type.__new__(cache_type)
    cache.length = length
    cache.key_length = LengthRecorder(length)
    return cache


class VerifyLayoutChecks(unittest.TestCase):
    def test_right_down_bounds_match_scalar_causal_mask(self):
        for prefix, layout in layouts():
            with self.subTest(prefix=prefix, rows=layout.query_rows):
                for query, bound in enumerate(layout.key_bounds):
                    visible = tuple(
                        key for key in range(layout.cache_length)
                        if key <= prefix + query
                    )
                    self.assertEqual(visible, tuple(range(bound)))

    def test_one_request_length_and_operator_output_shape(self):
        for prefix, layout in layouts():
            self.assertEqual(
                [layout.query_rows], [layout.cache_length - prefix]
            )
            self.assertEqual(
                [layout.cache_length], [prefix + layout.query_rows]
            )
            self.assertEqual(layout.sparse_shape,
                             (layout.query_rows, 1, TOP_K))

    def test_boundary_rows_have_distinct_valid_counts(self):
        layout = VerifyLayout(4, 2050)
        self.assertEqual(layout.key_bounds, (2047, 2048, 2049, 2050))
        self.assertEqual(layout.valid_counts, (2047, 2048, 2048, 2048))
        self.assertEqual(VerifyLayout(4, 4).valid_counts, (1, 2, 3, 4))

    def test_m1_retains_original_visible_length(self):
        for prefix in PREFIXES:
            layout = VerifyLayout(1, prefix + 1)
            self.assertEqual(layout.key_bounds, (prefix + 1,))
            self.assertEqual(layout.valid_counts, (min(prefix + 1, TOP_K),))

    def test_valid_rows_cover_all_shapes_and_leave_inputs_unchanged(self):
        for prefix, layout in layouts():
            with self.subTest(prefix=prefix, rows=layout.query_rows):
                rows = selected_rows(layout)
                original = [row[:] for row in rows]
                layout.validate_indices(rows)
                self.assertEqual(rows, original)

    def test_future_indices_are_rejected_before_global_cache_end(self):
        layout = VerifyLayout(4, 8196)
        for query in range(3):
            rows = selected_rows(layout)
            rows[query][0] = layout.key_bounds[query]
            with self.subTest(query=query), self.assertRaises(ValueError):
                layout.validate_indices(rows)

    def test_row_permutation_cannot_hide_future_keys(self):
        layout = VerifyLayout(4, 4)
        rows = [list(range(count)) + [-1] * (TOP_K - count)
                for count in range(1, 5)]
        rows[0], rows[-1] = rows[-1], rows[0]
        with self.assertRaises(ValueError):
            layout.validate_indices(rows)

    def test_duplicate_and_out_of_range_indices_are_rejected(self):
        layout = VerifyLayout(2, 8194)
        for invalid in (-2, -1, 8194, 8195, True, 1.0, "1"):
            rows = selected_rows(layout)
            rows[1][0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                layout.validate_indices(rows)
        rows = selected_rows(layout)
        rows[1][0] = rows[1][1]
        with self.assertRaises(ValueError):
            layout.validate_indices(rows)

    def test_exact_selection_rejects_missing_or_extra_keys(self):
        layout = VerifyLayout(2, 8)
        for query in range(2):
            count = layout.valid_counts[query]
            for slot, value in ((count - 1, -1), (count, 0)):
                rows = selected_rows(layout)
                rows[query][slot] = value
                with self.subTest(query=query, slot=slot):
                    with self.assertRaises(ValueError):
                        layout.validate_indices(rows)

    def test_index_row_shapes_are_checked(self):
        layout = VerifyLayout(2, 8)
        valid = selected_rows(layout)
        invalid_rows = (
            [], valid[:1], valid + valid[:1],
            [valid[0][:-1], valid[1]], [valid[0] + [-1], valid[1]],
        )
        for rows in invalid_rows:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                layout.validate_indices(rows)

    def test_partial_mtp_indices_require_explicit_opt_in(self):
        layout = VerifyLayout(4, 12)
        rows = [[3, 1] + [-1] * (TOP_K - 2) for _ in range(4)]
        with self.assertRaises(ValueError):
            layout.validate_indices(rows)
        layout.validate_indices(rows, allow_partial=True)

    def test_partial_mtp_indices_still_reject_holes_and_empty_sets(self):
        layout = VerifyLayout(2, 8)
        invalid_rows = (
            [-1] * TOP_K,
            [0, -1, 1] + [-1] * (TOP_K - 3),
            [0, 0] + [-1] * (TOP_K - 2),
            [7] + [-1] * (TOP_K - 1),
            list(range(8)) + [-1] * (TOP_K - 8),
        )
        for invalid in invalid_rows:
            rows = [invalid, [0] + [-1] * (TOP_K - 1)]
            with self.subTest(first=invalid[:8]):
                with self.assertRaises(ValueError):
                    layout.validate_indices(rows, allow_partial=True)

    def test_invalid_layout_sizes_are_rejected(self):
        for rows in (0, 3, 5, True, 1.0, "1"):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                VerifyLayout(rows, 8192)
        for length in (-1, 0, 3, True, 8.0, "8"):
            with self.subTest(length=length), self.assertRaises(ValueError):
                VerifyLayout(4, length)

    def test_layout_is_frozen(self):
        layout = VerifyLayout(4, 8196)
        with self.assertRaises(FrozenInstanceError):
            layout.cache_length = 8192

    def test_append_slices_cover_each_query_once(self):
        for prefix, layout in layouts():
            target = cache_append_range(prefix, layout.query_rows,
                                        layout.cache_length)
            self.assertEqual(
                tuple(range(target.start, target.stop)),
                tuple(prefix + row for row in range(layout.query_rows)),
            )

    def test_append_capacity_and_row_counts_are_checked(self):
        for rows in (0, 3, 5, True, 1.0):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                cache_append_range(0, rows, 16)
        for length, rows, capacity in ((16, 1, 16), (15, 2, 16),
                                      (14, 4, 16), (-1, 1, 16)):
            with self.subTest(length=length), self.assertRaises(ValueError):
                cache_append_range(length, rows, capacity)

    def test_actual_cache_truncate_updates_both_length_fields(self):
        for cache_type in (NativeIndexer, NativeMla):
            for accepted in range(5):
                cache = cache_metadata(cache_type, 8196)
                cache.truncate(8192 + accepted)
                self.assertEqual(cache.length, 8192 + accepted)
                self.assertEqual(cache.key_length.value, cache.length)
                self.assertEqual(cache.key_length.writes, [cache.length])

    def test_actual_cache_reset_uses_zero_length(self):
        for cache_type in (NativeIndexer, NativeMla):
            cache = cache_metadata(cache_type, 8196)
            cache.reset()
            self.assertEqual((cache.length, cache.key_length.value), (0, 0))

    def test_invalid_truncate_is_atomic_for_length_metadata(self):
        for cache_type in (NativeIndexer, NativeMla):
            for invalid in (-1, 8197, True, 2.0, "2", None):
                cache = cache_metadata(cache_type, 8196)
                with self.subTest(cache=cache_type.__name__, value=invalid):
                    with self.assertRaises(ValueError):
                        cache.truncate(invalid)
                    self.assertEqual(cache.length, 8196)
                    self.assertEqual(cache.key_length.value, 8196)
                    self.assertEqual(cache.key_length.writes, [])

    def test_rollback_then_append_overwrites_only_rejected_suffix(self):
        for prefix, layout in layouts():
            count = layout.query_rows
            for accepted in range(count + 1):
                capacity = prefix + 2 * count
                storage = list(range(layout.cache_length))
                storage.extend([None] * (capacity - len(storage)))
                retained = storage[:prefix + accepted]
                valid_length = prefix + accepted
                validate_truncate(layout.cache_length, valid_length)
                target = cache_append_range(valid_length, count, capacity)
                replacements = [-(row + 1) for row in range(count)]
                storage[target] = replacements
                self.assertEqual(storage[:valid_length], retained)
                self.assertEqual(storage[target], replacements)
                next_layout = VerifyLayout(count, target.stop)
                self.assertEqual(next_layout.key_bounds,
                                 tuple(valid_length + row + 1
                                       for row in range(count)))

    def test_import_and_metadata_calls_do_not_load_torch(self):
        self.assertFalse(any(
            name == "torch" or name.startswith("torch.")
            for name in sys.modules
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(VerifyLayoutChecks)
    )
    print(stream.getvalue(), end="")
    files = (
        Path(__file__),
        ROOT / "tilert/models/glm_5_2_ascend/ops/indexer.py",
        ROOT / "tilert/models/glm_5_2_ascend/ops/mla.py",
        ROOT / "tilert/models/glm_5_2_ascend/ops/__init__.py",
    )
    report = {
        "status": "pass" if result.wasSuccessful() else "fail",
        "scope": "stdlib causal layout and cache-length checks",
        "device_execution": False,
        "tensor_paths_executed": False,
        "hostname": platform.node(), "python": platform.python_version(),
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "query_rows": [1, 2, 4], "prefixes": list(PREFIXES),
        "layout_fixtures": sum(1 for _ in layouts()),
        "causal_key_bound": "prefix_length + query_row + 1",
        "query_lengths": "[M]", "key_lengths": "[P+M]",
        "native_index_shape": "[M,1,2048]",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in files
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
