"""Exercise bounded profiling lifetime and actual method ownership on CPU."""

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bounded_round_profile import RoundCapture


class Model:
    def compute_logits(self, value, *, factor=1):
        return value * factor


class Target:
    def __init__(self):
        self.model = Model()
        self.calls = []

    def _metadata(self, value, *, position=0):
        self.calls.append((value, position))
        return value + position

    def verify(self, value, *, position=0, factor=1):
        metadata = self._metadata(value, position=position)
        return self.model.compute_logits(metadata, factor=factor)

    def host_tokens(self, value, *, suffix=()):
        return (value, *suffix)


class Profiler:
    def __init__(self, fixture):
        self.fixture = fixture

    def __enter__(self):
        self.fixture.events.append(("profile_enter",))
        return self

    def __exit__(self, kind, error, traceback):
        fixture = self.fixture
        fixture.events.append(("profile_exit", kind))
        fixture.test.assertEqual(fixture.open_ranges, [])
        fixture.assert_restored()
        if fixture.cleanup_error:
            raise fixture.cleanup_error
        return fixture.suppress


class Fixture:
    def __init__(self, test, target=None):
        self.test = test
        self.target = Target() if target is None else target
        self.events, self.open_ranges = [], []
        self.cleanup_error = None
        self.suppress = False
        self.ownership = [
            (owner, name, name in vars(owner), vars(owner).get(name))
            for owner, name in (
                (self.target, "_metadata"),
                (self.target, "verify"),
                (self.target, "host_tokens"),
                (self.target.model, "compute_logits"),
            )
        ]
        self.helper = ModuleType("profile_native_session")

        def make_profiler(path, rank):
            self.events.append(("factory", path, rank))
            return Profiler(self)

        @contextmanager
        def npu_range(name):
            self.events.append(("range_enter", name))
            self.open_ranges.append(name)
            try:
                yield
            finally:
                self.test.assertEqual(self.open_ranges.pop(), name)
                self.events.append(("range_exit", name))

        self.helper.make_profiler = make_profiler
        self.helper.npu_range = npu_range

    def assert_restored(self):
        for owner, name, existed, value in self.ownership:
            self.test.assertEqual(name in vars(owner), existed)
            if existed:
                self.test.assertIs(vars(owner)[name], value)


class BoundedRoundChecks(unittest.TestCase):
    def capture(self, fixture, directory, limit=2):
        return RoundCapture(directory, 3, limit, fixture.target)

    def test_limit_marks_real_calls_and_restores_before_profiler_exit(self):
        fixture = Fixture(self)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            self.assertEqual(fixture.target.verify(2, position=3, factor=4), 20)
            capture.start()
            self.assertTrue(capture.active)
            for index in range(2):
                with capture.round():
                    value = fixture.target.verify(index, position=2, factor=3)
                    self.assertEqual(
                        fixture.target.host_tokens(value, suffix=(7,)),
                        (3 * (index + 2), 7),
                    )
                self.assertEqual(capture.completed, index + 1)
            self.assertFalse(capture.active)
            self.assertTrue(capture.closed)
            fixture.assert_restored()
            count = len(fixture.events)
            with capture.round():
                self.assertEqual(fixture.target.verify(4, position=2), 6)
            capture.close()
            self.assertEqual(len(fixture.events), count)
            receipt = capture.receipt()
            self.assertEqual(
                [
                    entry["round_index"]
                    for entry in receipt["ranges"]
                    if entry["name"] == "trt.round"
                ],
                [0, 1],
            )
            self.assertEqual(
                {entry["name"] for entry in receipt["ranges"]},
                {
                    "trt.round",
                    "trt.verify",
                    "trt.metadata",
                    "trt.head",
                    "trt.host_tokens",
                },
            )
            self.assertTrue(
                all(
                    row["completed"] and row["end_ns"] >= row["start_ns"]
                    for row in receipt["ranges"]
                )
            )
            self.assertEqual(Path(receipt["path"]), Path(directory) / "capture/rank-3")
            for name, expected in receipt["source_sha256"].items():
                self.assertEqual(
                    hashlib.sha256(Path(name).read_bytes()).hexdigest(), expected
                )
            self.assertFalse(receipt["performance_measured"])
            self.assertFalse(receipt["formal_precision_gate_passed"])
            json.dumps(receipt, allow_nan=False)

    def test_finish_before_limit_closes_without_an_extra_round(self):
        fixture = Fixture(self)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            capture.start()
            with capture.round():
                fixture.target.verify(1)
            capture.close()
            self.assertEqual(
                (capture.completed, capture.closed, capture.active), (1, True, False)
            )
            fixture.assert_restored()

    def test_nested_phase_exception_unwinds_all_ranges_and_is_not_suppressed(self):
        fixture = Fixture(self)
        failure = RuntimeError("actual metadata failed")

        def fail(value, *, position=0):
            raise failure

        fixture.target._metadata = fail
        fixture.ownership[0] = (fixture.target, "_metadata", True, fail)
        fixture.suppress = True
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            capture.start()
            with self.assertRaises(RuntimeError) as error, capture.round():
                fixture.target.verify(1)
            self.assertIs(error.exception, failure)
            self.assertEqual((capture.completed, capture.closed), (0, True))
            self.assertEqual(fixture.events[-1], ("profile_exit", RuntimeError))
            self.assertTrue(
                all(not row["completed"] for row in capture.receipt()["ranges"])
            )
            fixture.assert_restored()

    def test_failure_outside_round_still_closes_after_parent_phase(self):
        fixture = Fixture(self)

        def fail(value, *, factor=1):
            raise ValueError("head failed")

        fixture.target.model.compute_logits = fail
        fixture.ownership[-1] = (fixture.target.model, "compute_logits", True, fail)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            capture.start()
            with self.assertRaisesRegex(ValueError, "head failed"):
                fixture.target.verify(1)
            self.assertEqual((capture.completed, capture.closed), (0, True))
            self.assertEqual(fixture.open_ranges, [])

    def test_partial_install_failure_restores_already_patched_methods(self):
        fixture = Fixture(self)
        fixture.target.model = ModuleType("missing_head")
        fixture.ownership[-1] = (fixture.target.model, "compute_logits", False, None)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            with self.assertRaises(AttributeError):
                capture.start()
            self.assertTrue(capture.closed)
            fixture.assert_restored()

    def test_stop_failure_propagates_and_is_recorded_after_restoration(self):
        fixture = Fixture(self)
        fixture.cleanup_error = OSError("profiler stop failed")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory, limit=1)
            capture.start()
            with self.assertRaises(OSError) as caught, capture.round():
                fixture.target.verify(1)
            self.assertIs(caught.exception, fixture.cleanup_error)
            self.assertEqual((capture.completed, capture.closed), (1, True))
            self.assertEqual(
                capture.receipt()["error"], "OSError: profiler stop failed"
            )
            fixture.assert_restored()

    def test_failed_profiler_start_never_claims_started_or_leaves_hooks(self):
        fixture = Fixture(self)

        def fail(path, rank):
            raise RuntimeError("profiler construction failed")

        fixture.helper.make_profiler = fail
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            with self.assertRaisesRegex(RuntimeError, "profiler construction failed"):
                capture.start()
            self.assertEqual(
                (capture.started, capture.closed, capture.active), (False, True, False)
            )
            self.assertEqual(capture.receipt()["ranges"], [])
            fixture.assert_restored()

    def test_cleanup_failure_keeps_original_exception_and_restores_ownership(self):
        fixture = Fixture(self)
        fixture.cleanup_error = OSError("profiler stop failed")
        failure = RuntimeError("round failed")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            capture.start()
            with self.assertRaises(RuntimeError) as caught, capture.round():
                raise failure
            self.assertIs(caught.exception, failure)
            self.assertIs(caught.exception.__cause__, fixture.cleanup_error)
            fixture.assert_restored()

    def test_explicit_close_inside_round_waits_for_range_end(self):
        fixture = Fixture(self)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            capture.start()
            with capture.round():
                capture.close()
                self.assertFalse(capture.closed)
            self.assertEqual((capture.completed, capture.closed), (1, True))
            fixture.assert_restored()

    def test_restart_and_nested_rounds_are_rejected(self):
        fixture = Fixture(self)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"profile_native_session": fixture.helper}),
        ):
            capture = self.capture(fixture, directory)
            capture.start()
            with (
                self.assertRaisesRegex(RuntimeError, "cannot nest"),
                capture.round(),
                capture.round(),
            ):
                pass
            self.assertTrue(capture.closed)
            with self.assertRaisesRegex(RuntimeError, "twice"):
                capture.start()


if __name__ == "__main__":
    unittest.main()
