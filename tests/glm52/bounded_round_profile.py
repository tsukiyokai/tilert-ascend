"""Bound CPU/NPU profiling to complete rounds of an existing rank engine."""

import hashlib
import inspect
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class RoundCapture:
    """Temporarily mark the caller's methods and close after complete rounds."""

    def __init__(self, output, rank, limit, target):
        if type(limit) is not int or limit < 1:
            raise ValueError("Round capture requires a positive integer limit")
        self.path = Path(output) / "capture" / f"rank-{rank}"
        self.rank = rank
        self.limit = limit
        self.target = target
        self.started = False
        self.closed = False
        self.active = False
        self.completed = 0
        self.ranges = []
        self._stack = None
        self._range = None
        self._depth = 0
        self._round_open = False
        self._close_requested = False
        self._error = None
        self._sources = {str(Path(__file__).resolve()): digest(__file__)}

    def start(self):
        """Enter the existing CPU/NPU profiler before installing markers."""
        if self.started or self.closed:
            raise RuntimeError("Round capture cannot be started twice")
        from profile_native_session import make_profiler, npu_range

        self._range = npu_range
        self._stack = ExitStack()
        try:
            for function in (make_profiler, npu_range):
                path = Path(inspect.getfile(inspect.unwrap(function))).resolve()
                self._sources[str(path)] = digest(path)
            self.path.mkdir(parents=True, exist_ok=False)
            self._stack.enter_context(make_profiler(self.path, self.rank))
            self.started = True
            for owner, name, phase in (
                (self.target, "_metadata", "trt.metadata"),
                (self.target, "verify", "trt.verify"),
                (self.target, "host_tokens", "trt.host_tokens"),
                (self.target.model, "compute_logits", "trt.head"),
            ):
                self._install(owner, name, phase)
            self.active = True
        except BaseException as error:
            self.close(error)
            raise

    def _install(self, owner, name, phase):
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            with self._span(phase):
                return original(*args, **kwargs)

        self._stack.enter_context(patch.object(owner, name, wrapped))

    @contextmanager
    def _span(self, name):
        entry = {
            "name": name,
            "round_index": self.completed if self._round_open else None,
            "start_ns": perf_counter_ns(),
            "completed": False,
        }
        if name == "trt.round":
            entry["index"] = self.completed
        self.ranges.append(entry)
        self._depth += 1
        try:
            with self._range(name):
                yield
            entry["completed"] = True
        except BaseException as error:
            entry["error"] = type(error).__name__ + ": " + str(error)
            self._close_requested = True
            self._error = error
            raise
        finally:
            entry["end_ns"] = perf_counter_ns()
            self._depth -= 1
            if not self._depth and self._close_requested:
                self.close(self._error)

    @contextmanager
    def round(self):
        """Mark one complete execution round; closed captures pass through."""
        if not self.active:
            yield
            return
        if self._round_open:
            raise RuntimeError("Round capture cannot nest execution rounds")
        self._round_open = True
        try:
            with self._span("trt.round"):
                yield
            self.completed += 1
        finally:
            self._round_open = False
        if self.completed >= self.limit:
            self.close()

    def close(self, error=None):
        """Restore ownership and stop profiling after all active ranges exit."""
        if self.closed:
            return
        if error is not None:
            self._error = error
        if self._depth:
            self._close_requested = True
            return
        self.active = False
        self.closed = True
        if self._stack is not None:
            stack, self._stack = self._stack, None
            error = self._error
            try:
                stack.__exit__(
                    type(error) if error is not None else None,
                    error,
                    error.__traceback__ if error is not None else None,
                )
            except BaseException as cleanup_error:
                if error is not None:
                    raise error from cleanup_error
                self._error = cleanup_error
                raise

    def receipt(self):
        """Describe local raw capture coverage without parsing or timing it."""
        return {
            "rank": self.rank,
            "started": self.started,
            "closed": self.closed,
            "active": self.active,
            "completed": self.completed,
            "limit": self.limit,
            "ranges": [dict(entry) for entry in self.ranges],
            "path": str(self.path),
            "source_sha256": dict(self._sources),
            "error": (
                None
                if self._error is None
                else type(self._error).__name__ + ": " + str(self._error)
            ),
            "clock": "perf_counter_ns; local host ranges include instrumentation",
            "scope": "Existing rank-local complete rounds; CPU+NPU+MSTX; no collective or raw parsing added",
            "raw_parsed": False,
            "performance_measured": False,
            "formal_precision_gate_passed": False,
        }
