"""Optional host timing for successful token callbacks and TP commands."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter_ns


def timing_path(output_dir, rid):
    name = hashlib.sha256(rid.encode("utf-8")).hexdigest() + ".json"
    return Path(output_dir) / name


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                      indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_timing(output_dir, rid):
    report = json.loads(timing_path(output_dir, rid).read_text())
    if report.get("schema_version") != 1 or report.get("rid") != rid:
        raise ValueError("Timing report has a different schema or request")
    return report


def token_metrics(token_ids, timestamps_ns):
    if len(token_ids) != len(timestamps_ns):
        raise ValueError("Token IDs and timestamps have different lengths")
    if any(type(value) is not int or value < 0 for value in timestamps_ns):
        raise ValueError("Token timestamps must be nonnegative integers")
    intervals = [right - left for left, right in
                 zip(timestamps_ns, timestamps_ns[1:])]
    if any(value < 0 for value in intervals):
        raise ValueError("Token timestamps are not monotonic")
    span = timestamps_ns[-1] - timestamps_ns[0] if timestamps_ns else 0
    return {
        "output_tokens": len(token_ids), "span_ns": span,
        "tpot_ms": span / len(intervals) / 1e6 if intervals else None,
        "intervals_ns": intervals,
        "zero_intervals": sum(value == 0 for value in intervals),
    }


class TimedBackend:
    """Time the existing rank-zero command facade without device events."""

    def __init__(self, backend, clock=perf_counter_ns):
        self.backend = backend
        self.clock = clock
        self.operations = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _call(self, operation, *args):
        entry = {"operation": operation, "start_ns": self.clock(),
                 "position_before": self.backend.position}
        self.operations.append(entry)
        try:
            result = getattr(self.backend, operation)(*args)
        except BaseException as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            entry["end_ns"] = self.clock()
            entry["position_after"] = self.backend.position
        if operation == "advance_mtp":
            commit = result.commit
            entry.update(
                width=commit.width, accepted=commit.committed,
                matched_drafts=commit.matched_drafts,
                natural_accepted=commit.natural_accepted,
                accepted_tokens=list(commit.accepted_tokens),
            )
        elif operation == "advance_mtp_n":
            commits = [item.commit for item in result]
            entry.update(
                rounds=len(commits),
                accepted=sum(commit.committed for commit in commits),
                matched_drafts=sum(commit.matched_drafts
                                   for commit in commits),
                natural_accepted=sum(commit.natural_accepted
                                     for commit in commits),
                accepted_tokens=[token for commit in commits
                                 for token in commit.accepted_tokens],
            )
        return result

    def inject(self, request):
        self.operations = []
        return self._call("inject", request)

    def step(self, token):
        return self._call("step", token)

    def start_mtp(self, first_token, max_tokens, width, eos_tokens):
        return self._call("start_mtp", first_token, max_tokens, width,
                          eos_tokens)

    def advance_mtp(self):
        return self._call("advance_mtp")

    def advance_mtp_n(self, count):
        return self._call("advance_mtp_n", count)

    def finish_mtp(self, retained_position):
        return self._call("finish_mtp", retained_position)

    def reset(self):
        return self._call("reset")


class TimedEngine:
    """Persist callback completion times separately from HTTP delivery."""

    def __init__(self, engine, timed_backend, output_dir,
                 clock=perf_counter_ns, metadata=None):
        self.engine = engine
        self.timed_backend = timed_backend
        self.output_dir = Path(output_dir)
        self.clock = clock
        self.metadata = dict(metadata or {})
        self.report = None
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _save(self):
        self.report["operations"] = self.timed_backend.operations
        atomic_json(timing_path(self.output_dir, self.report["rid"]),
                    self.report)

    def inject(self, request):
        self.report = {
            "schema_version": 1, "rid": request.request_id,
            "seq_len": request.seq_len, "clock": "perf_counter_ns",
            "scope": "rank0 successful callback completion; host wall time",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime": self.metadata,
            "timing_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "token_ids": [], "callback_ns": [], "status": "injected",
        }
        try:
            return self.engine.inject(request)
        except BaseException as error:
            self.report.update(status="error", stage="inject",
                               error=f"{type(error).__name__}: {error}")
            self._save()
            raise

    def decode(self, first_token_id, max_tokens, on_token=None, **kwargs):
        if self.report is None:
            raise RuntimeError("Inject a request before timing decode")
        self.report.update(first_token_id=first_token_id,
                           max_tokens=max_tokens, status="decoding",
                           decode_start_ns=self.clock())

        def delivered(token, *args):
            result = on_token(token, *args) if on_token is not None else None
            timestamp = self.clock()
            self.report["token_ids"].append(int(token))
            self.report["callback_ns"].append(timestamp)
            return result

        failure = None
        try:
            tokens = self.engine.decode(
                first_token_id, max_tokens=max_tokens, on_token=delivered,
                **kwargs,
            )
            self.report["returned_token_ids"] = list(tokens)
            self.report["status"] = "complete"
            return tokens
        except BaseException as error:
            failure = error
            self.report.update(status="error", stage="decode",
                               error=f"{type(error).__name__}: {error}")
            raise
        finally:
            self.report["decode_end_ns"] = self.clock()
            self.report["engine_stats"] = dict(self.engine.last_stats)
            self.report["producer"] = token_metrics(
                self.report["token_ids"], self.report["callback_ns"],
            )
            try:
                self._save()
            except OSError as error:
                if failure is None:
                    raise
                failure.add_note(f"Saving decode timing also failed: {error}")

    def reset(self):
        try:
            result = self.engine.reset()
        except BaseException as error:
            if self.report is not None:
                self.report["reset_error"] = f"{type(error).__name__}: {error}"
                self._save()
            raise
        if self.report is not None:
            self.report["reset_complete"] = True
            self._save()
            self.report = None
        return result
