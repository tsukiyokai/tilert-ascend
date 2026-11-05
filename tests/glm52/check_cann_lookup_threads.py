"""Check sixteen CPU command workers with scalar targets and graph markers.

The command executors and serving lifecycle are real. Queues replace the
control transport; synthetic graph markers exercise capture ownership and
ordering without running torch, an ACL graph, HIXL or an NPU.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from queue import Queue
import sys
from threading import Barrier, BrokenBarrierError, Event, Thread
from time import monotonic
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from check_cann_pd import FullTarget, full_prefix  # noqa: E402
from check_mtp_tp import serial_oracle  # noqa: E402
from serve_cann_pd import ServingRank  # noqa: E402
from tilert.models.glm_5_2_ascend.session import PrefixRequest  # noqa: E402
from tilert.models.glm_5_2_ascend.tp_runtime import (  # noqa: E402
    CoordinatedBackend, RankExecutor, TP_SIZE,
)
from tilert.pd_vllm.ascend_server import create_session  # noqa: E402


class ScalarGraphTarget(FullTarget):
    def __init__(self):
        super().__init__(128)
        self.graph = None
        self.events = []

    def inject(self, prefix):
        super().inject(prefix)
        self.events.append({"event": "inject", "position": self.position})

    def predict(self, parent):
        result = super().predict(parent)
        self.events.append({"event": "predict", "position": self.position,
                            "token": int(result.token_ids[0])})
        return result

    def verify(self, tokens, **kwargs):
        self.events.append({"event": "verify", "position": self.position,
                            "tokens": list(tokens)})
        assert self.position + len(tokens) <= self.capacity
        return super().verify(tokens, **kwargs)

    def host_tokens(self, *groups):
        assert len(groups) == 1
        self.events.append({"event": "host_read", "groups": len(groups)})
        return super().host_tokens(*groups)

    def truncate(self, position):
        super().truncate(position)
        self.events.append({"event": "truncate", "position": position})

    def enable_graph(self, width):
        self.events.append({"event": "capture_marker", "width": width})
        self.graph = SimpleNamespace(width=width)


class ObservedRank(ServingRank):
    """Retain each worker's accepted stream without changing its commands."""

    def start_mtp(self, first_token, *args):
        super().start_mtp(first_token, *args)
        self.output_tokens = [first_token]

    def advance_mtp(self):
        result = super().advance_mtp()
        self.output_tokens.extend(result.commit.accepted_tokens)
        assert self.position == self._mtp.state.main_length
        return result


def load_prefix(request, rank):
    prompt = request.metadata["prompt_token_ids"]
    prefix = full_prefix(PrefixRequest(request.request_id, len(prompt),
                                       {"prompt": prompt}), rank)
    return replace(prefix, prompt_token_ids=tuple(prompt))


class StopWorker(Exception):
    pass


class WorkerChannel:
    def __init__(self, commands, stopped):
        self.commands = commands
        self.stopped = stopped

    def receive(self):
        if self.stopped.is_set():
            raise StopWorker
        command = self.commands.get()
        if command is None:
            raise StopWorker
        return command


class Cohort:
    def __init__(self, timeout, *, barrier=None):
        self.timeout = timeout
        self.barrier = (Barrier(TP_SIZE, timeout=timeout)
                        if barrier is None else barrier)
        self.stopped = Event()
        self.errors = []
        self.commands = [Queue() for _ in range(TP_SIZE - 1)]
        self.ranks = []
        for rank in range(TP_SIZE):
            target = ScalarGraphTarget()
            engine = ObservedRank(
                target, None, width=32, target_graph=True,
                barrier=lambda target=target: self.meet(target),
            )
            engine.on_inject = lambda target=target: target.events.append(
                {"event": "injected", "position": target.position},
            )
            self.ranks.append(RankExecutor(engine, rank, load_prefix))
        self.backend = CoordinatedBackend(self.ranks[0], self)
        self.workers = [Thread(
            target=self.serve, args=(rank, commands), daemon=True,
            name=f"lookup-cpu-rank-{rank.rank}",
        ) for rank, commands in zip(self.ranks[1:], self.commands)]
        for worker in self.workers:
            worker.start()

    def meet(self, target):
        target.events.append({"event": "barrier"})
        self.barrier.wait()

    def send(self, command):
        for commands in self.commands:
            commands.put(command)

    def serve(self, rank, commands):
        try:
            rank.serve(WorkerChannel(commands, self.stopped))
        except StopWorker:
            pass
        except BaseException as error:
            self.errors.append({"rank": rank.rank,
                                "error": type(error).__name__,
                                "message": str(error)})

    def wait(self):
        """Wait only for an audit snapshot, outside the serving commands."""
        serial = self.ranks[0].serial
        deadline = monotonic() + self.timeout
        while any(rank.serial != serial for rank in self.ranks):
            if self.errors:
                raise RuntimeError(self.errors)
            if monotonic() >= deadline:
                raise TimeoutError(f"Workers did not complete serial {serial}")
            self.stopped.wait(0.001)

    def close(self):
        self.stopped.set()
        self.barrier.abort()
        for commands in self.commands:
            commands.put(None)
        deadline = monotonic() + self.timeout
        for worker in self.workers:
            worker.join(max(0, deadline - monotonic()))
        alive = [worker.name for worker in self.workers if worker.is_alive()]
        if alive:
            raise RuntimeError(f"CPU workers did not stop: {alive}")


def check_injection(events, prompt, first, capture):
    names = [event["event"] for event in events]
    expected = ["inject", "predict"]
    if capture:
        expected += ["verify", "host_read", "truncate"] * 3
        expected += ["barrier", "capture_marker", "verify", "host_read",
                     "truncate", "barrier"]
    assert names == [*expected, "injected"], names
    for event in events:
        if "position" in event:
            assert event["position"] == len(prompt), event
        if event["event"] == "verify":
            assert event["tokens"] == [first] * 32
        if event["event"] == "capture_marker":
            assert event["width"] == 32


def run_lifecycle(report, timeout):
    cohort = Cohort(timeout)
    session = create_session(cohort.backend, 32, ())
    report["requests"] = []
    report["rank_events"] = {str(rank.rank): rank.engine.target.events
                             for rank in cohort.ranks}
    prompts = ([row % 251 for row in range(127)], [3, 10, 17], [1, 9, 2, 1])
    try:
        for generation, prompt in enumerate(prompts, 1):
            starts = [len(rank.engine.target.events) for rank in cohort.ranks]
            session.inject(PrefixRequest(str(generation), len(prompt),
                                          {"prompt_token_ids": prompt}))
            cohort.wait()
            expected, rows = serial_oracle(prompt, 40, 128)
            for rank, start in zip(cohort.ranks, starts):
                engine = rank.engine
                check_injection(engine.target.events[start:], prompt,
                                expected[0], capture=generation == 2)
                assert engine._request_prompt == tuple(prompt)
                assert engine.position == len(prompt)
                assert engine.target.rows[:engine.position] == load_prefix(
                    PrefixRequest(str(generation), len(prompt),
                                  {"prompt_token_ids": prompt}), rank.rank,
                ).main
            assert session.decode(expected[0], 40) == expected
            cohort.wait()
            for rank in cohort.ranks:
                assert rank.engine.output_tokens == expected
                assert rank.engine.target.rows[:rank.engine.position] == rows
            report["requests"].append({
                "generation": generation, "prompt_token_ids": prompt,
                "expected_output_token_ids": expected,
                "expected_valid_cache_rows": rows,
                "ranks": [{
                    "rank": rank.rank,
                    "output_token_ids": rank.engine.output_tokens,
                    "valid_cache_rows": rank.engine.target.rows[
                        :rank.engine.position],
                } for rank in cohort.ranks],
                "all_rank_outputs_and_caches_match": True,
            })
            session.reset()
            cohort.wait()
            assert all(rank.engine.position == 0
                       and rank.engine._request_prompt is None
                       and rank.engine.parent_hidden is None
                       for rank in cohort.ranks)
        cohort.backend.close()
        cohort.wait()
    finally:
        cohort.close()
        report["worker_errors"] = cohort.errors
        report["all_workers_stopped"] = all(
            not worker.is_alive() for worker in cohort.workers)
    assert not cohort.errors, cohort.errors
    assert all(rank.closed for rank in cohort.ranks)
    for rank in cohort.ranks:
        names = [event["event"] for event in rank.engine.target.events]
        assert names.count("capture_marker") == 1
        assert names.count("barrier") == 2


def run_timeout_control(report, timeout):
    """A missing barrier participant must fail and release every worker."""
    cohort = Cohort(timeout, barrier=Barrier(TP_SIZE + 1, timeout=0.1))
    timed_out = False
    try:
        try:
            cohort.backend.inject(PrefixRequest(
                "timeout", 3, {"prompt_token_ids": [3, 10, 17]},
            ))
        except BrokenBarrierError:
            timed_out = True
    finally:
        cohort.close()
    assert timed_out
    assert cohort.backend.failed
    assert all(error["error"] == "BrokenBarrierError"
               for error in cohort.errors)
    report["timeout_control"] = {
        "barrier_parties": TP_SIZE + 1, "timeout_s": 0.1,
        "failed_as_expected": True, "backend_failed": cohort.backend.failed,
        "worker_errors": cohort.errors,
        "all_workers_stopped": all(not worker.is_alive()
                                   for worker in cohort.workers),
    }


def source_hashes():
    paths = {Path(module.__file__).resolve() for module in sys.modules.values()
             if getattr(module, "__file__", None)}
    return {str(path.relative_to(ROOT)):
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)
            if path.is_relative_to(ROOT) and path.suffix == ".py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "hostname": platform.node(), "ranks": TP_SIZE,
        "scope": "CPU command workers, scalar target and synthetic graph "
                 "markers; no torch, ACLGraph, HIXL or NPU execution",
        "precision_verified": False, "performance_measured": False,
    }
    try:
        run_lifecycle(report, args.timeout)
        run_timeout_control(report, args.timeout)
        report["status"] = "pass"
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
    report["torch_imported"] = "torch" in sys.modules
    report["torch_npu_imported"] = "torch_npu" in sys.modules
    report["source_sha256"] = source_hashes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps({"status": report["status"], "ranks": TP_SIZE,
                      "output": str(args.output)}))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
