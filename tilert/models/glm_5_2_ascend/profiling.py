"""Bounded decode capture over the production TP16 request loop."""

from contextlib import contextmanager, nullcontext
from dataclasses import replace
from time import perf_counter_ns
import os
import threading

from .tp_runtime import CoordinatedBackend, RankExecutor
from tilert.pd_vllm.ascend_cache import AscendRankEngine
from tilert.pd_vllm.ascend_server import create_session


TORCH_NPU_COMMIT = "fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d"
PROFILER_SOURCES = {
    "profiler/profiler.py":
        "1a04902572383b74a401449e73929b2c4ce728d6c81bb59341876b03c7a82168",
    "profiler/profiler_interface.py":
        "8183b6470d552c33907784252f4009050aa7066955371c460232bc60b8e0af65",
    "profiler/experimental_config.py":
        "08e71971c0251a3cc2ba2705aa699def42922e7cb8ab2752325e9e5dd2f97579",
}
PROFILER_CONTRACT = {
    "torch_npu": "2.12.0", "git_commit": TORCH_NPU_COMMIT,
    "cann": "9.2.0",
    "msprof_sha256":
        "fa65082c08d82eb094d2083966fcac46a7866e4e2f54d91e4f0bb5edab0e733b",
    "level": 1, "aic_metrics": "AiCoreNone", "analyse": False,
    "level_flags": ["TASK_TIME", "ACL_API", "HCCL_TRACE", "AICORE_METRICS"],
    "profiler_mgr.h_sha256":
        "a10d913c3ac15a89e1bd2a8a61104bee4afffb3457b8163288b6ba383498c0d7",
    "anchors": {
        "csrc/profiler/profiler_mgr.h:15-17": "Level1 includes HCCL_TRACE",
        "csrc/profiler/profiler_mgr.cpp:232-246": "current device only",
        "csrc/profiler/npu_profiler.cpp:256-260": "thread-local CPU callback",
        "profiler/profiler_interface.py:163-170": "stop synchronizes NPU",
        "profiler/profiler.py:180-196": "deferred trace analysis",
    },
}


def reject_msprof_environment(environment=None):
    environment = os.environ if environment is None else environment
    active = [name for name in ("PROFILER_SAMPLECONFIG", "PROFILING_MODE")
              if environment.get(name)]
    if active:
        raise RuntimeError("torch_npu profiler is disabled by "
                           + ", ".join(active))


class StageMarkers:
    """Record stage identity in the same thread that calls the backend."""

    def __init__(self, rank, range_factory=nullcontext):
        self.rank = rank
        self.range_factory = range_factory
        self.phase = "warmup"
        self.events = []

    @contextmanager
    def span(self, name, **fields):
        event = {
            "name": name, "rank": self.rank, "phase": self.phase,
            "thread_id": threading.get_ident(),
            "native_thread_id": threading.get_native_id(), **fields,
        }
        # The clock makes this stream readable without a profiler attached,
        # which matters because a profiler adds host work to a device the host
        # feeds and so changes the durations it reports.
        self.events.append({**event, "event": "begin",
                            "ts_ns": perf_counter_ns()})
        try:
            with self.range_factory("tilert." + name):
                yield
        finally:
            self.events.append({**event, "event": "end",
                                "ts_ns": perf_counter_ns()})


class _Proxy:
    def __init__(self, wrapped, state):
        self.wrapped = wrapped
        self.state = state

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class _TraceState:
    def __init__(self, markers):
        self.markers = markers
        self.mode = None
        self.round = 0
        self.verified = False
        self.commit = None


class _Target(_Proxy):
    def verify(self, tokens, **kwargs):
        name = "prompt_replay" if self.state.mode == "inject" else "verify"
        with self.state.markers.span(name, rows=len(tokens),
                                     position=self.position):
            result = self.wrapped.verify(tokens, **kwargs)
        if self.state.mode == "round":
            self.state.verified = True
        return result


class _Draft(_Proxy):
    def forward(self, tokens, parent_hidden, **kwargs):
        if kwargs.get("external_indices") is not None:
            name = "chain"
        else:
            name = "conditioned" if self.state.verified else "seed"
        with self.state.markers.span(name, rows=len(tokens),
                                     position=self.position):
            result = self.wrapped.forward(tokens, parent_hidden, **kwargs)
        if name == "conditioned":
            # The caller resolves acceptance and truncates both caches next.
            self.state.commit = self.state.markers.span("commit")
            self.state.commit.__enter__()
        return result


class _Rank(_Proxy):
    def inject(self, prefix):
        self.state.mode = "inject"
        self.state.round = 0
        try:
            with self.state.markers.span("inject"):
                return self.wrapped.inject(prefix)
        finally:
            self.state.mode = None

    def step(self, token):
        with self.state.markers.span("step", position=self.position):
            return self.wrapped.step(token)

    def advance_mtp(self):
        self.state.mode = "round"
        self.state.round += 1
        self.state.verified = False
        try:
            with self.state.markers.span("round", number=self.state.round,
                                         position=self.position):
                try:
                    return self.wrapped.advance_mtp()
                finally:
                    if self.state.commit is not None:
                        self.state.commit.__exit__(None, None, None)
                        self.state.commit = None
        finally:
            self.state.mode = None

    def advance_mtp_n(self, count):
        results = []
        for _ in range(count):
            results.append(self.advance_mtp())
            if results[-1].commit.finish_reason is not None:
                break
        return results

    def finish_mtp(self, retained_position):
        with self.state.markers.span("finish", retained=retained_position):
            return self.wrapped.finish_mtp(retained_position)

    def reset(self):
        with self.state.markers.span("reset"):
            return self.wrapped.reset()


def instrument_engine(target, draft, markers, *,
                      engine_class=AscendRankEngine):
    """Wrap calls without replacing verification, acceptance or cache logic."""
    state = _TraceState(markers)
    engine = engine_class(_Target(target, state), _Draft(draft, state))
    return _Rank(engine, state)


def capture_phase(profiler, markers, barrier, request_runner):
    """Start every rank before decode and stop after its command loop exits."""
    with profiler:
        with markers.span("capture_ready"):
            barrier()
        result = request_runner()
        with markers.span("capture_complete"):
            barrier()
    return result


def run_phase(
    engine, rank, channel, prefix_loader, request, width, max_tokens,
    markers, *, requests=1, stop_ids=(), ignore_eos=False, on_token=None,
    cancel_event=None,
):
    """Run one command-stream phase, closing workers before returning."""
    if not 1 <= max_tokens <= 6 or requests < 1:
        raise ValueError("Capture requests require 1..6 output tokens")
    executor = RankExecutor(engine, rank, prefix_loader)
    if rank != 0:
        executor.serve(channel)
        return {"command_serial": executor.serial, "requests": []}
    backend = CoordinatedBackend(executor, channel)
    session = create_session(backend, width, stop_ids)
    reports = []
    try:
        for index in range(requests):
            current = replace(request,
                              request_id=f"{markers.phase}-{index}")
            session.inject(current)
            delivered = []

            def deliver(token):
                name = "first_token" if not delivered else "token_delivery"
                with markers.span(name, offset=len(delivered)):
                    if on_token is not None:
                        on_token(token)
                    delivered.append(token)

            try:
                with markers.span("decode"):
                    output = session.decode(
                        engine.first_token_id, max_tokens=max_tokens,
                        sampling={"temperature": 0.0,
                                  "ignore_eos": ignore_eos},
                        on_token=deliver,
                        cancel_event=cancel_event,
                    )
                if output != delivered:
                    raise RuntimeError("Returned tokens differ from delivery")
                reports.append({"tokens": output, **session.last_stats})
            finally:
                if not backend.failed:
                    session.reset()
    finally:
        if not backend.failed:
            backend.close()
    return {"command_serial": executor.serial, "requests": reports}
