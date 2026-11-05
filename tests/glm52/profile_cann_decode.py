"""Capture bounded CANN sessions after all ranks pass numerical validation."""

from collections import Counter
import inspect
import json
from pathlib import Path

from profile_native_session import (
    digest, make_profiler, npu_range, profiler_identity, trace_inventory,
)
from tilert.models.glm_5_2_ascend.profiling import (
    StageMarkers, capture_phase, instrument_engine, reject_msprof_environment,
    run_phase,
)
from tilert.models.glm_5_2_ascend.session import AscendPDEngine, PrefixRequest
from tilert.models.glm_5_2_ascend.tp_runtime import TorchControlChannel
from tilert.pd_vllm.ascend_server import create_session
from tilert.pd_vllm.ascend_cache import CannRankEngine


class _MarkedModel:
    def __init__(self, model, markers, first_token):
        self.model = model
        self.markers = markers
        self.first_token_id = first_token

    def __getattr__(self, name):
        return getattr(self.model, name)

    def inject(self, prefix):
        with self.markers.span("inject"):
            return self.model.inject(prefix)

    def step(self, token):
        with self.markers.span("step", position=self.position, token=token):
            return self.model.step(token)

    def reset(self):
        with self.markers.span("reset"):
            return self.model.reset()


def capture_validated_model(engine, prefixes, first_token, rank,
                            directory, *, with_stack=False):
    """Profile the caller's accepted model without reloading or prompt replay.

    All sixteen ranks must call this after their numerical checks pass.
    The caller binds those reports and the frozen prefix to this receipt.
    """
    import torch
    import torch.distributed as distributed
    import torch_npu
    from vllm.distributed import get_world_group

    reject_msprof_environment()
    if (len(prefixes) != 78 or prefixes[0]["latent"].shape[0] != 8192
            or engine.capacity < 8194):
        raise ValueError("Capture requires full-P 8K KV and two decode rows")
    if type(first_token) is not int or not 0 <= first_token < 154880:
        raise ValueError("Capture requires the prefill vocabulary token")
    if distributed.get_rank() != rank or distributed.get_world_size() != 16:
        raise ValueError("Capture requires all sixteen TP ranks")
    identity = profiler_identity(torch_npu)
    directory = Path(directory)
    raw = directory / f"raw-rank-{rank}"
    raw.mkdir(parents=True, exist_ok=False)
    markers = StageMarkers(rank)
    model = _MarkedModel(engine, markers, first_token)
    channel = TorchControlChannel(get_world_group().cpu_group)
    request = PrefixRequest("capture", 8192)
    phase_args = (model, rank, channel, lambda request, rank: prefixes,
                  request, 1, 3, markers)

    def barrier():
        distributed.barrier(group=channel.group)

    with torch.npu.device(engine.device), torch.inference_mode():
        warmup = run_phase(*phase_args, requests=3, ignore_eos=True)
        torch.npu.synchronize(engine.device)
        markers.phase = "capture"
        markers.range_factory = npu_range
        captured = capture_phase(
            make_profiler(raw, rank, with_stack=with_stack), markers, barrier,
            lambda: run_phase(*phase_args, ignore_eos=True),
        )
    consistent = [True]
    if rank == 0:
        records = warmup["requests"] + captured["requests"]
        expected = records[0]["tokens"]
        consistent[0] = (len(expected) == 3 and all(
            item["tokens"] == expected and item["step_calls"] == 2
            and item["committed_position"] == 8194 for item in records
        ))
    distributed.broadcast_object_list(consistent, src=0, group=channel.group)
    objects = (run_phase, make_profiler, TorchControlChannel,
               AscendPDEngine, create_session)
    sources = {str(Path(inspect.getfile(value)).resolve()): digest(
        Path(inspect.getfile(value))) for value in objects}
    sources[str(Path(__file__).resolve())] = digest(Path(__file__))
    report = {
        "status": "raw_capture" if consistent[0] else "token_mismatch",
        "rank": rank, "world_size": 16, "device": str(engine.device),
        "scope": "Full-P CANN M1 session; HTTP and HIXL excluded",
        "profiled": True, "tpot_measured": False, "mtp_width": 1,
        "initial_position": 8192, "first_token_id": first_token,
        "max_tokens": 3, "ignore_eos": True,
        "profiler_api": identity, "warmup": warmup, "capture": captured,
        "with_stack": with_stack,
        "markers": markers.events, "trace_directory": str(raw),
        "source_sha256": {**engine.source_metadata(), **sources},
        "trace_files": trace_inventory(raw),
    }
    (directory / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    if not consistent[0]:
        raise RuntimeError("Profiled request differs from warmup replay")
    return report


def capture_validated_mtp(target, draft, prefix, first_token, rank,
                          directory, width, *, with_stack=False):
    """Profile the accepted objects and their existing target graph.

    Six outputs require at most five rounds. The frozen 8K input uses
    three; the receipt records the actual count and stage markers.

    ``width`` is the verify width under test. It used to be a literal 4 here
    and in the precondition, which made the capture refuse any other width --
    and the width that now holds the best point is 8.
    """
    import torch
    import torch.distributed as distributed
    import torch_npu
    from vllm.distributed import get_world_group

    reject_msprof_environment()
    if (prefix.seq_len != 8192 or target.capacity < 8200
            or distributed.get_rank() != rank
            or distributed.get_world_size() != 16
            or (target.graph is not None and target.graph.width != width)):
        raise ValueError("Capture requires TP16, full 8K and the tested width")
    identity = profiler_identity(torch_npu)
    directory = Path(directory)
    raw = directory / f"raw-rank-{rank}"
    raw.mkdir(parents=True, exist_ok=False)
    markers = StageMarkers(rank)
    engine = instrument_engine(target, draft, markers,
                               engine_class=CannRankEngine)
    channel = TorchControlChannel(get_world_group().cpu_group)
    phase_args = (engine, rank, channel, lambda request, rank: prefix,
                  PrefixRequest("capture-mtp", 8192), width, 6, markers)

    def barrier():
        distributed.barrier(group=channel.group)

    with torch.npu.device(target.device), torch.inference_mode():
        warmup = run_phase(*phase_args, requests=3, ignore_eos=True)
        torch.npu.synchronize(target.device)
        markers.phase, markers.range_factory = "capture", npu_range
        captured = capture_phase(
            make_profiler(raw, rank, with_stack=with_stack), markers, barrier,
            lambda: run_phase(*phase_args, ignore_eos=True),
        )
    counts = Counter(event["name"] for event in markers.events
                     if event["phase"] == "capture"
                     and event["event"] == "begin")
    rounds = counts["round"]
    stages = {name: counts[name] for name in
              ("round", "seed", "chain", "verify", "conditioned", "commit")}
    local_ok = (1 <= rounds <= 5 and counts["seed"] == 1
                and counts["chain"] == 2 * rounds
                and all(counts[name] == rounds for name in
                        ("verify", "conditioned", "commit"))
                and target.position == draft.position == 0)
    ranks = [None] * 16
    distributed.all_gather_object(ranks, {"rank": rank, "stages": stages,
                                        "passed": local_ok},
                                  group=channel.group)
    consistent = [all(item["passed"] and item["stages"] == stages
                      for item in ranks)]
    if rank == 0:
        records = warmup["requests"] + captured["requests"]
        expected = records[0]["tokens"]
        consistent[0] &= (len(expected) == 6 and expected[0] == first_token
                          and all(item["tokens"] == expected
                                  and item["accepted"] == 5
                                  and item["committed_position"] == 8197
                                  and item["finish_reason"] == "length"
                                  and item["finish_error"] is None
                                  for item in records)
                          and captured["requests"][0]["round_calls"] == rounds)
    distributed.broadcast_object_list(consistent, src=0, group=channel.group)
    objects = (capture_validated_mtp, instrument_engine, make_profiler,
               CannRankEngine, type(draft), type(draft.model))
    sources = {str(Path(inspect.getfile(value)).resolve()): digest(
        Path(inspect.getfile(value))) for value in objects}
    report = {
        "status": "raw_capture" if consistent[0] else "session_mismatch",
        "rank": rank, "world_size": 16, "device": str(target.device),
        "scope": "Full-P CANN M4 session; HTTP and HIXL excluded",
        "profiled": True, "tpot_measured": False, "mtp_width": 4,
        "max_tokens": 6, "initial_position": 8192, "ignore_eos": True,
        "first_token_id": first_token, "prompt_replayed": False,
        "target_graph": target.graph is not None,
        "draft_execution": "eager" if draft.graph is None else "chain_graph",
        "actual_rounds": rounds, "rank_stages": ranks,
        "profiler_api": identity, "warmup": warmup, "capture": captured,
        "with_stack": with_stack,
        "markers": markers.events, "trace_directory": str(raw),
        "source_sha256": {**target.source_metadata(), **sources},
        "trace_files": trace_inventory(raw),
    }
    (directory / f"rank-{rank}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    if not consistent[0]:
        raise RuntimeError("M4 capture differs from warmup or rank stages")
    return report
