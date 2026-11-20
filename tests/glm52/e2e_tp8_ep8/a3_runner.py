from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
from typing import Any
import uuid

from .contracts import (
    AB_BA_FORMAL,
    AB_BA_WARM,
    CaseSpec,
    DECODE_TOKENS,
    DEFAULT_CASE_ROOT,
    DEFAULT_KBENV_BIN,
    DEFAULT_RUNNER_TIMEOUT_S,
    DEFAULT_TRACK,
    EP_SIZE,
    FINAL_POS,
    MIN_FORMALS,
    MIN_WARMUPS,
    SCHEMA_VERSION,
    TARGET_LAYERS,
    TARGET_SFA_TOPK,
    TPOT_TARGET_MS,
    VLLM_FREE_MILESTONES,
    WIDTHS,
    WORLD_SIZE,
    VerificationError,
    file_sha256,
    get_case_spec,
    get_track_spec,
    golden_result_path,
    milestone_tpot_limit,
    tpot_formula_for_world_size,
)
from .hygiene_gate import (
    allocate_free_port_base,
    allocate_isolation_env,
    assert_host_allowed,
    expected_isolation_metadata,
    run_kbenv_precheck,
    select_free_die_slice,
    validate_run_id,
    verify_isolation_env,
)
from .parity_gate import (
    ZeroLagHorizonRef,
    compute_p90,
    select_optimal_w_act,
    verify_e2e_run,
    verify_prompt_case,
)


def _synth_sha256(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def validate_protocol_counts(warmups: int, formals: int, ab_ba: bool) -> tuple[int, int]:
    req_warm = AB_BA_WARM if ab_ba else MIN_WARMUPS
    req_form = AB_BA_FORMAL if ab_ba else MIN_FORMALS
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < req_warm:
        raise VerificationError(
            f"warmups ({warmups}) must be >= {req_warm} for ab_ba={ab_ba}"
        )
    if isinstance(formals, bool) or not isinstance(formals, int) or formals < req_form:
        raise VerificationError(
            f"formals ({formals}) must be >= {req_form} for ab_ba={ab_ba}"
        )
    return warmups, formals


def build_synthetic_cache_state(
    case_id: str,
    num_layers: int,
    kind: str,
) -> dict[str, Any]:
    layers: list[dict[str, dict[str, Any]]] = []
    for layer_idx in range(num_layers):
        layers.append(
            {
                "latent": {
                    "dtype": "torch.bfloat16",
                    "shape": [FINAL_POS, 512],
                    "sha256": _synth_sha256(f"{case_id}:{kind}:latent:{layer_idx}"),
                },
                "rope": {
                    "dtype": "torch.bfloat16",
                    "shape": [FINAL_POS, 64],
                    "sha256": _synth_sha256(f"{case_id}:{kind}:rope:{layer_idx}"),
                },
                "index_keys": {
                    "dtype": "torch.int8",
                    "shape": [FINAL_POS, 128],
                    "sha256": _synth_sha256(f"{case_id}:{kind}:index_keys:{layer_idx}"),
                },
            }
        )
    return {"position": FINAL_POS, "layers": layers}


def _accepted_schedule_for_case(case_id: str) -> list[int]:
    step = 4 if case_id == "case-02-review" else 8 if case_id == "case-03-refactor" else 6
    rounds: list[int] = []
    remaining = DECODE_TOKENS
    while remaining > 0:
        chunk = min(step, remaining)
        rounds.append(chunk)
        remaining -= chunk
    return rounds


def _raw_lps_for_case(case_id: str, width: int) -> list[float]:
    decay = -0.55 if case_id == "case-02-review" else -0.18
    return [0.0 if i == 0 else decay * float(i) for i in range(width)]


def _build_round_traces(
    case_id: str,
    accepted_per_round: list[int],
    width: int,
    min_w_act: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    raw_lps = _raw_lps_for_case(case_id, width)
    w_act = select_optimal_w_act(
        raw_lps,
        width=width,
        min_w_act=min_w_act,
        t_base_ms=46.5,
        c_w_ms=0.68,
    )
    mc2_mask = [idx < w_act for idx in range(width)]
    horizon = ZeroLagHorizonRef()
    traces: list[dict[str, Any]] = []
    total_rounds = len(accepted_per_round)
    for round_idx, acc in enumerate(accepted_per_round):
        is_last = round_idx == total_rounds - 1
        level = horizon.update(acc)
        traces.append(
            {
                "round_idx": round_idx,
                "width": width,
                "min_w_act": min_w_act,
                "t_base_ms": 46.5,
                "c_w_ms": 0.68,
                "raw_lps": raw_lps,
                "w_act": w_act,
                "mc2_mask": mc2_mask,
                "dcci_issued": True,
                "collective_rows": 16,
                "l78_gemm_chunk_rows": 8,
                "kv_commit_q_len": 0 if is_last else acc,
                "kv_commit_wrote": not is_last,
                "done": is_last,
                "drain_q_len": 0 if is_last else acc,
                "final_pos": (
                    FINAL_POS if is_last else 8192 + sum(accepted_per_round[: round_idx + 1])
                ),
                "accepted": acc,
                "horizon_level": level,
            }
        )
    return traces, [w_act]


def _candidate_tpots_for_milestone(milestone: str, count: int) -> list[float]:
    base_by_milestone = {
        "M1": 6.35,
        "M2": 5.45,
        "M3": 3.55,
        "M4": 2.78,
        "M5": 2.78,
    }
    base = base_by_milestone[milestone]
    return [round(base + 0.03 * float(i % 5), 4) for i in range(count)]


def _make_trial_record(
    trial_idx: int,
    is_warmup: bool,
    arm: str,
    target_tpot: float,
    spec: CaseSpec,
    output_tokens: list[int],
    accepted_per_round: list[int],
    pair_idx: int | None = None,
    world_size: int = WORLD_SIZE,
) -> dict[str, Any]:
    rank_dt_ms = [
        round((target_tpot - 0.004 * float(world_size - 1 - r)) * float(DECODE_TOKENS), 6)
        for r in range(world_size)
    ]
    max_tpot = max(rank_dt_ms) / float(DECODE_TOKENS)
    record: dict[str, Any] = {
        "trial_idx": trial_idx,
        "is_warmup": is_warmup,
        "arm": arm,
        "rank_dt_ms": rank_dt_ms,
        "max_tpot_ms": max_tpot,
        "rounds": len(accepted_per_round),
        "accepted_per_round": list(accepted_per_round),
        "mean_accepted": float(DECODE_TOKENS) / float(len(accepted_per_round)),
        "first_token_id": spec.first_token_id,
        "output_token_ids": list(output_tokens),
        "output_sha256": spec.output_sha256,
        "eos_drained_cleanly": True,
    }
    if pair_idx is not None:
        record["pair_idx"] = pair_idx
    return record


def _build_dry_run_trials(
    milestone: str,
    n_warm: int,
    n_form: int,
    ab_ba: bool,
    spec: CaseSpec,
    output_tokens: list[int],
    accepted_per_round: list[int],
    world_size: int = WORLD_SIZE,
) -> list[dict[str, Any]]:
    cand_warm_tpots = _candidate_tpots_for_milestone(milestone, n_warm)
    cand_form_tpots = _candidate_tpots_for_milestone(milestone, n_form)
    trials: list[dict[str, Any]] = []
    trial_idx = 0
    if ab_ba:
        for pair_idx in range(n_warm + n_form):
            is_warm = pair_idx < n_warm
            tpot_c = (
                cand_warm_tpots[pair_idx]
                if is_warm
                else cand_form_tpots[pair_idx - n_warm]
            )
            tpot_b = round(tpot_c + 0.15, 4)
            order = ("baseline", "candidate") if pair_idx % 2 == 0 else ("candidate", "baseline")
            for arm in order:
                trials.append(
                    _make_trial_record(
                        trial_idx=trial_idx,
                        is_warmup=is_warm,
                        arm=arm,
                        target_tpot=tpot_b if arm == "baseline" else tpot_c,
                        spec=spec,
                        output_tokens=output_tokens,
                        accepted_per_round=accepted_per_round,
                        pair_idx=pair_idx,
                        world_size=world_size,
                    )
                )
                trial_idx += 1
        return trials
    for tpot_val in cand_warm_tpots:
        trials.append(
            _make_trial_record(
                trial_idx=trial_idx,
                is_warmup=True,
                arm="candidate",
                target_tpot=tpot_val,
                spec=spec,
                output_tokens=output_tokens,
                accepted_per_round=accepted_per_round,
                world_size=world_size,
            )
        )
        trial_idx += 1
    for tpot_val in cand_form_tpots:
        trials.append(
            _make_trial_record(
                trial_idx=trial_idx,
                is_warmup=False,
                arm="candidate",
                target_tpot=tpot_val,
                spec=spec,
                output_tokens=output_tokens,
                accepted_per_round=accepted_per_round,
                world_size=world_size,
            )
        )
        trial_idx += 1
    return trials


def _write_dry_run_ranks(
    output_dir: Path,
    spec: CaseSpec,
    output_tokens: list[int],
    accepted_per_round: list[int],
    horizon_levels: list[int],
    last_cand: dict[str, Any],
    main_state: dict[str, Any],
    draft_state: dict[str, Any],
    runtime_invariants: dict[str, Any],
    world_size: int = WORLD_SIZE,
) -> list[dict[str, Any]]:
    ranks_manifest: list[dict[str, Any]] = []
    for rank in range(world_size):
        rank_dt = float(last_cand["rank_dt_ms"][rank])
        rank_payload = {
            "rank": rank,
            "world_size": world_size,
            "case_id": spec.case_id,
            "prompt_sha256": spec.prompt_sha256,
            "first_token_id": spec.first_token_id,
            "output_token_ids": output_tokens,
            "output_sha256": spec.output_sha256,
            "rounds": len(accepted_per_round),
            "accepted_per_round": accepted_per_round,
            "horizon_levels": horizon_levels,
            "dt_ms": rank_dt,
            "tpot_ms": rank_dt / float(DECODE_TOKENS),
            "main_state": main_state,
            "draft_state": draft_state,
            "serial_draft_match": True,
            "ekv_serial_diffs": {},
            "runtime_invariants": runtime_invariants,
        }
        rank_file = output_dir / f"rank-{rank}.json"
        rank_file.write_text(json.dumps(rank_payload) + "\n", encoding="utf-8")
        ranks_manifest.append(
            {
                "rank": rank,
                "report_path": rank_file.name,
                "report_sha256": file_sha256(rank_file),
            }
        )
    return ranks_manifest


def build_dry_run_bundle(
    case_id: str,
    output_dir: Path,
    case_root: Path = DEFAULT_CASE_ROOT,
    warmups: int = MIN_WARMUPS,
    formals: int = MIN_FORMALS,
    ab_ba: bool = False,
    milestone: str = "M5",
    width: int = 16,
    min_w_act: int = 10,
    draft_sfa_topk: int = 1024,
    run_id: str | None = None,
    port_base: int | None = 29900,
    track: str = DEFAULT_TRACK,
    output_tokens: list[int] | None = None,
) -> Path:
    tspec = get_track_spec(track)
    spec = get_case_spec(case_id, track=tspec.track)
    ws = tspec.world_size
    if width not in WIDTHS:
        raise VerificationError(f"Unsupported width: {width} (expected in {WIDTHS})")
    n_warm, n_form = validate_protocol_counts(warmups, formals, ab_ba)
    case_dir = case_root / case_id
    verify_prompt_case(case_dir, spec, track=tspec.track)
    if output_tokens is None:
        result_path = golden_result_path(case_dir, tspec.track)
        if not result_path.is_file():
            raise VerificationError(
                f"Missing golden result for track {tspec.track}: {result_path}"
            )
        bench_data = json.loads(result_path.read_text(encoding="utf-8"))
        resolved_tokens: list[int] = list(bench_data["output_token_ids"])
    else:
        resolved_tokens = list(output_tokens)

    resolved_run_id = validate_run_id(run_id or f"dry_{case_id}_{uuid.uuid4().hex[:8]}")
    iso_env = allocate_isolation_env(resolved_run_id, port_base=port_base, world_size=ws)
    verify_isolation_env(iso_env, world_size=ws)

    accepted_per_round = _accepted_schedule_for_case(case_id)
    round_traces, w_history = _build_round_traces(
        case_id, accepted_per_round, width=width, min_w_act=min_w_act
    )
    horizon_levels = [int(trace["horizon_level"]) for trace in round_traces]
    main_state = build_synthetic_cache_state(case_id, TARGET_LAYERS, "main")
    draft_state = build_synthetic_cache_state(case_id, 1, "draft")
    trials = _build_dry_run_trials(
        milestone,
        n_warm,
        n_form,
        ab_ba,
        spec,
        resolved_tokens,
        accepted_per_round,
        world_size=ws,
    )

    warm_ms = [
        float(t["max_tpot_ms"])
        for t in trials
        if t["is_warmup"] and t["arm"] == "candidate"
    ]
    formal_ms = [
        float(t["max_tpot_ms"])
        for t in trials
        if not t["is_warmup"] and t["arm"] == "candidate"
    ]
    base_formal_ms = [
        float(t["max_tpot_ms"])
        for t in trials
        if not t["is_warmup"] and t["arm"] == "baseline"
    ]
    ab_ba_delta = (
        float(statistics.median(formal_ms) - statistics.median(base_formal_ms))
        if base_formal_ms
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    last_cand = [t for t in trials if not t["is_warmup"] and t["arm"] == "candidate"][-1]
    runtime_invariants = {
        "w_act_min": min_w_act,
        "w_act_max": max(w_history),
        "w_act_history": w_history,
        "collective_split_max_rows": 16,
        "l78_gemm_chunk_max_rows": 8,
        "in_flight_depth": 2,
        "draft_sfa_topk": draft_sfa_topk,
        "target_sfa_topk": TARGET_SFA_TOPK,
    }
    ranks_manifest = _write_dry_run_ranks(
        output_dir,
        spec,
        resolved_tokens,
        accepted_per_round,
        horizon_levels,
        last_cand,
        main_state,
        draft_state,
        runtime_invariants,
        world_size=ws,
    )
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "milestone": milestone,
        "track": tspec.track,
        "run_id": resolved_run_id,
        "case_id": case_id,
        "width": width,
        "prompt_sha256": spec.prompt_sha256,
        "world_size": ws,
        "expert_parallel": True,
        "protocol": {
            "warmups": n_warm,
            "formals": n_form,
            "paired_order": "ab_ba" if ab_ba else "single",
            "profiler_enabled": False,
            "sync_debug_flags": [],
            "head_warmed": True,
            "head_execution_before_timing": {
                "wrapped": True,
                "active_graph_width": width,
                "graph_widths": [12, 16, 24],
            },
            "prefix_restored": True,
            "synchronized_start": True,
            "hccl_op_expansion_mode": "AIV",
            "hccl_deterministic": True,
            "tpot_formula": tpot_formula_for_world_size(ws),
        },
        "isolation": expected_isolation_metadata(iso_env),
        "runtime_hygiene": {
            "loaded_modules_has_vllm": milestone not in VLLM_FREE_MILESTONES,
            "compiled_so_dir": f"{iso_env['TILERT_RUN_DIR']}/builds",
            "draft_head_distinct_from_lm_head": True,
        },
        "round_traces": round_traces,
        "trials": trials,
        "summary": {
            "warmups_ms": warm_ms,
            "formal_ms": formal_ms,
            "median_tpot_ms": float(statistics.median(formal_ms)),
            "min_tpot_ms": float(min(formal_ms)),
            "p90_tpot_ms": compute_p90(formal_ms),
            "output_sha256": spec.output_sha256,
            "all_ranks_exact": True,
            "ekv_serial_diffs": {},
            "ab_ba_delta_ms": ab_ba_delta,
        },
        "ranks": ranks_manifest,
    }
    (output_dir / "e2e_result.json").write_text(
        json.dumps(result_payload, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def compute_live_cache_digest(cache: Any, position: int) -> dict[str, Any]:
    if isinstance(position, bool) or not isinstance(position, int) or position <= 0:
        raise VerificationError(f"position must be positive int, got {position!r}")
    if not isinstance(cache, Sequence) or isinstance(cache, (str, bytes)) or not cache:
        raise VerificationError("cache must be a non-empty sequence of layer mappings")
    for layer_idx, layer in enumerate(cache):
        if not isinstance(layer, Mapping) or not layer:
            raise VerificationError(f"cache[{layer_idx}] must be a non-empty mapping")
        for field_name, tensor in layer.items():
            shape = getattr(tensor, "shape", None)
            if (
                shape is None
                or len(shape) < 1
                or isinstance(shape[0], bool)
                or not isinstance(shape[0], int)
                or shape[0] < position
            ):
                raise VerificationError(
                    f"cache[{layer_idx}][{field_name!r}] shape {shape!r} < position {position}"
                )

    import importlib.util
    import torch

    try:
        from tests.glm52.cache_digest import cache_digest
    except ModuleNotFoundError:
        digest_path = Path(__file__).resolve().parents[1] / "cache_digest.py"
        spec = importlib.util.spec_from_file_location("glm52_cache_digest", digest_path)
        if spec is None or spec.loader is None:
            raise VerificationError(f"Unable to load cache_digest module at {digest_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cache_digest = mod.cache_digest

    if not hasattr(torch, "uint8"):
        raise VerificationError("torch runtime missing uint8 tensor support")
    digest = cache_digest(cache, position)
    if (
        not isinstance(digest, dict)
        or digest.get("position") != position
        or not isinstance(digest.get("layers"), list)
        or len(digest["layers"]) != len(cache)
    ):
        raise VerificationError(f"Invalid live cache digest returned: {digest!r}")
    return digest


def build_runner_command(
    runner_cmd: str,
    case_dir: Path,
    output_dir: Path,
    warmups: int,
    formals: int,
    ab_ba: bool,
    milestone: str,
    width: int,
    world_size: int = WORLD_SIZE,
    ep_size: int = EP_SIZE,
) -> list[str]:
    base = shlex.split(runner_cmd)
    if not base:
        raise VerificationError("Empty runner_cmd")
    return [
        *base,
        "--case-dir",
        str(case_dir),
        "--output-dir",
        str(output_dir),
        "--warmups",
        str(warmups),
        "--trials",
        str(formals),
        "--protocol",
        "ab_ba" if ab_ba else "single",
        "--max-tokens",
        "1024",
        "--isl",
        "8192",
        "--tp-size",
        str(world_size),
        "--ep-size",
        str(ep_size),
        "--width",
        str(width),
        "--check-digests",
        "1",
        "--milestone",
        milestone,
    ]


def _check_fresh_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise VerificationError(f"output_dir {output_dir} exists and is not a directory")
    stale = sorted(entry.name for entry in output_dir.iterdir())
    if stale:
        raise VerificationError(
            f"Refusing non-empty output_dir {output_dir} containing pre-existing {stale}"
        )


def run_a3_formal(
    case_id: str,
    output_dir: Path,
    case_root: Path = DEFAULT_CASE_ROOT,
    runner_cmd: str | None = None,
    kbenv_bin: str = DEFAULT_KBENV_BIN,
    kbenv_args: list[str] | None = None,
    warmups: int = MIN_WARMUPS,
    formals: int = MIN_FORMALS,
    ab_ba: bool = False,
    milestone: str = "M5",
    max_tpot_ms: float = TPOT_TARGET_MS,
    width: int = 16,
    target_host: str | None = None,
    port_base: int | None = None,
    run_id: str | None = None,
    timeout_s: float = DEFAULT_RUNNER_TIMEOUT_S,
    visible_devices: Sequence[int] | None = None,
    main_cache: Any = None,
    serial_draft_cache: Any = None,
    track: str = DEFAULT_TRACK,
) -> dict[str, Any]:
    tspec = get_track_spec(track)
    spec = get_case_spec(case_id, track=tspec.track)
    ws = tspec.world_size
    if width not in WIDTHS:
        raise VerificationError(f"Unsupported width: {width} (expected in {WIDTHS})")
    if (
        isinstance(max_tpot_ms, bool)
        or not isinstance(max_tpot_ms, (int, float))
        or not math.isfinite(float(max_tpot_ms))
        or float(max_tpot_ms) <= 0.0
    ):
        raise VerificationError(f"max_tpot_ms must be finite positive, got {max_tpot_ms!r}")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or float(timeout_s) <= 0.0
    ):
        raise VerificationError(f"timeout_s must be finite positive, got {timeout_s!r}")
    n_warm, n_form = validate_protocol_counts(warmups, formals, ab_ba)
    case_dir = case_root / case_id
    verify_prompt_case(case_dir, spec, track=tspec.track)
    milestone_tpot_limit(milestone)

    if target_host is not None:
        selected_host = assert_host_allowed(target_host)
        chosen_dies = visible_devices
    else:
        _, selected = run_kbenv_precheck(
            kbenv_bin=kbenv_bin, extra_args=kbenv_args, min_free_dies=ws
        )
        selected_host = assert_host_allowed(selected.ip)
        chosen_dies = (
            visible_devices
            if visible_devices is not None
            else select_free_die_slice(selected.free_die_indices, ws)
        )

    resolved_run_id = validate_run_id(run_id or f"a3_{case_id}_{uuid.uuid4().hex[:8]}")
    resolved_port = allocate_free_port_base() if port_base is None else port_base
    iso_env = allocate_isolation_env(
        resolved_run_id,
        port_base=resolved_port,
        target_host=selected_host,
        visible_devices=chosen_dies,
        world_size=ws,
    )
    merged_env = {**os.environ, **iso_env}
    verify_isolation_env(merged_env, target_host=selected_host, world_size=ws)

    resolved_cmd = runner_cmd or os.environ.get("TILERT_E2E_RUNNER_CMD", "")
    if not resolved_cmd.strip():
        raise VerificationError(
            "Live a3-formal execution requires --runner-cmd or TILERT_E2E_RUNNER_CMD"
        )
    _check_fresh_output_dir(output_dir)
    cmd = build_runner_command(
        resolved_cmd,
        case_dir=case_dir,
        output_dir=output_dir,
        warmups=n_warm,
        formals=n_form,
        ab_ba=ab_ba,
        milestone=milestone,
        width=width,
        world_size=ws,
        ep_size=tspec.ep_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        cmd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=float(timeout_s))
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise VerificationError(f"Runner timed out after {float(timeout_s):.1f}s") from exc
    if proc.returncode != 0:
        raise VerificationError(
            f"Runner exited {proc.returncode}: {stderr.strip() or stdout.strip()}"
        )

    main_ref = compute_live_cache_digest(main_cache, FINAL_POS) if main_cache is not None else None
    serial_draft_ref = (
        compute_live_cache_digest(serial_draft_cache, FINAL_POS)
        if serial_draft_cache is not None
        else None
    )
    result = verify_e2e_run(
        output_dir,
        case_dir,
        max_tpot_ms=max_tpot_ms,
        require_warm_head=True,
        require_ab_ba=ab_ba,
        milestone=milestone,
        serial_draft_ref=serial_draft_ref,
        main_ref=main_ref,
        track=tspec.track,
    )
    report = json.loads((output_dir / "e2e_result.json").read_text(encoding="utf-8"))
    if report.get("run_id") != resolved_run_id:
        raise VerificationError(
            f"Report run_id {report.get('run_id')!r} != expected {resolved_run_id!r}"
        )
    expected_iso = expected_isolation_metadata(iso_env)
    if report.get("isolation") != expected_iso:
        raise VerificationError(
            f"Report isolation {report.get('isolation')!r} != expected {expected_iso!r}"
        )
    protocol = report.get("protocol", {})
    if protocol.get("warmups") != n_warm:
        raise VerificationError(
            f"Report protocol.warmups {protocol.get('warmups')!r} != expected {n_warm}"
        )
    if protocol.get("formals") != n_form:
        raise VerificationError(
            f"Report protocol.formals {protocol.get('formals')!r} != expected {n_form}"
        )
    if report.get("width") != width:
        raise VerificationError(
            f"Report width {report.get('width')!r} != expected {width}"
        )
    if report.get("milestone") != milestone:
        raise VerificationError(
            f"Report milestone {report.get('milestone')!r} != expected {milestone!r}"
        )
    result["host"] = selected_host
    result["run_id"] = resolved_run_id
    result["visible_devices"] = iso_env["ASCEND_RT_VISIBLE_DEVICES"]
    return result
