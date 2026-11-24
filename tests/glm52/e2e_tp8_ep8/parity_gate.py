from collections.abc import Sequence
import json
import math
from pathlib import Path
import posixpath
import statistics
from typing import Any, TypeGuard

from .contracts import (
    AB_BA_FORMAL,
    AB_BA_WARM,
    ALLOW_RELAXED_PARITY,
    CaseSpec,
    DECODE_TOKENS,
    DEFAULT_TRACK,
    DRAFT_LAYERS,
    DRAFT_SFA_TOPK_OPTIONS,
    FINAL_POS,
    ISL,
    IsolationError,
    MAX_COLLECTIVE_ROWS,
    MAX_L78_CHUNK_ROWS,
    MIN_FORMALS,
    MIN_WARMUPS,
    OSL,
    PROMPT_HEAD_TOKENS,
    PROMPT_TAIL_TOKENS,
    SCHEMA_VERSION,
    STRICT_LT_MILESTONES,
    TARGET_LAYERS,
    TARGET_SFA_TOPK,
    TPOT_TARGET_MS,
    TrackSpec,
    VLLM_FREE_MILESTONES,
    VOCAB_SIZE,
    WIDTHS,
    WORLD_SIZE,
    VerificationError,
    compact_tokens_sha256,
    default_tokens_sha256,
    file_sha256,
    get_case_spec,
    get_track_spec,
    golden_result_path,
    is_valid_sha256_hex,
    milestone_tpot_limit,
    tpot_formula_for_world_size,
)
from .hygiene_gate import (
    _parse_endpoint_port,
    _parse_port_range,
    validate_run_id,
    verify_shm_isolation,
)


def _reject_relaxed_parity(allow_relaxed_parity: bool) -> None:
    if allow_relaxed_parity is not ALLOW_RELAXED_PARITY:
        raise VerificationError("allow_relaxed_parity cannot be enabled")


def _is_int_token(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_positive_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} must be numeric, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval) or fval <= 0.0:
        raise VerificationError(f"{label} must be finite positive, got {value!r}")
    return fval


def _validate_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} must be numeric, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval):
        raise VerificationError(f"{label} must be finite, got {value!r}")
    return fval


def _validate_width(value: object, label: str = "width") -> int:
    if not _is_int_token(value) or value not in WIDTHS:
        raise VerificationError(f"{label} must be in {WIDTHS}, got {value!r}")
    return value


def _validate_vocab_tokens(tokens: object, expected_len: int, label: str) -> list[int]:
    if not isinstance(tokens, list):
        raise VerificationError(f"{label} must be a list of token IDs")
    if len(tokens) != expected_len:
        raise VerificationError(
            f"{label} length mismatch: expected {expected_len}, got {len(tokens)}"
        )
    validated: list[int] = []
    for idx, tok in enumerate(tokens):
        if not _is_int_token(tok) or not (0 <= tok < VOCAB_SIZE):
            raise VerificationError(
                f"Out-of-vocab or non-int token at {label}[{idx}]: {tok!r}"
            )
        validated.append(tok)
    return validated


def compute_p90(values: Sequence[float]) -> float:
    if not values:
        raise VerificationError("Cannot compute p90 of empty sequence")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = 0.9 * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _resolve_report_track(report: dict[str, Any], track: str | None = None) -> TrackSpec:
    if track is not None:
        tspec = get_track_spec(track)
        if "track" in report and report["track"] != tspec.track:
            raise VerificationError(
                f"Report track {report['track']!r} != expected {tspec.track!r}"
            )
        return tspec
    raw_track = report.get("track", DEFAULT_TRACK)
    if not isinstance(raw_track, str):
        raise VerificationError(f"Invalid report track: {raw_track!r}")
    return get_track_spec(raw_track)


def verify_prompt_case(
    case_dir: Path,
    expected_spec: CaseSpec,
    track: str = DEFAULT_TRACK,
) -> dict[str, Any]:
    prompt_path = case_dir / "prompt.json"
    if not prompt_path.is_file():
        raise VerificationError(f"Missing prompt.json in {case_dir}")
    prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(prompt_data, dict):
        raise VerificationError(f"Expected JSON object in {prompt_path}")
    if prompt_data.get("case_id") != expected_spec.case_id:
        raise VerificationError(
            f"case_id mismatch in {prompt_path}: {prompt_data.get('case_id')!r}"
        )
    if not _is_int_token(prompt_data.get("seq_len")) or prompt_data.get("seq_len") != ISL:
        raise VerificationError(
            f"seq_len mismatch in {prompt_path}: {prompt_data.get('seq_len')!r}"
        )
    tokens = _validate_vocab_tokens(
        prompt_data.get("prompt_token_ids"),
        ISL,
        f"{expected_spec.case_id}.prompt_token_ids",
    )
    if tuple(tokens[:3]) != PROMPT_HEAD_TOKENS:
        raise VerificationError(
            f"Unexpected chat prefix tokens in {prompt_path}: {tokens[:3]}"
        )
    if tuple(tokens[-3:]) != PROMPT_TAIL_TOKENS:
        raise VerificationError(
            f"Unexpected chat suffix tokens in {prompt_path}: {tokens[-3:]}"
        )
    actual_sha = compact_tokens_sha256(tokens)
    if prompt_data.get("prompt_sha256") != expected_spec.prompt_sha256:
        raise VerificationError(
            f"prompt_sha256 field mismatch in {prompt_path}: {prompt_data.get('prompt_sha256')}"
        )
    if actual_sha != expected_spec.prompt_sha256:
        raise VerificationError(
            f"Computed compact prompt SHA256 mismatch for {expected_spec.case_id}: {actual_sha}"
        )
    bench_path = golden_result_path(case_dir, track)
    if bench_path.is_file():
        bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
        if not isinstance(bench_data, dict):
            raise VerificationError(f"Expected JSON object in {bench_path}")
        if bench_data.get("prompt_sha256") != expected_spec.prompt_sha256:
            raise VerificationError(f"Golden result prompt_sha256 tampered in {bench_path}")
        if (
            not _is_int_token(bench_data.get("first_token_id"))
            or bench_data.get("first_token_id") != expected_spec.first_token_id
        ):
            raise VerificationError(f"Golden result first_token_id tampered in {bench_path}")
        if bench_data.get("output_sha256") != expected_spec.output_sha256:
            raise VerificationError(f"Golden result output_sha256 tampered in {bench_path}")
        bench_tokens = _validate_vocab_tokens(
            bench_data.get("output_token_ids"),
            OSL,
            f"{expected_spec.case_id}.golden.{track}.output_token_ids",
        )
        if default_tokens_sha256(bench_tokens) != expected_spec.output_sha256:
            raise VerificationError(f"Golden result tokens tampered in {bench_path}")
    return prompt_data


def verify_output_tokens(
    tokens: object,
    expected_sha256: str,
    first_token_id: int,
    accepted_per_round: object = None,
    allow_relaxed_parity: bool = False,
) -> str:
    _reject_relaxed_parity(allow_relaxed_parity)
    if not is_valid_sha256_hex(expected_sha256):
        raise VerificationError(f"Invalid expected_sha256: {expected_sha256!r}")
    validated = _validate_vocab_tokens(tokens, OSL, "output_token_ids")
    if validated[0] != first_token_id:
        raise VerificationError(
            f"first_token_id mismatch: expected {first_token_id}, got {validated[0]}"
        )
    if accepted_per_round is not None:
        if not isinstance(accepted_per_round, list) or not accepted_per_round:
            raise VerificationError("accepted_per_round must be a non-empty list")
        acc_ints: list[int] = []
        for idx, count in enumerate(accepted_per_round):
            if not _is_int_token(count) or count <= 0:
                raise VerificationError(
                    f"accepted_per_round[{idx}] must be positive int, got {count!r}"
                )
            acc_ints.append(count)
        if sum(acc_ints) != DECODE_TOKENS:
            raise VerificationError(
                f"sum(accepted_per_round) must equal {DECODE_TOKENS}, got {sum(acc_ints)}"
            )
    actual_sha = default_tokens_sha256(validated)
    if actual_sha != expected_sha256:
        raise VerificationError(
            f"output_sha256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    return actual_sha


def _validate_cache_state_dict(
    state: object,
    expected_layers: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise VerificationError(f"{label} must be a dict")
    pos = state.get("position")
    if not _is_int_token(pos) or pos != FINAL_POS:
        raise VerificationError(
            f"{label}.position mismatch: expected {FINAL_POS}, got {pos!r}"
        )
    layers = state.get("layers")
    if not isinstance(layers, list) or len(layers) != expected_layers:
        raise VerificationError(
            f"{label}.layers must have {expected_layers} entries, "
            f"got {len(layers) if isinstance(layers, list) else type(layers).__name__}"
        )
    for layer_idx, layer in enumerate(layers):
        if not isinstance(layer, dict) or not layer:
            raise VerificationError(f"{label}.layers[{layer_idx}] must be a non-empty dict")
        for field_name, meta in layer.items():
            if not isinstance(meta, dict):
                raise VerificationError(
                    f"{label}.layers[{layer_idx}][{field_name!r}] must be a dict"
                )
            dtype = meta.get("dtype")
            if not isinstance(dtype, str) or not dtype.strip():
                raise VerificationError(
                    f"{label}.layers[{layer_idx}][{field_name!r}].dtype invalid"
                )
            shape = meta.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or not all(_is_int_token(dim) and dim > 0 for dim in shape)
                or shape[0] < FINAL_POS
            ):
                raise VerificationError(
                    f"{label}.layers[{layer_idx}][{field_name!r}].shape invalid: {shape!r}"
                )
            if not is_valid_sha256_hex(meta.get("sha256")):
                raise VerificationError(
                    f"{label}.layers[{layer_idx}][{field_name!r}].sha256 invalid"
                )
    return state


def verify_8rank_states(
    rank_reports: list[dict[str, Any]],
    serial_draft_ref: dict[str, Any] | None = None,
    main_ref: dict[str, Any] | None = None,
    expected_spec: CaseSpec | None = None,
    allow_relaxed_parity: bool = False,
    world_size: int = WORLD_SIZE,
) -> None:
    _reject_relaxed_parity(allow_relaxed_parity)
    if not _is_int_token(world_size) or not (1 <= world_size <= 16):
        raise VerificationError(f"Unsupported world_size: {world_size!r}")
    if not isinstance(rank_reports, list) or len(rank_reports) != world_size:
        actual = (
            len(rank_reports)
            if isinstance(rank_reports, list)
            else type(rank_reports).__name__
        )
        raise VerificationError(f"Expected {world_size} rank reports, got {actual}")
    seen_ranks: dict[int, dict[str, Any]] = {}
    for idx, report in enumerate(rank_reports):
        if not isinstance(report, dict):
            raise VerificationError(f"rank_reports[{idx}] must be a dict")
        rank = report.get("rank")
        if not _is_int_token(rank) or not (0 <= rank < world_size):
            raise VerificationError(f"Invalid rank ID: {rank!r}")
        if rank in seen_ranks:
            raise VerificationError(f"Duplicate rank ID detected: {rank}")
        row_ws = report.get("world_size")
        if not _is_int_token(row_ws) or row_ws != world_size:
            raise VerificationError(
                f"Rank {rank} world_size must be {world_size}, got {row_ws!r}"
            )
        seen_ranks[rank] = report
    if set(seen_ranks) != set(range(world_size)):
        missing = sorted(set(range(world_size)) - set(seen_ranks))
        raise VerificationError(f"Missing rank reports: {missing}")

    ref_rank0 = seen_ranks[0]
    ref_tokens = ref_rank0.get("output_token_ids")
    ref_first = ref_rank0.get("first_token_id")
    ref_sha = ref_rank0.get("output_sha256")
    if not _is_int_token(ref_first):
        raise VerificationError(f"Rank 0 first_token_id must be int, got {ref_first!r}")
    if expected_spec is not None:
        if ref_first != expected_spec.first_token_id:
            raise VerificationError(
                f"Rank 0 first_token_id {ref_first!r} != expected {expected_spec.first_token_id}"
            )
        if ref_sha != expected_spec.output_sha256:
            raise VerificationError(
                f"Rank 0 output_sha256 {ref_sha!r} != expected {expected_spec.output_sha256}"
            )
    ref_accepted = ref_rank0.get("accepted_per_round")
    if ref_accepted is None:
        raise VerificationError("Rank 0 missing accepted_per_round")
    verify_output_tokens(
        ref_tokens,
        expected_sha256=str(ref_sha),
        first_token_id=ref_first,
        accepted_per_round=ref_accepted,
        allow_relaxed_parity=allow_relaxed_parity,
    )
    ref_main = _validate_cache_state_dict(
        ref_rank0.get("main_state"), TARGET_LAYERS, "rank-0.main_state"
    )
    ref_draft = _validate_cache_state_dict(
        ref_rank0.get("draft_state"), DRAFT_LAYERS, "rank-0.draft_state"
    )
    if main_ref is not None:
        validated_main_ref = _validate_cache_state_dict(main_ref, TARGET_LAYERS, "main_ref")
        if ref_main != validated_main_ref:
            raise VerificationError("Rank 0 main_state diverges from main_ref")
    if serial_draft_ref is not None:
        validated_draft_ref = _validate_cache_state_dict(
            serial_draft_ref, DRAFT_LAYERS, "serial_draft_ref"
        )
        if ref_draft != validated_draft_ref:
            raise VerificationError("Rank 0 draft_state diverges from serial_draft_ref")

    for rank in range(world_size):
        row = seen_ranks[rank]
        if expected_spec is not None:
            if "case_id" in row and row["case_id"] != expected_spec.case_id:
                raise VerificationError(
                    f"Rank {rank} case_id {row['case_id']!r} != {expected_spec.case_id!r}"
                )
            if "prompt_sha256" in row and row["prompt_sha256"] != expected_spec.prompt_sha256:
                raise VerificationError(
                    f"Rank {rank} prompt_sha256 diverges from {expected_spec.case_id}"
                )
        if row.get("serial_draft_match") is not True:
            raise VerificationError(f"Rank {rank} serial_draft_match is not True")
        if row.get("ekv_serial_diffs") != {}:
            raise VerificationError(
                f"Rank {rank} ekv_serial_diffs must be empty, got {row.get('ekv_serial_diffs')!r}"
            )
        row_tokens = _validate_vocab_tokens(
            row.get("output_token_ids"), OSL, f"rank-{rank}.output_token_ids"
        )
        if row_tokens != ref_tokens:
            raise VerificationError(f"Rank {rank} output_token_ids diverges from rank 0")
        row_first = row.get("first_token_id")
        if not _is_int_token(row_first) or row_first != ref_first:
            raise VerificationError(f"Rank {rank} first_token_id diverges from rank 0")
        if row.get("output_sha256") != ref_sha:
            raise VerificationError(f"Rank {rank} output_sha256 diverges from rank 0")
        row_acc = row.get("accepted_per_round")
        if (
            not isinstance(row_acc, list)
            or not row_acc
            or any(not _is_int_token(v) or v <= 0 for v in row_acc)
            or row_acc != ref_accepted
        ):
            raise VerificationError(f"Rank {rank} accepted_per_round diverges from rank 0")
        row_rounds = row.get("rounds")
        if not _is_int_token(row_rounds) or row_rounds != len(row_acc):
            raise VerificationError(
                f"Rank {rank} rounds ({row_rounds!r}) != len(accepted_per_round)"
            )
        row_main = _validate_cache_state_dict(
            row.get("main_state"), TARGET_LAYERS, f"rank-{rank}.main_state"
        )
        row_draft = _validate_cache_state_dict(
            row.get("draft_state"), DRAFT_LAYERS, f"rank-{rank}.draft_state"
        )
        if row_main != ref_main:
            raise VerificationError(f"Rank {rank} main_state diverges from rank 0")
        if row_draft != ref_draft:
            raise VerificationError(f"Rank {rank} draft_state diverges from rank 0")
        dt_ms = _validate_positive_finite(row.get("dt_ms"), f"rank-{rank}.dt_ms")
        tpot_ms = _validate_finite_number(row.get("tpot_ms"), f"rank-{rank}.tpot_ms")
        expected_tpot = dt_ms / float(DECODE_TOKENS)
        if abs(tpot_ms - expected_tpot) > 1e-4:
            raise VerificationError(
                f"Rank {rank} tpot_ms {tpot_ms} != dt_ms / {DECODE_TOKENS} ({expected_tpot:.6f})"
            )


def _check_finite_positive_list(values: object, expected_len: int, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != expected_len:
        raise VerificationError(f"{label} must have length {expected_len}")
    return [_validate_positive_finite(val, f"{label}[{idx}]") for idx, val in enumerate(values)]


def _validate_protocol_metadata(
    report: dict[str, Any],
    require_warm_head: bool,
    milestone: str,
    expected_world_size: int = WORLD_SIZE,
) -> tuple[dict[str, Any], int, int, str]:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise VerificationError(
            f"Unsupported schema_version: {report.get('schema_version')!r}"
        )
    world_size = report.get("world_size")
    if not _is_int_token(world_size) or world_size != expected_world_size:
        raise VerificationError(
            f"Unsupported world_size: {world_size!r} (expected {expected_world_size})"
        )
    if report.get("expert_parallel") is not True:
        raise VerificationError("expert_parallel must be True")
    hygiene = report.get("runtime_hygiene")
    if not isinstance(hygiene, dict):
        raise VerificationError("Missing runtime_hygiene object in report")
    if milestone in VLLM_FREE_MILESTONES and hygiene.get("loaded_modules_has_vllm") is not False:
        raise VerificationError(
            f"Milestone {milestone} forbids vllm/vllm_ascend in loaded modules"
        )
    if hygiene.get("draft_head_distinct_from_lm_head") is not True:
        raise VerificationError("draft_head_distinct_from_lm_head must be True")

    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise VerificationError("Missing protocol object in report")
    warmups = protocol.get("warmups")
    formals = protocol.get("formals")
    if not _is_int_token(warmups) or warmups < MIN_WARMUPS:
        raise VerificationError(f"warmups must be >= {MIN_WARMUPS}, got {warmups!r}")
    if not _is_int_token(formals) or formals < MIN_FORMALS:
        raise VerificationError(f"formals must be >= {MIN_FORMALS}, got {formals!r}")
    paired_order = protocol.get("paired_order")
    if paired_order not in ("single", "ab_ba"):
        raise VerificationError(
            f"paired_order must be 'single' or 'ab_ba', got {paired_order!r}"
        )
    if protocol.get("prefix_restored") is not True:
        raise VerificationError("prefix_restored must be True before formal timing")
    if protocol.get("synchronized_start") is not True:
        raise VerificationError(
            f"synchronized_start must be True across all {expected_world_size} ranks"
        )
    if protocol.get("profiler_enabled") is not False:
        raise VerificationError("profiler_enabled must be False during TPOT timing")
    if protocol.get("sync_debug_flags") != []:
        raise VerificationError(
            f"sync_debug_flags must be empty, got {protocol.get('sync_debug_flags')!r}"
        )
    expected_formula = tpot_formula_for_world_size(expected_world_size)
    if protocol.get("tpot_formula") != expected_formula:
        raise VerificationError(f"tpot_formula must be {expected_formula!r}")
    if require_warm_head is not True:
        raise VerificationError("require_warm_head cannot be disabled")
    if protocol.get("head_warmed") is not True:
        raise VerificationError("head_warmed must be True before formal timing")
    head_meta = protocol.get("head_execution_before_timing")
    if head_meta is not None:
        if not isinstance(head_meta, dict):
            raise VerificationError("head_execution_before_timing must be a dict")
        wrapped = head_meta.get("wrapped")
        if not isinstance(wrapped, bool):
            raise VerificationError("head_execution_before_timing.wrapped must be bool")
        active_w = head_meta.get("active_graph_width")
        widths = head_meta.get("graph_widths", [])
        if not wrapped:
            report_width = _validate_width(report.get("width", 16), "report.width")
            if (
                not _is_int_token(active_w)
                or active_w != report_width
                or not isinstance(widths, list)
                or not all(_is_int_token(w) for w in widths)
                or active_w not in widths
            ):
                raise VerificationError(
                    "Cold head detected: neither wrapped nor matching active_graph_width"
                )
    return protocol, warmups, formals, str(paired_order)


def _collect_trial_metrics(
    trials: object,
    paired_order: str,
    expected_world_size: int = WORLD_SIZE,
) -> tuple[list[float], list[float], list[float], dict[int, list[str]], dict[int, list[bool]]]:
    if not isinstance(trials, list) or not trials:
        raise VerificationError("Missing or empty trials list in report")
    warmups_ms: list[float] = []
    formal_ms: list[float] = []
    baseline_formal_ms: list[float] = []
    ab_ba_pairs: dict[int, list[str]] = {}
    pair_warm_flags: dict[int, list[bool]] = {}
    seen_formal = False
    for idx, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise VerificationError(f"trials[{idx}] must be a dict")
        if "trial_idx" in trial:
            trial_idx = trial["trial_idx"]
            if not _is_int_token(trial_idx) or trial_idx != idx:
                raise VerificationError(
                    f"trials[{idx}].trial_idx {trial_idx!r} != sequential index {idx}"
                )
        is_warmup = trial.get("is_warmup")
        if not isinstance(is_warmup, bool):
            raise VerificationError(
                f"trials[{idx}].is_warmup must be bool, got {is_warmup!r}"
            )
        if is_warmup:
            if seen_formal:
                raise VerificationError(f"trials[{idx}] warmup appears after formal trial")
        else:
            seen_formal = True
        if trial.get("eos_drained_cleanly") is not True:
            raise VerificationError(f"trials[{idx}] eos_drained_cleanly is not True")
        rank_dts = _check_finite_positive_list(
            trial.get("rank_dt_ms"), expected_world_size, f"trials[{idx}].rank_dt_ms"
        )
        max_tpot = _validate_finite_number(
            trial.get("max_tpot_ms"), f"trials[{idx}].max_tpot_ms"
        )
        expected_max_tpot = max(rank_dts) / float(DECODE_TOKENS)
        if abs(max_tpot - expected_max_tpot) > 1e-4:
            raise VerificationError(
                f"trials[{idx}].max_tpot_ms {max_tpot} != max(rank_dt_ms)/{DECODE_TOKENS} "
                f"({expected_max_tpot:.6f})"
            )
        if paired_order == "ab_ba":
            if "arm" not in trial:
                raise VerificationError(f"trials[{idx}] missing mandatory arm in ab_ba mode")
            if "pair_idx" not in trial:
                raise VerificationError(f"trials[{idx}] missing mandatory pair_idx in ab_ba mode")
        arm = trial.get("arm", "candidate")
        if arm not in ("candidate", "baseline"):
            raise VerificationError(f"trials[{idx}] invalid arm: {arm!r}")
        if paired_order == "single" and arm != "candidate":
            raise VerificationError(f"trials[{idx}] arm {arm!r} invalid in single mode")
        if arm == "candidate":
            if is_warmup:
                warmups_ms.append(expected_max_tpot)
            else:
                formal_ms.append(expected_max_tpot)
        elif not is_warmup:
            baseline_formal_ms.append(expected_max_tpot)
        if paired_order == "ab_ba" or "pair_idx" in trial:
            pidx = trial["pair_idx"]
            if not _is_int_token(pidx) or pidx < 0:
                raise VerificationError(f"Invalid pair_idx in trials[{idx}]: {pidx!r}")
            if paired_order == "ab_ba" and pidx != idx // 2:
                raise VerificationError(
                    f"Non-contiguous AB_BA pair_idx in trials[{idx}]: {pidx} != {idx // 2}"
                )
            ab_ba_pairs.setdefault(pidx, []).append(str(arm))
            pair_warm_flags.setdefault(pidx, []).append(is_warmup)
    return warmups_ms, formal_ms, baseline_formal_ms, ab_ba_pairs, pair_warm_flags


def _validate_ab_ba_pairs(
    protocol: dict[str, Any],
    warmups: int,
    formals: int,
    ab_ba_pairs: dict[int, list[str]],
    pair_warm_flags: dict[int, list[bool]],
) -> None:
    if protocol.get("hccl_op_expansion_mode") != "AIV":
        raise VerificationError("AB_BA requires hccl_op_expansion_mode == 'AIV'")
    if protocol.get("hccl_deterministic") is not True:
        raise VerificationError("AB_BA requires hccl_deterministic == True")
    if warmups < AB_BA_WARM or formals < AB_BA_FORMAL:
        raise VerificationError(
            f"AB_BA requires >= {AB_BA_WARM} warmups and >= {AB_BA_FORMAL} formals"
        )
    if len(ab_ba_pairs) != warmups + formals:
        raise VerificationError(
            f"AB_BA pair count ({len(ab_ba_pairs)}) != warmups + formals ({warmups + formals})"
        )
    warm_pair_count = 0
    formal_pair_count = 0
    for pidx in range(len(ab_ba_pairs)):
        if pidx not in ab_ba_pairs:
            raise VerificationError(f"Non-contiguous AB_BA pair_idx missing {pidx}")
        expected_seq = ["baseline", "candidate"] if pidx % 2 == 0 else ["candidate", "baseline"]
        if ab_ba_pairs[pidx] != expected_seq:
            raise VerificationError(
                f"AB_BA pair {pidx} order {ab_ba_pairs[pidx]} != expected {expected_seq}"
            )
        flags = pair_warm_flags[pidx]
        if pidx < warmups:
            if not all(flags):
                raise VerificationError(f"AB_BA warmup pair {pidx} contains formal trial")
            warm_pair_count += 1
        else:
            if any(flags):
                raise VerificationError(f"AB_BA formal pair {pidx} contains warmup trial")
            formal_pair_count += 1
    if warm_pair_count != warmups or formal_pair_count != formals:
        raise VerificationError(
            f"AB_BA pair counts ({warm_pair_count} warm, {formal_pair_count} formal) "
            f"!= protocol ({warmups} warm, {formals} formal)"
        )


def _validate_summary_metrics(
    report: dict[str, Any],
    warmups_ms: list[float],
    formal_ms: list[float],
    baseline_formal_ms: list[float],
    paired_order: str,
    effective_limit: float,
    milestone: str,
) -> dict[str, float]:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise VerificationError("Missing summary object in report")
    if summary.get("all_ranks_exact") is not True:
        raise VerificationError("summary.all_ranks_exact must be True")
    summary_warmups = summary.get("warmups_ms")
    if not isinstance(summary_warmups, list) or len(summary_warmups) != len(warmups_ms):
        raise VerificationError(
            "summary.warmups_ms count does not match candidate warmup trials"
        )
    for idx, (sval, tval) in enumerate(zip(summary_warmups, warmups_ms)):
        f_sval = _validate_positive_finite(sval, f"summary.warmups_ms[{idx}]")
        if abs(f_sval - tval) > 1e-4:
            raise VerificationError(f"summary.warmups_ms[{idx}] {sval} != trial {tval}")
    summary_formals = summary.get("formal_ms")
    if not isinstance(summary_formals, list) or len(summary_formals) != len(formal_ms):
        raise VerificationError("summary.formal_ms count does not match candidate formal trials")
    for idx, (sval, tval) in enumerate(zip(summary_formals, formal_ms)):
        f_sval = _validate_positive_finite(sval, f"summary.formal_ms[{idx}]")
        if abs(f_sval - tval) > 1e-4:
            raise VerificationError(f"summary.formal_ms[{idx}] {sval} != trial {tval}")
    expected_median = float(statistics.median(formal_ms))
    expected_min = float(min(formal_ms))
    expected_p90 = compute_p90(formal_ms)
    for field_name, exp_val in (
        ("median_tpot_ms", expected_median),
        ("min_tpot_ms", expected_min),
        ("p90_tpot_ms", expected_p90),
    ):
        actual_val = _validate_finite_number(summary.get(field_name), f"summary.{field_name}")
        if abs(actual_val - exp_val) > 1e-4:
            raise VerificationError(
                f"summary.{field_name} {actual_val} != recomputed {exp_val:.6f}"
            )
    if paired_order == "ab_ba":
        if len(baseline_formal_ms) != len(formal_ms):
            raise VerificationError("AB_BA baseline formal count != candidate formal count")
        expected_delta = expected_median - float(statistics.median(baseline_formal_ms))
        actual_delta = _validate_finite_number(
            summary.get("ab_ba_delta_ms"), "summary.ab_ba_delta_ms"
        )
        if abs(actual_delta - expected_delta) > 1e-4:
            raise VerificationError(
                f"summary.ab_ba_delta_ms {actual_delta} != recomputed {expected_delta:.6f}"
            )
    if milestone in STRICT_LT_MILESTONES:
        if expected_median >= effective_limit:
            raise VerificationError(
                f"Median TPOT {expected_median:.4f} ms >= target {effective_limit:.4f} ms"
            )
    elif expected_median > effective_limit:
        raise VerificationError(
            f"Median TPOT {expected_median:.4f} ms > milestone limit {effective_limit:.4f} ms"
        )
    return {
        "median_tpot_ms": expected_median,
        "min_tpot_ms": expected_min,
        "p90_tpot_ms": expected_p90,
    }


def verify_timing_and_protocol(
    report: dict[str, Any],
    max_tpot_ms: float = TPOT_TARGET_MS,
    require_warm_head: bool = True,
    require_ab_ba: bool = False,
    milestone: str = "M5",
    track: str | None = None,
) -> dict[str, float]:
    if not isinstance(report, dict):
        raise VerificationError("report must be a dict")
    max_tpot = _validate_positive_finite(max_tpot_ms, "max_tpot_ms")
    effective_limit = min(max_tpot, milestone_tpot_limit(milestone))
    tspec = _resolve_report_track(report, track)
    protocol, warmups, formals, paired_order = _validate_protocol_metadata(
        report,
        require_warm_head=require_warm_head,
        milestone=milestone,
        expected_world_size=tspec.world_size,
    )
    warmups_ms, formal_ms, baseline_formal_ms, ab_ba_pairs, pair_warm_flags = (
        _collect_trial_metrics(
            report.get("trials"), paired_order, expected_world_size=tspec.world_size
        )
    )
    if len(warmups_ms) < MIN_WARMUPS or len(warmups_ms) != warmups:
        raise VerificationError(
            f"Candidate warmup trials ({len(warmups_ms)}) != protocol.warmups ({warmups})"
        )
    if len(formal_ms) < MIN_FORMALS or len(formal_ms) != formals:
        raise VerificationError(
            f"Candidate formal trials ({len(formal_ms)}) != protocol.formals ({formals})"
        )
    if require_ab_ba and paired_order != "ab_ba":
        raise VerificationError(f"paired_order must be 'ab_ba', got {paired_order!r}")
    if paired_order == "ab_ba":
        _validate_ab_ba_pairs(protocol, warmups, formals, ab_ba_pairs, pair_warm_flags)
    return _validate_summary_metrics(
        report,
        warmups_ms,
        formal_ms,
        baseline_formal_ms,
        paired_order,
        effective_limit,
        milestone,
    )


def select_optimal_w_act(
    raw_lps: Sequence[float],
    width: int,
    min_w_act: int = 8,
    t_base_ms: float = 47.0,
    c_w_ms: float = 0.72,
) -> int:
    resolved_width = _validate_width(width, "width")
    if isinstance(min_w_act, bool) or not isinstance(min_w_act, int):
        raise VerificationError(f"min_w_act must be int, got {min_w_act!r}")
    base_ms = _validate_positive_finite(t_base_ms, "t_base_ms")
    step_ms = _validate_positive_finite(c_w_ms, "c_w_ms")
    if not isinstance(raw_lps, Sequence) or isinstance(raw_lps, (str, bytes)):
        raise VerificationError("raw_lps must be a sequence of finite numbers")
    if len(raw_lps) < resolved_width:
        raise VerificationError(f"raw_lps length {len(raw_lps)} < width {resolved_width}")
    validated_lps = [
        _validate_finite_number(val, f"raw_lps[{idx}]") for idx, val in enumerate(raw_lps)
    ]
    if min_w_act <= 0 or min_w_act >= resolved_width:
        return resolved_width
    e_tau = 0.0
    best_tpot = 1.0e30
    best_w = resolved_width
    for r in range(resolved_width):
        if r == 0:
            p_r = 1.0
        else:
            rlp = min(max(validated_lps[r], -20.0), 0.0)
            p_r = math.exp(rlp)
        e_tau += p_r
        cand_w = r + 1
        if cand_w >= min_w_act and ((cand_w & 1) == 0 or cand_w == resolved_width):
            est_tpot = (base_ms + step_ms * float(cand_w)) / e_tau
            if est_tpot <= best_tpot:
                best_tpot = est_tpot
                best_w = cand_w
    return best_w


class ZeroLagHorizonRef:
    def __init__(
        self,
        num_levels: int = 3,
        init_level: int = 1,
        init_ema: float = 6.2,
        alpha: float = 0.28,
        mid_thresh: float = 5.40,
        up_thresh: float = 7.15,
        down_thresh: float = 4.90,
        drop_acc: int = 3,
        jump_acc: int = 9,
        streak_acc: int = 7,
    ) -> None:
        self.num_levels = max(2, int(num_levels))
        self.init_level = min(max(int(init_level), 0), self.num_levels - 1)
        self.init_ema = float(init_ema)
        self.alpha = float(alpha)
        self.mid_thresh = float(mid_thresh)
        self.up_thresh = float(up_thresh)
        self.down_thresh = float(down_thresh)
        self.drop_acc = int(drop_acc)
        self.jump_acc = int(jump_acc)
        self.streak_acc = int(streak_acc)
        self.ema = self.init_ema
        self.level = self.init_level
        self.hi_streak = 0
        self.lo_streak = 0

    def reset(self) -> None:
        self.ema = self.init_ema
        self.level = self.init_level
        self.hi_streak = 0
        self.lo_streak = 0

    def update(self, accepted: int) -> int:
        acc_f = float(accepted)
        self.ema = (1.0 - self.alpha) * self.ema + self.alpha * acc_f
        if accepted >= self.streak_acc:
            self.hi_streak += 1
            self.lo_streak = 0
        elif accepted <= self.drop_acc:
            self.lo_streak += 1
            self.hi_streak = 0
        else:
            self.hi_streak = 0
            self.lo_streak = 0

        top = self.num_levels - 1
        if accepted <= 2 or self.lo_streak >= 2:
            self.level = 0
            self.ema = min(self.ema, self.down_thresh)
        elif accepted >= self.jump_acc or self.hi_streak >= 2:
            self.level = top
            self.ema = max(self.ema, self.up_thresh)
        elif self.num_levels >= 3:
            if self.ema >= self.up_thresh:
                self.level = 2
            elif self.ema >= self.mid_thresh:
                self.level = 1
            elif self.ema < self.down_thresh:
                self.level = 0
        else:
            if self.ema >= self.up_thresh:
                self.level = 1
            elif self.ema < self.down_thresh:
                self.level = 0
        return self.level


def _validate_w_act_value(w_act: object, min_w_act: int, width: int) -> int:
    if not _is_int_token(w_act):
        raise VerificationError(f"w_act must be int, got {w_act!r}")
    if w_act < min_w_act or w_act > width:
        raise VerificationError(f"w_act {w_act} outside [{min_w_act}, {width}]")
    if (w_act & 1) != 0 and w_act != width:
        raise VerificationError(f"w_act {w_act} must be even unless equal to width {width}")
    return w_act


def verify_speculative_invariants(
    round_traces: object,
    runtime_invariants: object,
    width: int = 16,
) -> None:
    resolved_width = _validate_width(width, "width")
    if not isinstance(runtime_invariants, dict):
        raise VerificationError("runtime_invariants must be a dict")
    w_min = runtime_invariants.get("w_act_min")
    w_max = runtime_invariants.get("w_act_max")
    if not _is_int_token(w_min) or not _is_int_token(w_max):
        raise VerificationError("w_act_min and w_act_max must be ints")
    if w_min < 1 or w_max > resolved_width or w_min > w_max:
        raise VerificationError(
            f"Invalid w_act bounds [{w_min}, {w_max}] for width {resolved_width}"
        )
    w_history = runtime_invariants.get("w_act_history")
    if not isinstance(w_history, list) or not w_history:
        raise VerificationError("w_act_history must be a non-empty list")
    for w_val in w_history:
        checked_w = _validate_w_act_value(w_val, w_min, resolved_width)
        if checked_w > w_max:
            raise VerificationError(f"w_act_history entry {checked_w} exceeds w_act_max {w_max}")

    collective_rows = runtime_invariants.get("collective_split_max_rows")
    if not _is_int_token(collective_rows) or not (1 <= collective_rows <= MAX_COLLECTIVE_ROWS):
        raise VerificationError(
            f"collective_split_max_rows {collective_rows!r} exceeds {MAX_COLLECTIVE_ROWS}"
        )
    l78_rows = runtime_invariants.get("l78_gemm_chunk_max_rows")
    if not _is_int_token(l78_rows) or not (1 <= l78_rows <= MAX_L78_CHUNK_ROWS):
        raise VerificationError(
            f"l78_gemm_chunk_max_rows {l78_rows!r} exceeds {MAX_L78_CHUNK_ROWS}"
        )
    in_flight_depth = runtime_invariants.get("in_flight_depth")
    if not _is_int_token(in_flight_depth) or in_flight_depth not in (1, 2):
        raise VerificationError(f"in_flight_depth must be 1 or 2, got {in_flight_depth!r}")
    draft_topk = runtime_invariants.get("draft_sfa_topk")
    if not _is_int_token(draft_topk) or draft_topk not in DRAFT_SFA_TOPK_OPTIONS:
        raise VerificationError(
            f"draft_sfa_topk {draft_topk!r} not in {DRAFT_SFA_TOPK_OPTIONS}"
        )
    target_topk = runtime_invariants.get("target_sfa_topk", TARGET_SFA_TOPK)
    if not _is_int_token(target_topk) or target_topk != TARGET_SFA_TOPK:
        raise VerificationError(
            f"target_sfa_topk must be {TARGET_SFA_TOPK}, got {target_topk!r}"
        )

    if not isinstance(round_traces, list) or not round_traces:
        raise VerificationError("round_traces must be a non-empty list")

    horizon_ctrl = ZeroLagHorizonRef()
    last_idx = len(round_traces) - 1
    prev_pos: int | None = None
    for idx, trace in enumerate(round_traces):
        if not isinstance(trace, dict):
            raise VerificationError(f"round_traces[{idx}] must be a dict")
        trace_width = _validate_width(
            trace.get("width", resolved_width), f"round_traces[{idx}].width"
        )
        if trace_width != resolved_width:
            raise VerificationError(
                f"round_traces[{idx}].width {trace_width} != report.width {resolved_width}"
            )
        trace_min_w = trace.get("min_w_act", w_min)
        if not _is_int_token(trace_min_w) or trace_min_w != w_min:
            raise VerificationError(
                f"round_traces[{idx}].min_w_act {trace_min_w!r} != w_act_min {w_min}"
            )
        if "w_act" not in trace:
            raise VerificationError(f"round_traces[{idx}] missing mandatory w_act")
        w_act = _validate_w_act_value(trace["w_act"], w_min, resolved_width)
        if w_act > w_max:
            raise VerificationError(
                f"round_traces[{idx}] w_act {w_act} exceeds w_act_max {w_max}"
            )
        if w_act not in w_history:
            raise VerificationError(
                f"round_traces[{idx}] w_act {w_act} not in w_act_history {w_history}"
            )
        if "raw_lps" in trace:
            trace_t_base = _validate_positive_finite(
                trace.get("t_base_ms", 47.0), f"round_traces[{idx}].t_base_ms"
            )
            trace_c_w = _validate_positive_finite(
                trace.get("c_w_ms", 0.72), f"round_traces[{idx}].c_w_ms"
            )
            expected_w = select_optimal_w_act(
                trace["raw_lps"],
                width=resolved_width,
                min_w_act=w_min,
                t_base_ms=trace_t_base,
                c_w_ms=trace_c_w,
            )
            if w_act != expected_w:
                raise VerificationError(
                    f"round_traces[{idx}] w_act {w_act} != select_optimal_w_act {expected_w}"
                )
        mask = trace.get("mc2_mask")
        if not isinstance(mask, list) or len(mask) != resolved_width:
            raise VerificationError(f"round_traces[{idx}] invalid mc2_mask")
        if not all(
            isinstance(x, bool) or (_is_int_token(x) and x in (0, 1)) for x in mask
        ):
            raise VerificationError(f"round_traces[{idx}] mc2_mask entries must be bool/0/1")
        if not all(bool(x) for x in mask[:w_act]) or any(bool(x) for x in mask[w_act:]):
            raise VerificationError(
                f"round_traces[{idx}] mc2_mask tail not zeroed past w_act={w_act}"
            )
        if trace.get("dcci_issued") is not True:
            raise VerificationError(f"round_traces[{idx}] missing mandatory DCCI barrier")
        c_rows = trace.get("collective_rows")
        if not _is_int_token(c_rows) or not (1 <= c_rows <= collective_rows):
            raise VerificationError(
                f"round_traces[{idx}] collective_rows {c_rows!r} exceeds {collective_rows}"
            )
        g_rows = trace.get("l78_gemm_chunk_rows")
        if not _is_int_token(g_rows) or not (1 <= g_rows <= l78_rows):
            raise VerificationError(
                f"round_traces[{idx}] l78_gemm_chunk_rows {g_rows!r} exceeds {l78_rows}"
            )
        if "kv_commit_q_len" in trace:
            q_len = trace["kv_commit_q_len"]
            if not _is_int_token(q_len) or q_len < 0:
                raise VerificationError(
                    f"round_traces[{idx}] kv_commit_q_len must be non-negative int, got {q_len!r}"
                )
            if q_len == 0 and trace.get("kv_commit_wrote", False) is not False:
                raise VerificationError(
                    f"round_traces[{idx}] RunKvCommit wrote KV when q_len == 0"
                )
        acc = trace.get("accepted")
        if not _is_int_token(acc) or not (1 <= acc <= w_act + 1):
            raise VerificationError(
                f"round_traces[{idx}] invalid accepted count {acc!r} for w_act={w_act}"
            )
        if "final_pos" in trace:
            trace_pos = trace["final_pos"]
            if not _is_int_token(trace_pos) or not (0 <= trace_pos <= FINAL_POS):
                raise VerificationError(
                    f"round_traces[{idx}].final_pos out of bounds: {trace_pos!r}"
                )
            if prev_pos is not None and trace_pos < prev_pos:
                raise VerificationError(
                    f"round_traces[{idx}].final_pos {trace_pos} regressed below {prev_pos}"
                )
            prev_pos = trace_pos
        h_level = trace.get("horizon_level")
        if not _is_int_token(h_level):
            raise VerificationError(f"round_traces[{idx}] missing integer horizon_level")
        expected_level = horizon_ctrl.update(acc)
        if h_level != expected_level:
            raise VerificationError(
                f"round_traces[{idx}] horizon_level {h_level} != expected {expected_level}"
            )
        done = trace.get("done")
        if not isinstance(done, bool):
            raise VerificationError(f"round_traces[{idx}].done must be bool, got {done!r}")
        if idx < last_idx and done:
            raise VerificationError(f"round_traces[{idx}] marked done before final trace")
        if idx == last_idx and not done:
            raise VerificationError("Final round trace must set done=True")

    last_trace = round_traces[-1]
    final_pos = last_trace.get("final_pos")
    if not _is_int_token(final_pos) or final_pos != FINAL_POS:
        raise VerificationError(
            f"Final round trace final_pos {final_pos!r} != {FINAL_POS}"
        )
    if in_flight_depth == 2:
        drain_q_len = last_trace.get("drain_q_len")
        if not _is_int_token(drain_q_len) or drain_q_len != 0:
            raise VerificationError(
                f"Final round trace in_flight=2 drain must set int drain_q_len=0, "
                f"got {drain_q_len!r}"
            )


def _validate_compiled_so_dir(hygiene: object, run_id: str) -> None:
    if not isinstance(hygiene, dict) or "compiled_so_dir" not in hygiene:
        return
    raw_so = hygiene["compiled_so_dir"]
    if not isinstance(raw_so, str) or not raw_so or raw_so != raw_so.strip():
        raise VerificationError(f"Invalid runtime_hygiene.compiled_so_dir: {raw_so!r}")
    if ".." in Path(raw_so).parts or ".." in raw_so.split("/"):
        raise VerificationError(
            f"Path traversal in runtime_hygiene.compiled_so_dir: {raw_so!r}"
        )
    norm_so = posixpath.normpath(raw_so)
    shm_root = f"/dev/shm/{run_id}"
    rel_builds = "tests/.artifacts/builds"
    abs_builds = posixpath.normpath(
        str(Path(__file__).resolve().parents[3] / "tests" / ".artifacts" / "builds")
    )
    in_shm = norm_so == shm_root or norm_so.startswith(f"{shm_root}/")
    in_rel = norm_so == rel_builds or norm_so.startswith(f"{rel_builds}/")
    in_abs = norm_so == abs_builds or norm_so.startswith(f"{abs_builds}/")
    if not (in_shm or in_rel or in_abs):
        raise VerificationError(
            f"Unisolated runtime_hygiene.compiled_so_dir: {raw_so!r}"
        )


def _verify_report_isolation(report: dict[str, Any]) -> None:
    isolation = report.get("isolation")
    if not isinstance(isolation, dict):
        raise VerificationError("Missing isolation object in e2e_result.json")
    raw_run_id = report.get("run_id")
    if not isinstance(raw_run_id, str):
        raise VerificationError(f"report.run_id must be str, got {raw_run_id!r}")
    shm_dir = isolation.get("shm_dir")
    if not isinstance(shm_dir, str) or not shm_dir or shm_dir != shm_dir.strip():
        raise VerificationError(f"Invalid isolation.shm_dir: {shm_dir!r}")
    whole_ep = isolation.get("whole_layer_endpoint")
    if not isinstance(whole_ep, str) or not whole_ep or whole_ep != whole_ep.strip():
        raise VerificationError(f"Invalid isolation.whole_layer_endpoint: {whole_ep!r}")
    onesided_ep = isolation.get("onesided_endpoint")
    if not isinstance(onesided_ep, str) or not onesided_ep or onesided_ep != onesided_ep.strip():
        raise VerificationError(f"Invalid isolation.onesided_endpoint: {onesided_ep!r}")
    hccl_range = isolation.get("hccl_port_range")
    if not isinstance(hccl_range, str) or not hccl_range or hccl_range != hccl_range.strip():
        raise VerificationError(f"Invalid isolation.hccl_port_range: {hccl_range!r}")
    master_port = isolation.get("master_port")
    if not _is_int_token(master_port) or not (1024 <= master_port <= 65535):
        raise VerificationError(f"Invalid isolation.master_port: {master_port!r}")
    try:
        run_id = validate_run_id(raw_run_id)
        verify_shm_isolation(shm_dir, run_id)
        whole_port = _parse_endpoint_port(whole_ep, "isolation.whole_layer_endpoint")
        onesided_port = _parse_endpoint_port(onesided_ep, "isolation.onesided_endpoint")
        lo, hi = _parse_port_range(hccl_range)
    except IsolationError as exc:
        raise VerificationError(str(exc)) from exc
    endpoint_ports = {master_port, whole_port, onesided_port}
    if len(endpoint_ports) != 3 or (endpoint_ports & set(range(lo, hi + 1))):
        raise VerificationError("Isolation endpoint ports collide or overlap hccl_port_range")
    atb_suffix = isolation.get("atb_share_memory_suffix")
    if (
        not isinstance(atb_suffix, str)
        or not atb_suffix.strip()
        or atb_suffix != atb_suffix.strip()
        or run_id not in atb_suffix
        or atb_suffix == "shared_global"
    ):
        raise VerificationError(
            f"Invalid isolation.atb_share_memory_suffix: {atb_suffix!r}"
        )
    _validate_compiled_so_dir(report.get("runtime_hygiene"), run_id)


def verify_e2e_run(
    result_dir: Path,
    case_dir: Path,
    max_tpot_ms: float = TPOT_TARGET_MS,
    require_warm_head: bool = True,
    require_ab_ba: bool = False,
    milestone: str = "M5",
    serial_draft_ref: dict[str, Any] | None = None,
    main_ref: dict[str, Any] | None = None,
    allow_relaxed_parity: bool = False,
    track: str | None = None,
) -> dict[str, Any]:
    _reject_relaxed_parity(allow_relaxed_parity)
    _validate_positive_finite(max_tpot_ms, "max_tpot_ms")
    milestone_tpot_limit(milestone)
    result_path = result_dir / "e2e_result.json"
    if not result_path.is_file():
        raise VerificationError(f"Missing e2e_result.json in {result_dir}")
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise VerificationError(f"Expected JSON object in {result_path}")
    if report.get("milestone") != milestone:
        raise VerificationError(
            f"Report milestone {report.get('milestone')!r} != expected {milestone!r}"
        )
    tspec = _resolve_report_track(report, track)
    ws = tspec.world_size
    case_id = str(report.get("case_id", ""))
    spec = get_case_spec(case_id, track=tspec.track)
    verify_prompt_case(case_dir, spec, track=tspec.track)
    if report.get("prompt_sha256") != spec.prompt_sha256:
        raise VerificationError(
            f"Report prompt_sha256 {report.get('prompt_sha256')!r} != {spec.prompt_sha256!r}"
        )

    _verify_report_isolation(report)

    ranks_manifest = report.get("ranks")
    if not isinstance(ranks_manifest, list) or len(ranks_manifest) != ws:
        raise VerificationError(f"ranks manifest must be a list of length {ws}")
    manifest_by_rank: dict[int, dict[str, Any]] = {}
    for entry in ranks_manifest:
        if not isinstance(entry, dict):
            raise VerificationError("ranks manifest entries must be dicts")
        erank = entry.get("rank")
        if not _is_int_token(erank) or not (0 <= erank < ws) or erank in manifest_by_rank:
            raise VerificationError(f"Invalid or duplicate rank in ranks manifest: {erank!r}")
        manifest_by_rank[erank] = entry

    rank_reports: list[dict[str, Any]] = []
    for rank in range(ws):
        rank_path = result_dir / f"rank-{rank}.json"
        if not rank_path.is_file():
            raise VerificationError(f"Missing {rank_path}")
        entry = manifest_by_rank[rank]
        expected_name = f"rank-{rank}.json"
        if entry.get("report_path") != expected_name:
            raise VerificationError(
                f"Invalid report_path for rank {rank}: "
                f"{entry.get('report_path')!r} != {expected_name!r}"
            )
        expected_rank_sha = entry.get("report_sha256")
        if not is_valid_sha256_hex(expected_rank_sha):
            raise VerificationError(f"Missing/invalid report_sha256 for rank {rank}")
        if file_sha256(rank_path) != expected_rank_sha:
            raise VerificationError(f"Manifest SHA256 mismatch for {rank_path.name}")
        rank_obj = json.loads(rank_path.read_text(encoding="utf-8"))
        if not isinstance(rank_obj, dict):
            raise VerificationError(f"Expected JSON object in {rank_path}")
        file_rank = rank_obj.get("rank")
        if not _is_int_token(file_rank) or file_rank != rank:
            raise VerificationError(
                f"{rank_path.name} rank field {file_rank!r} != expected {rank}"
            )
        rank_reports.append(rank_obj)

    verify_8rank_states(
        rank_reports,
        serial_draft_ref=serial_draft_ref,
        main_ref=main_ref,
        expected_spec=spec,
        allow_relaxed_parity=allow_relaxed_parity,
        world_size=ws,
    )

    ref_invariants = rank_reports[0].get("runtime_invariants")
    width = _validate_width(report.get("width", 16), "report.width")
    for rank_idx, r_data in enumerate(rank_reports):
        if r_data.get("runtime_invariants") != ref_invariants:
            raise VerificationError(f"Rank {rank_idx} runtime_invariants diverges from rank 0")
    round_traces = report.get("round_traces")
    verify_speculative_invariants(round_traces, ref_invariants, width=width)
    assert isinstance(round_traces, list)
    if [t["accepted"] for t in round_traces] != rank_reports[0]["accepted_per_round"]:
        raise VerificationError(
            "round_traces accepted sequence diverges from rank-0 accepted_per_round"
        )
    if any("horizon_levels" in r_data for r_data in rank_reports):
        expected_levels = [t["horizon_level"] for t in round_traces]
        for rank_idx, r_data in enumerate(rank_reports):
            levels = r_data.get("horizon_levels")
            if (
                not isinstance(levels, list)
                or len(levels) != len(expected_levels)
                or any(not _is_int_token(lvl) for lvl in levels)
                or levels != expected_levels
            ):
                raise VerificationError(
                    f"Rank {rank_idx} horizon_levels diverges from round_traces"
                )

    trials = report.get("trials")
    if not isinstance(trials, list) or not trials:
        raise VerificationError("Missing or empty trials list in report")
    for idx, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise VerificationError(f"trials[{idx}] must be a dict")
        is_warmup = trial.get("is_warmup")
        if not isinstance(is_warmup, bool):
            raise VerificationError(
                f"trials[{idx}].is_warmup must be bool, got {is_warmup!r}"
            )
        if not is_warmup:
            trial_acc = trial.get("accepted_per_round")
            if trial_acc is None:
                raise VerificationError(f"trials[{idx}] missing mandatory accepted_per_round")
            verify_output_tokens(
                trial.get("output_token_ids"),
                expected_sha256=spec.output_sha256,
                first_token_id=spec.first_token_id,
                accepted_per_round=trial_acc,
                allow_relaxed_parity=allow_relaxed_parity,
            )
            if trial_acc != rank_reports[0]["accepted_per_round"]:
                raise VerificationError(
                    f"trials[{idx}].accepted_per_round diverges from rank-0 accepted_per_round"
                )
            if trial.get("output_sha256") != spec.output_sha256:
                raise VerificationError(
                    f"trials[{idx}] output_sha256 mismatch: {trial.get('output_sha256')!r}"
                )
            trial_first = trial.get("first_token_id")
            if not _is_int_token(trial_first) or trial_first != spec.first_token_id:
                raise VerificationError(
                    f"trials[{idx}] first_token_id mismatch: {trial_first!r}"
                )

    summary = report.get("summary", {})
    summary_sha = summary.get("output_sha256") if isinstance(summary, dict) else None
    if summary_sha != spec.output_sha256:
        raise VerificationError(
            f"summary.output_sha256 {summary_sha!r} != {spec.output_sha256!r}"
        )
    summary_ekv = summary.get("ekv_serial_diffs") if isinstance(summary, dict) else None
    if summary_ekv != {}:
        raise VerificationError(
            f"summary.ekv_serial_diffs must be empty, got {summary_ekv!r}"
        )
    if "ekv_serial_diffs" in report and report["ekv_serial_diffs"] != {}:
        raise VerificationError(
            f"report.ekv_serial_diffs must be empty, got {report['ekv_serial_diffs']!r}"
        )

    metrics = verify_timing_and_protocol(
        report,
        max_tpot_ms=max_tpot_ms,
        require_warm_head=require_warm_head,
        require_ab_ba=require_ab_ba,
        milestone=milestone,
        track=tspec.track,
    )
    formal_cands = [
        t
        for t in trials
        if not t["is_warmup"] and t.get("arm", "candidate") == "candidate"
    ]
    rank_dts_report = [float(rank_reports[r]["dt_ms"]) for r in range(ws)]
    if not any(
        all(
            abs(rank_dts_report[r] - float(cand["rank_dt_ms"][r])) <= 1e-3
            for r in range(ws)
        )
        for cand in formal_cands
    ):
        raise VerificationError(
            f"rank-{{0..{ws - 1}}}.json dt_ms values do not match any formal candidate trial"
        )
    effective_limit = min(
        _validate_positive_finite(max_tpot_ms, "max_tpot_ms"),
        milestone_tpot_limit(milestone),
    )
    max_rank_tpot = max(rank_dts_report) / float(DECODE_TOKENS)
    if milestone in STRICT_LT_MILESTONES:
        if max_rank_tpot >= effective_limit:
            raise VerificationError(
                f"Rank max TPOT {max_rank_tpot:.4f} ms >= target {effective_limit:.4f} ms"
            )
    elif max_rank_tpot > effective_limit:
        raise VerificationError(
            f"Rank max TPOT {max_rank_tpot:.4f} ms > milestone limit {effective_limit:.4f} ms"
        )
    return {
        "case_id": case_id,
        "track": tspec.track,
        "milestone": milestone,
        "output_sha256": spec.output_sha256,
        **metrics,
    }
