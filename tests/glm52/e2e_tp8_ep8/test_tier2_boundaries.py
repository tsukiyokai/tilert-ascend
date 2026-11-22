from copy import deepcopy
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, TypeGuard, get_type_hints

import pytest

from .a3_runner import build_dry_run_bundle
from .contracts import (
    DECODE_TOKENS,
    FINAL_POS,
    FORBIDDEN_HOSTS,
    GOLDEN_TRACKS,
    HygieneError,
    IsolationError,
    OSL,
    TrackSpec,
    VOCAB_SIZE,
    WORLD_SIZE,
    VerificationError,
    compact_tokens_sha256,
    default_tokens_sha256,
    file_sha256,
    get_case_spec,
    get_track_spec,
    golden_result_path,
    resolve_track,
)
from .hygiene_gate import (
    allocate_isolation_env,
    assert_host_allowed,
    parse_kbenv_output,
    run_kbenv_precheck,
    select_a3_host,
    select_free_die_slice,
    validate_run_id,
    verify_cpp_no_comments,
    verify_isolation_env,
    verify_python_source_hygiene,
    verify_root_hygiene,
    verify_shm_isolation,
)
from .parity_gate import (
    _is_int_token,
    select_optimal_w_act,
    verify_8rank_states,
    verify_e2e_run,
    verify_output_tokens,
    verify_prompt_case,
    verify_speculative_invariants,
    verify_timing_and_protocol,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def _load_ranks(bundle_dir: Path) -> list[dict[str, Any]]:
    return [_load_json(bundle_dir / f"rank-{rank}.json") for rank in range(WORLD_SIZE)]


def test_t2_f1_rejects_bad_prompt_length(tmp_path: Path, case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    prompt_data = _load_json(case_root / "case-01-coding" / "prompt.json")
    for bad_len in (8191, 8193):
        case_dir = tmp_path / f"bad_len_{bad_len}"
        case_dir.mkdir(parents=True)
        mutated = deepcopy(prompt_data)
        mutated["prompt_token_ids"] = (
            mutated["prompt_token_ids"][:8191]
            if bad_len == 8191
            else mutated["prompt_token_ids"] + [1]
        )
        _write_json(case_dir / "prompt.json", mutated)
        with pytest.raises(VerificationError):
            verify_prompt_case(case_dir, spec)


def test_t2_f1_rejects_corrupted_prompt_sha256(tmp_path: Path, case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    prompt_data = _load_json(case_root / "case-01-coding" / "prompt.json")
    case_dir = tmp_path / "bad_sha"
    case_dir.mkdir(parents=True)
    mutated = deepcopy(prompt_data)
    mutated["prompt_token_ids"][10] ^= 1
    _write_json(case_dir / "prompt.json", mutated)
    with pytest.raises(VerificationError):
        verify_prompt_case(case_dir, spec)


def test_t2_f1_rejects_wrong_first_token_id(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(golden_result_path(case_root / "case-01-coding"))
    with pytest.raises(VerificationError):
        verify_output_tokens(
            bench["output_token_ids"],
            expected_sha256=spec.output_sha256,
            first_token_id=99999,
        )


def test_t2_f1_rejects_one_token_flip_in_output(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(golden_result_path(case_root / "case-01-coding"))
    tokens = list(bench["output_token_ids"])
    tokens[42] = (tokens[42] + 1) % VOCAB_SIZE
    with pytest.raises(VerificationError):
        verify_output_tokens(
            tokens,
            expected_sha256=spec.output_sha256,
            first_token_id=spec.first_token_id,
        )


def test_t2_f1_rejects_compact_json_sha_for_output(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(golden_result_path(case_root / "case-01-coding"))
    compact_sha = compact_tokens_sha256(bench["output_token_ids"])
    assert compact_sha != spec.output_sha256
    with pytest.raises(VerificationError):
        verify_output_tokens(
            bench["output_token_ids"],
            expected_sha256=compact_sha,
            first_token_id=spec.first_token_id,
        )


def test_t2_f2_rejects_missing_rank_report(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)[:-1]
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks)


def test_t2_f2_rejects_rank_layer_cache_digest_mismatch(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)
    old_sha = ranks[5]["main_state"]["layers"][22]["latent"]["sha256"]
    flipped = ("0" if old_sha[0] != "0" else "1") + old_sha[1:]
    ranks[5]["main_state"]["layers"][22]["latent"]["sha256"] = flipped
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks)


def test_t2_f2_rejects_wrong_final_position(case01_bundle: Path) -> None:
    for bad_pos in (9214, 9216):
        ranks = _load_ranks(case01_bundle)
        for row in ranks:
            row["main_state"]["position"] = bad_pos
        with pytest.raises(VerificationError):
            verify_8rank_states(ranks)


def test_t2_f2_rejects_serial_draft_mismatch(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)
    ranks[3]["serial_draft_match"] = False
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks)

    ranks_ekv = _load_ranks(case01_bundle)
    ranks_ekv[0]["ekv_serial_diffs"] = {"layer_78": {"max_abs": 1.0}}
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks_ekv)

    ranks_missing_ekv = _load_ranks(case01_bundle)
    ranks_missing_ekv[2].pop("ekv_serial_diffs")
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks_missing_ekv)


def test_t2_f2_rejects_nonempty_ekv_serial_diffs_in_e2e_bundle(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "ekv_nonempty",
        case_root=case_root,
        port_base=32700,
    )
    rep = _load_json(bundle / "e2e_result.json")
    r0_path = bundle / "rank-0.json"
    r0 = _load_json(r0_path)
    r0["ekv_serial_diffs"] = {"layer_78": {"max_abs": 1.0}}
    _write_json(r0_path, r0)
    rep["ranks"][0]["report_sha256"] = file_sha256(r0_path)
    rep["summary"]["ekv_serial_diffs"] = {"layer_78": {"max_abs": 1.0}}
    _write_json(bundle / "e2e_result.json", rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)

    r0["ekv_serial_diffs"] = {}
    _write_json(r0_path, r0)
    rep["ranks"][0]["report_sha256"] = file_sha256(r0_path)
    _write_json(bundle / "e2e_result.json", rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)

    rep["summary"]["ekv_serial_diffs"] = {}
    rep["ekv_serial_diffs"] = {"layer_78": {"max_abs": 1.0}}
    _write_json(bundle / "e2e_result.json", rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)


def test_t2_f2_rejects_shared_draft_and_lm_head(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    report["runtime_hygiene"]["draft_head_distinct_from_lm_head"] = False
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(report)


def test_t2_f3_rejects_insufficient_warmups(case01_bundle: Path) -> None:
    for bad_warm in (0, 1):
        report = _load_json(case01_bundle / "e2e_result.json")
        report["protocol"]["warmups"] = bad_warm
        with pytest.raises(VerificationError):
            verify_timing_and_protocol(report)


def test_t2_f3_rejects_cold_head_or_unrestored_prefix(case01_bundle: Path) -> None:
    cold = _load_json(case01_bundle / "e2e_result.json")
    cold["protocol"]["head_execution_before_timing"] = {
        "wrapped": False,
        "active_graph_width": 16,
        "graph_widths": [12, 24],
    }
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(cold, require_warm_head=True)
    unrestored = _load_json(case01_bundle / "e2e_result.json")
    unrestored["protocol"]["prefix_restored"] = False
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(unrestored)


def test_t2_f3_rejects_profiler_or_sync_debug_enabled(case01_bundle: Path) -> None:
    prof = _load_json(case01_bundle / "e2e_result.json")
    prof["protocol"]["profiler_enabled"] = True
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(prof)
    sync = _load_json(case01_bundle / "e2e_result.json")
    sync["protocol"]["sync_debug_flags"] = ["ASCEND_LAUNCH_BLOCKING=1"]
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(sync)


def test_t2_f3_rejects_1024_divisor_or_rank0_only_tpot(case01_bundle: Path) -> None:
    div1024 = _load_json(case01_bundle / "e2e_result.json")
    for trial in div1024["trials"]:
        trial["max_tpot_ms"] = max(trial["rank_dt_ms"]) / 1024.0
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(div1024)
    rank0_only = _load_json(case01_bundle / "e2e_result.json")
    rank0_only["trials"][2]["rank_dt_ms"][7] = 5.0 * float(DECODE_TOKENS)
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(rank0_only)


def test_t2_f3_rejects_tpot_above_target_or_lt_5_formals(case01_bundle: Path) -> None:
    slow = _load_json(case01_bundle / "e2e_result.json")
    slow_tpots = [3.001, 3.002, 3.003, 3.004, 3.005]
    formal_idx = 0
    for trial in slow["trials"]:
        if not trial["is_warmup"]:
            tpot = slow_tpots[formal_idx]
            trial["rank_dt_ms"] = [tpot * float(DECODE_TOKENS)] * WORLD_SIZE
            trial["max_tpot_ms"] = tpot
            formal_idx += 1
    slow["summary"]["formal_ms"] = slow_tpots
    slow["summary"]["median_tpot_ms"] = 3.003
    slow["summary"]["min_tpot_ms"] = 3.001
    slow["summary"]["p90_tpot_ms"] = 3.0046
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(slow, max_tpot_ms=3.0, milestone="M5")

    few = _load_json(case01_bundle / "e2e_result.json")
    few["protocol"]["formals"] = 4
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(few)


def test_t2_f4_rejects_invalid_w_act_and_unmasked_mc2_tail(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    traces = report["round_traces"]
    rank0 = _load_json(case01_bundle / "rank-0.json")
    bad_inv = deepcopy(rank0["runtime_invariants"])
    bad_inv["w_act_history"] = [13]
    with pytest.raises(VerificationError):
        verify_speculative_invariants(traces, bad_inv, width=16)

    bad_trace = deepcopy(traces)
    bad_trace[0]["w_act"] = 12
    bad_trace[0]["mc2_mask"] = [True] * 16
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_trace, rank0["runtime_invariants"], width=16)


def test_t2_f4_rejects_unsplit_32_row_collective(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    rank0 = _load_json(case01_bundle / "rank-0.json")
    bad_inv = deepcopy(rank0["runtime_invariants"])
    bad_inv["collective_split_max_rows"] = 32
    with pytest.raises(VerificationError):
        verify_speculative_invariants(report["round_traces"], bad_inv, width=16)


def test_t2_f4_rejects_over_8_row_l78_gemm_chunk(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    rank0 = _load_json(case01_bundle / "rank-0.json")
    bad_inv = deepcopy(rank0["runtime_invariants"])
    bad_inv["l78_gemm_chunk_max_rows"] = 16
    with pytest.raises(VerificationError):
        verify_speculative_invariants(report["round_traces"], bad_inv, width=16)


def test_t2_f4_rejects_accepted_sum_not_1023(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(golden_result_path(case_root / "case-01-coding"))
    for bad_sum in ([1022], [1024]):
        with pytest.raises(VerificationError):
            verify_output_tokens(
                bench["output_token_ids"],
                expected_sha256=spec.output_sha256,
                first_token_id=spec.first_token_id,
                accepted_per_round=bad_sum,
            )


def test_t2_f4_rejects_unclean_inflight_drain_or_kv_write_on_noop(
    case01_bundle: Path,
) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    rank0 = _load_json(case01_bundle / "rank-0.json")
    bad_noop = deepcopy(report["round_traces"])
    bad_noop[-1]["kv_commit_q_len"] = 0
    bad_noop[-1]["kv_commit_wrote"] = True
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_noop, rank0["runtime_invariants"], width=16)
    bad_drain = deepcopy(report["round_traces"])
    bad_drain[-1]["drain_q_len"] = 1
    bad_drain[-1]["final_pos"] = FINAL_POS + 1
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_drain, rank0["runtime_invariants"], width=16)


def test_t2_f5_rejects_duplicate_rank_ids(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)
    ranks[7]["rank"] = 0
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks)


def test_t2_f5_rejects_nan_or_inf_rank_dt(case01_bundle: Path) -> None:
    for bad_val in (math.nan, math.inf):
        report = _load_json(case01_bundle / "e2e_result.json")
        report["trials"][2]["rank_dt_ms"][0] = bad_val
        with pytest.raises(VerificationError):
            verify_timing_and_protocol(report)


def test_t2_f5_rejects_out_of_vocab_output_token(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(golden_result_path(case_root / "case-01-coding"))
    for bad_tok in (-1, VOCAB_SIZE):
        tokens = list(bench["output_token_ids"])
        tokens[500] = bad_tok
        with pytest.raises(VerificationError):
            verify_output_tokens(
                tokens,
                expected_sha256=spec.output_sha256,
                first_token_id=spec.first_token_id,
            )


def test_t2_f5_rejects_vllm_loaded_under_m2_plus(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    report["runtime_hygiene"]["loaded_modules_has_vllm"] = True
    for ms in ("M2", "M3", "M4", "M5"):
        with pytest.raises(VerificationError):
            verify_timing_and_protocol(report, milestone=ms)


def test_t2_f5_rejects_empty_expected_sha_or_relaxed_parity(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(golden_result_path(case_root / "case-01-coding"))
    with pytest.raises(VerificationError):
        verify_output_tokens(
            bench["output_token_ids"],
            expected_sha256="",
            first_token_id=spec.first_token_id,
        )
    with pytest.raises(VerificationError):
        verify_output_tokens(
            bench["output_token_ids"],
            expected_sha256=spec.output_sha256,
            first_token_id=spec.first_token_id,
            allow_relaxed_parity=True,
        )


def test_t2_f6_rejects_swapped_case02_case03_outputs(case_root: Path) -> None:
    spec2 = get_case_spec("case-02-review")
    bench3 = _load_json(golden_result_path(case_root / "case-03-refactor"))
    with pytest.raises(VerificationError):
        verify_output_tokens(
            bench3["output_token_ids"],
            expected_sha256=spec2.output_sha256,
            first_token_id=spec2.first_token_id,
        )


def test_t2_f6_rejects_case02_wrong_first_token_id(case_root: Path) -> None:
    spec2 = get_case_spec("case-02-review")
    bench2 = _load_json(golden_result_path(case_root / "case-02-review"))
    with pytest.raises(VerificationError):
        verify_output_tokens(
            bench2["output_token_ids"],
            expected_sha256=spec2.output_sha256,
            first_token_id=13534,
        )


def test_t2_f6_rejects_missing_case04_bench_result(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case4_copy = tmp_path / "case-04-math"
    case4_copy.mkdir(parents=True)
    shutil.copy2(case_root / "case-04-math" / "prompt.json", case4_copy / "prompt.json")
    with pytest.raises(VerificationError):
        verify_e2e_run(tmp_path / "missing_bundle", case4_copy)


def test_t2_f6_rejects_tampered_case01_golden_file(
    tmp_path: Path,
    case_root: Path,
) -> None:
    spec1 = get_case_spec("case-01-coding")
    tampered_dir = tmp_path / "case-01-tampered"
    tampered_dir.mkdir(parents=True)
    shutil.copy2(case_root / "case-01-coding" / "prompt.json", tampered_dir / "prompt.json")
    bench = _load_json(golden_result_path(case_root / "case-01-coding"))
    bench["output_token_ids"][10] ^= 1
    out_golden = golden_result_path(tampered_dir)
    out_golden.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_golden, bench)
    with pytest.raises(VerificationError):
        verify_prompt_case(tampered_dir, spec1)


def test_t2_f6_rejects_tp16_world_size(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    report["world_size"] = 16
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(report)


def test_t2_f6_rejects_cross_track_mismatch_and_unknown_track(
    case01_bundle: Path,
    case_root: Path,
) -> None:
    with pytest.raises(VerificationError):
        get_track_spec("unknown-track")
    with pytest.raises(VerificationError):
        get_case_spec("case-02-review", track="a3-tp16-ep16")
    with pytest.raises(VerificationError):
        resolve_track("a3-tp16-ep16", world_size=8)
    with pytest.raises(VerificationError):
        resolve_track("a3-tp8-ep8", world_size=16)
    with pytest.raises(VerificationError):
        verify_e2e_run(case01_bundle, case_root / "case-01-coding", track="a3-tp16-ep16")


def test_t2_f7_rejects_ineligible_kbenv_hosts() -> None:
    sample = "\n".join(
        [
            (
                "  7.150.10.78    A3  Ascend910_9392  ========··  84% "
                "················ 16/16  84% 预热   否"
            ),
            (
                "  7.150.10.79    A3  Ascend910_9392  ========··  84% "
                "#########·······  7/16  84% 部分   否"
            ),
            (
                "  1.95.7.108     A2  910B3           ========··  84% "
                "········          8/8   84% 空闲   长期"
            ),
            (
                "  7.150.10.80    A3  Ascend910_9392  ==········  20% "
                "########          0/8   84% 占用   否"
            ),
        ]
    )
    hosts = parse_kbenv_output(sample)
    assert len(hosts) == 4
    with pytest.raises(IsolationError):
        select_a3_host(hosts, min_free_dies=8)


def test_t2_f7_rejects_forbidden_hosts() -> None:
    for bad_ip in FORBIDDEN_HOSTS:
        with pytest.raises(IsolationError):
            assert_host_allowed(bad_ip)
        with pytest.raises(IsolationError):
            allocate_isolation_env("run_bad", target_host=bad_ip)


def test_t2_f7_rejects_shared_shm_and_missing_atb_suffix() -> None:
    env = allocate_isolation_env("run_shared")
    bad_shm = dict(env)
    bad_shm["TILERT_JOB_DIR"] = "/dev/shm/agy_tp8/control"
    with pytest.raises(IsolationError):
        verify_isolation_env(bad_shm)

    bad_atb = dict(env)
    bad_atb["ATB_SHARE_MEMORY_NAME_SUFFIX"] = ""
    with pytest.raises(IsolationError):
        verify_isolation_env(bad_atb)


def test_t2_f7_rejects_overlapping_ports_across_concurrent_runs() -> None:
    env_a = allocate_isolation_env("run_a", port_base=29100)
    env_b = allocate_isolation_env("run_b", port_base=29200)
    with pytest.raises(IsolationError):
        verify_isolation_env(env_a, peer_envs=[env_b])


def test_t2_f7_rejects_python_comment_docstring_and_cpp_comment(
    tmp_path: Path,
) -> None:
    py_comment = tmp_path / "has_comment.py"
    py_comment.write_text("x = 1  # forbidden\n", encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_python_source_hygiene([py_comment])

    py_doc = tmp_path / "has_doc.py"
    py_doc.write_text('"""forbidden docstring"""\nx = 1\n', encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_python_source_hygiene([py_doc])

    cpp_comment = tmp_path / "has_cpp.cpp"
    cpp_comment.write_text("int f() { // bad\n return 0; }\n", encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([cpp_comment])

    bad_root = tmp_path / "fake_repo"
    bad_root.mkdir()
    (bad_root / "scratch.log").write_text("junk\n", encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_root_hygiene(bad_root)


def test_t2_x1_x2_rejects_empty_none_or_incomplete_round_traces(
    tmp_path: Path,
    case01_bundle: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    work_dir = tmp_path / "x1_x2_copy"
    shutil.copytree(case01_bundle, work_dir)
    orig_report_bytes = (case01_bundle / "e2e_result.json").read_bytes()
    for bad_traces in ([], None, "bad"):
        report = _load_json(work_dir / "e2e_result.json")
        report["round_traces"] = bad_traces
        _write_json(work_dir / "e2e_result.json", report)
        with pytest.raises(VerificationError):
            verify_e2e_run(work_dir, case_dir)
    assert (case01_bundle / "e2e_result.json").read_bytes() == orig_report_bytes

    valid_traces = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fresh_x2",
        case_root=case_root,
        port_base=31300,
    )
    fresh_report = _load_json(valid_traces / "e2e_result.json")
    fresh_report["round_traces"][0].pop("dcci_issued")
    _write_json(valid_traces / "e2e_result.json", fresh_report)
    with pytest.raises(VerificationError):
        verify_e2e_run(valid_traces, case_dir)

    fresh_acc_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fresh_x2_acc",
        case_root=case_root,
        port_base=31600,
    )
    loaded = _load_json(fresh_acc_dir / "e2e_result.json")
    loaded["round_traces"][0]["accepted"] += 1
    _write_json(fresh_acc_dir / "e2e_result.json", loaded)
    with pytest.raises(VerificationError):
        verify_e2e_run(fresh_acc_dir, case_dir)


def test_t2_x3_x7_x9_x10_rejects_protocol_and_trial_bypasses(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "abba_proto",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        port_base=31900,
    )
    bad_order = _load_json(bundle / "e2e_result.json")
    bad_order["protocol"]["paired_order"] = "ab_ab"
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(bad_order, require_ab_ba=False)

    bad_warm_type = _load_json(bundle / "e2e_result.json")
    bad_warm_type["trials"][0]["is_warmup"] = "false"
    _write_json(bundle / "e2e_result.json", bad_warm_type)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)

    bundle2 = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "abba_counts",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        port_base=32200,
    )
    bad_counts = _load_json(bundle2 / "e2e_result.json")
    bad_counts["protocol"]["warmups"] = 10
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(bad_counts, require_ab_ba=True)

    bad_delta = _load_json(bundle2 / "e2e_result.json")
    bad_delta["summary"]["ab_ba_delta_ms"] = -10.0
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(bad_delta, require_ab_ba=True)


def test_t2_x4_x5_x6_x8_rejects_width_history_tolerance_and_horizon_drift(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "x4568",
        case_root=case_root,
        port_base=32500,
    )
    report = _load_json(bundle / "e2e_result.json")
    rank0 = _load_json(bundle / "rank-0.json")

    bad_width = deepcopy(report)
    bad_width["width"] = 13
    _write_json(bundle / "e2e_result.json", bad_width)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)
    _write_json(bundle / "e2e_result.json", report)

    bad_inv = deepcopy(rank0["runtime_invariants"])
    bad_inv["w_act_max"] = 12
    bad_inv["w_act_history"] = [16]
    with pytest.raises(VerificationError):
        verify_speculative_invariants(report["round_traces"], bad_inv, width=16)

    loophole = deepcopy(report)
    true_tpots = [3.00005] * 5
    f_idx = 0
    for trial in loophole["trials"]:
        if not trial["is_warmup"]:
            trial["rank_dt_ms"] = [true_tpots[f_idx] * float(DECODE_TOKENS)] * WORLD_SIZE
            trial["max_tpot_ms"] = 2.99996
            f_idx += 1
    loophole["summary"]["formal_ms"] = [2.99996] * 5
    loophole["summary"]["median_tpot_ms"] = 2.99996
    loophole["summary"]["min_tpot_ms"] = 2.99996
    loophole["summary"]["p90_tpot_ms"] = 2.99996
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(loophole, max_tpot_ms=3.0, milestone="M5")

    rank4 = _load_json(bundle / "rank-4.json")
    rank4["horizon_levels"][0] = (rank4["horizon_levels"][0] + 1) % 3
    _write_json(bundle / "rank-4.json", rank4)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)


def test_t2_r1_f4_rejects_rank_case_or_prompt_mismatch_and_typeguard(
    tmp_path: Path,
    case_root: Path,
) -> None:
    hints = get_type_hints(_is_int_token)
    assert hints["return"] is TypeGuard[int]
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "rank_meta",
        case_root=case_root,
        port_base=32800,
    )
    ranks = _load_ranks(bundle)
    ranks[2]["case_id"] = "case-02-review"
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks, expected_spec=get_case_spec("case-01-coding"))


def test_t2_b1_b2_b3_rejects_9382_nonstrict_ipv4_glyph_mismatch_and_ram0() -> None:
    for bad_host in (
        "010.83.185.129",
        "root@10.83.185.129",
        "10.83.185.129:22",
        "10.83.185.1290",
        "mujica",
        "pi-mujica",
    ):
        with pytest.raises(IsolationError):
            assert_host_allowed(bad_host)

    bad_kbenv = "\n".join(
        [
            "  7.150.99.10 A3 Ascend910_9382 ========·· 84% ················ 16/16 84% 空闲 长期",
            "  10.83.185.1290 A3 Ascend910_9392 ========·· 84% ················ 16/16 84% 空闲 长期",
            "  010.83.185.129 A3 Ascend910_9392 ========·· 84% ················ 16/16 84% 空闲 长期",
            "  7.150.99.12 A3 Ascend910_9392 ====······ 40% ····############ 8/16 84% 部分 否",
            "  7.150.99.13 A3 Ascend910_9392 ====······ 40% ········######## 8/16 0% 部分 否",
        ]
    )
    hosts = parse_kbenv_output(bad_kbenv)
    with pytest.raises(IsolationError):
        select_a3_host(hosts, min_free_dies=8)

    prefixed = (
        "> 7.156.98.198 A3 Ascend910_9392 ========·· 84% ··#············· 15/16 84% 部分 否\n"
    )
    parsed = parse_kbenv_output(prefixed)
    assert len(parsed) == 1
    assert select_free_die_slice(parsed[0].free_die_indices, 8) == tuple(range(8, 16))


def test_t2_b4_b5_rejects_shm_traversal_bad_run_id_and_port_bypass(
    tmp_path: Path,
    case_root: Path,
) -> None:
    for bad_id in ("agy_tp8", ".", "..", "../bad", "ab"):
        with pytest.raises(IsolationError):
            validate_run_id(bad_id)
    for bad_shm in ("/dev/shm/run_1/../../agy_tp8", "/dev/shm/agy_tp8/run_1", "/dev/shm/run_10"):
        with pytest.raises(IsolationError):
            verify_shm_isolation(bad_shm, "run_1")

    env = allocate_isolation_env("run_b5_ok", port_base=33100, target_host="7.150.12.20")
    zero_prof = {**env, "PROFILING_MODE": "0", "ASCEND_LAUNCH_BLOCKING": "0"}
    verify_isolation_env(zero_prof, target_host="7.150.12.20")

    bad_prof = {**env, "PROFILING_MODE": "1"}
    with pytest.raises(IsolationError):
        verify_isolation_env(bad_prof, target_host="7.150.12.20")

    for k, bad_val in (
        ("MASTER_PORT", "tcp://127.0.0.1:29900"),
        ("MASTER_PORT", "٢٩٩٠٠"),
        ("TILERT_WHOLE_LAYER_ENDPOINT", "29901"),
        ("TILERT_RUN_DIR", "/dev/shm/run_b5_ok_other"),
    ):
        with pytest.raises(IsolationError):
            verify_isolation_env({**env, k: bad_val})

    with pytest.raises(IsolationError):
        verify_isolation_env(env, peer_envs=[{"TILERT_E2E_RUN_ID": "run_peer_missing_ports"}])

    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "bad_iso_report",
        case_root=case_root,
        port_base=33400,
    )
    bad_rep = _load_json(bundle / "e2e_result.json")
    bad_rep["isolation"]["master_port"] = bad_rep["isolation"]["master_port"] + 1
    _write_json(bundle / "e2e_result.json", bad_rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_root / "case-01-coding")


def test_t2_b6_b7_cpp_raw_string_and_deep_py_and_tilert_artifact_hygiene(
    tmp_path: Path,
) -> None:
    raw_ok = tmp_path / "raw_ok.cpp"
    raw_ok.write_text(
        'const char* a = R"(say "hi // not a comment)";\n'
        'const char* b = R"tag(ok " ) // still raw)tag";\n',
        encoding="utf-8",
    )
    verify_cpp_no_comments([raw_ok])

    raw_comment = tmp_path / "raw_comment.cpp"
    raw_comment.write_text(
        'const char* s = R"(say "hi)"; // REAL COMMENT\n',
        encoding="utf-8",
    )
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([raw_comment])

    hidden_str = tmp_path / "hidden_str.py"
    hidden_str.write_text('x = 1\n"hidden comment"\n', encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_python_source_hygiene([hidden_str])

    fake_repo = tmp_path / "fake_tilert"
    fake_repo.mkdir()
    for name in (
        ".flake8",
        ".github",
        ".gitignore",
        ".pre-commit-config.yaml",
        "Dockerfile",
        "LICENSE",
        "README.md",
        "assets",
        "docs",
        "pyproject.toml",
        "requirements-dev.txt",
        "requirements.txt",
        "tests",
    ):
        (fake_repo / name).mkdir()
    scripts_dir = fake_repo / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "project.py").write_text(
        'import json\nprint(json.dumps({"pass": True, "staged_files": 0}))\n',
        encoding="utf-8",
    )
    tilert_dir = fake_repo / "tilert"
    tilert_dir.mkdir()
    (tilert_dir / "evil_untracked.so").write_bytes(b"elf")
    with pytest.raises(HygieneError):
        verify_root_hygiene(fake_repo)


def test_t2_b10_b11_b12_rejects_low_counts_nan_tpot_and_missing_hccl(
    tmp_path: Path,
    case_root: Path,
) -> None:
    with pytest.raises(VerificationError):
        build_dry_run_bundle(
            case_id="case-01-coding",
            output_dir=tmp_path / "low_warm",
            case_root=case_root,
            warmups=1,
            formals=5,
        )
    with pytest.raises(VerificationError):
        build_dry_run_bundle(
            case_id="case-01-coding",
            output_dir=tmp_path / "low_form",
            case_root=case_root,
            warmups=2,
            formals=4,
        )
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "b11_b12",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        port_base=33700,
    )
    report = _load_json(bundle / "e2e_result.json")
    for bad_limit in (math.nan, math.inf, 0.0, -1.0):
        with pytest.raises(VerificationError):
            verify_timing_and_protocol(report, max_tpot_ms=bad_limit)
    missing_hccl = deepcopy(report)
    missing_hccl["protocol"].pop("hccl_op_expansion_mode")
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(missing_hccl, require_ab_ba=True)
    missing_det = deepcopy(report)
    missing_det["protocol"].pop("hccl_deterministic")
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(missing_det, require_ab_ba=True)


def test_t2_p1_rejects_manifest_report_path_traversal_mismatch_or_missing(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "p1_bundle",
        case_root=case_root,
        port_base=34300,
    )
    orig = _load_json(bundle / "e2e_result.json")
    for bad_path in ("../../../../etc/passwd", "rank-7.json", "", None):
        mut = deepcopy(orig)
        if bad_path is None:
            mut["ranks"][0].pop("report_path")
        else:
            mut["ranks"][0]["report_path"] = bad_path
        _write_json(bundle / "e2e_result.json", mut)
        with pytest.raises(VerificationError):
            verify_e2e_run(bundle, case_dir)


def test_t2_p2_p4_rejects_bool_float_equivalence_and_blank_dtype(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "p2_p4",
        case_root=case_root,
        port_base=34600,
    )
    orig_report = _load_json(bundle / "e2e_result.json")
    orig_rank0 = _load_json(bundle / "rank-0.json")
    orig_rank1 = _load_json(bundle / "rank-1.json")

    bad_inv = deepcopy(orig_rank0["runtime_invariants"])
    bad_inv["in_flight_depth"] = True
    bad_drain = deepcopy(orig_report["round_traces"])
    bad_drain[-1]["drain_q_len"] = 99
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_drain, bad_inv, width=16)

    bad_drain_false = deepcopy(orig_report["round_traces"])
    bad_drain_false[-1]["drain_q_len"] = False
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_drain_false, orig_rank0["runtime_invariants"], width=16)

    bad_done_int = deepcopy(orig_report["round_traces"])
    bad_done_int[0]["done"] = 1
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_done_int, orig_rank0["runtime_invariants"], width=16)

    bad_final_pos = deepcopy(orig_report["round_traces"])
    bad_final_pos[-1]["final_pos"] = 9215.0
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_final_pos, orig_rank0["runtime_invariants"], width=16)

    ranks = _load_ranks(bundle)
    ranks[1]["accepted_per_round"][-1] = True
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks, expected_spec=get_case_spec("case-01-coding"))

    ranks_tok = _load_ranks(bundle)
    one_idx = ranks_tok[1]["output_token_ids"].index(1)
    ranks_tok[1]["output_token_ids"][one_idx] = True
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks_tok, expected_spec=get_case_spec("case-01-coding"))

    ranks_pos = _load_ranks(bundle)
    ranks_pos[0]["main_state"]["position"] = 9215.0
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks_pos)

    ranks_dtype = _load_ranks(bundle)
    ranks_dtype[0]["draft_state"]["layers"][0]["latent"]["dtype"] = "   "
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks_dtype)

    bad_rank1 = deepcopy(orig_rank1)
    bad_rank1["horizon_levels"] = [
        bool(v) if v in (0, 1) else v for v in bad_rank1["horizon_levels"]
    ]
    if all(isinstance(v, int) and not isinstance(v, bool) for v in bad_rank1["horizon_levels"]):
        bad_rank1["horizon_levels"][0] = False
        mut_rep = deepcopy(orig_report)
        mut_rep["round_traces"][0]["horizon_level"] = 0
    else:
        mut_rep = deepcopy(orig_report)
    _write_json(bundle / "rank-1.json", bad_rank1)
    mut_rep["ranks"][1]["report_sha256"] = file_sha256(bundle / "rank-1.json")
    _write_json(bundle / "e2e_result.json", mut_rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)


def test_t2_p2ijkl_rejects_trace_bounds_drift_and_nan_select_optimal_w_act(
    tmp_path: Path,
    case_root: Path,
) -> None:
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "p2ijkl",
        case_root=case_root,
        port_base=34900,
    )
    report = _load_json(bundle / "e2e_result.json")
    inv = _load_json(bundle / "rank-0.json")["runtime_invariants"]

    bad_min = deepcopy(report["round_traces"])
    bad_min[0]["min_w_act"] = 2
    bad_min[0]["w_act"] = 2
    bad_min[0]["mc2_mask"] = [i < 2 for i in range(16)]
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_min, inv, width=16)

    bad_width = deepcopy(report["round_traces"])
    bad_width[0]["width"] = 24
    bad_width[0]["w_act"] = 24
    bad_width[0]["mc2_mask"] = [True] * 24
    with pytest.raises(VerificationError):
        verify_speculative_invariants(bad_width, inv, width=16)

    bad_hist = deepcopy(inv)
    bad_hist["w_act_min"] = 10
    bad_hist["w_act_max"] = 16
    bad_hist["w_act_history"] = [12]
    with pytest.raises(VerificationError):
        verify_speculative_invariants(report["round_traces"], bad_hist, width=16)

    for bad_lps in ([0.0] + [math.nan] * 15, [0.0] + [math.inf] * 15):
        with pytest.raises(VerificationError):
            select_optimal_w_act(bad_lps, width=16, min_w_act=10)
    for bad_param in (math.nan, math.inf, -1.0, 0.0):
        with pytest.raises(VerificationError):
            select_optimal_w_act([0.0] * 16, width=16, min_w_act=10, t_base_ms=bad_param)
        with pytest.raises(VerificationError):
            select_optimal_w_act([0.0] * 16, width=16, min_w_act=10, c_w_ms=bad_param)


def test_t2_p3abc_rejects_unpaired_abba_missing_trial_acc_and_bad_warmups_ms(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "p3_abba",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        port_base=35200,
    )
    orig = _load_json(bundle / "e2e_result.json")

    diluted = deepcopy(orig)
    extra_cand = deepcopy(
        [t for t in diluted["trials"] if not t["is_warmup"] and t["arm"] == "candidate"][0]
    )
    extra_cand.pop("pair_idx")
    extra_cand["rank_dt_ms"] = [1.5 * float(DECODE_TOKENS)] * WORLD_SIZE
    extra_cand["max_tpot_ms"] = 1.5
    extra_base = deepcopy(
        [t for t in diluted["trials"] if not t["is_warmup"] and t["arm"] == "baseline"][0]
    )
    extra_base.pop("pair_idx")
    diluted["trials"].extend([extra_cand, extra_base])
    diluted["protocol"]["formals"] = 6
    diluted["summary"]["formal_ms"].append(1.5)
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(diluted, require_ab_ba=True, milestone="M5")

    no_acc = deepcopy(orig)
    for t in no_acc["trials"]:
        if not t["is_warmup"]:
            t.pop("accepted_per_round")
            break
    _write_json(bundle / "e2e_result.json", no_acc)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir, require_ab_ba=True)

    bad_warm = deepcopy(orig)
    bad_warm["summary"]["warmups_ms"] = [-999.0, "bad"]
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(bad_warm, require_ab_ba=True)


def test_t2_n1_n5_cpp_digit_sep_line_splice_and_py_standalone_bom(tmp_path: Path) -> None:
    bad_sep = tmp_path / "sep.cpp"
    bad_sep.write_text("int x = 1'000; // HIDDEN\n", encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([bad_sep])

    bad_hex = tmp_path / "hex.cpp"
    bad_hex.write_text("int x = 0xFF'FF; /* HIDDEN */\n", encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([bad_hex])

    bad_splice = tmp_path / "splice.cpp"
    bad_splice.write_text("/\\\n/ HIDDEN\n", encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([bad_splice])

    good_cpp = tmp_path / "good.cpp"
    good_cpp.write_text(
        "int a = 1'000;\nchar c = u8'a';\nconst char* s = u8R\"(// ok)\";\n",
        encoding="utf-8",
    )
    verify_cpp_no_comments([good_cpp])

    for name, src in (
        ("fstr.py", 'def f() -> int:\n    f"hidden {1}"\n    return 1\n'),
        ("bstr.py", 'def f() -> int:\n    b"hidden"\n    return 1\n'),
        ("tstr.py", 'def f() -> int:\n    ("a", "b")\n    return 1\n'),
        ("syn.py", "def f(:\n"),
    ):
        p = tmp_path / name
        p.write_text(src, encoding="utf-8")
        with pytest.raises(HygieneError):
            verify_python_source_hygiene([p])

    bom = tmp_path / "bom.py"
    bom.write_bytes(b"\xef\xbb\xbfx = 1\n")
    with pytest.raises(HygieneError):
        verify_python_source_hygiene([bom])


def test_t2_n3_n4_endpoint_loopback_shm_kbenv_timeout_and_16die_invariants(
    tmp_path: Path,
    case_root: Path,
) -> None:
    env = allocate_isolation_env("run_n3_ok", port_base=35500, target_host="7.150.12.20")
    for bad_ep in ("tcp://10.83.185.129:35501", "tcp://0.0.0.0:35501", "tcp://7.150.12.20:35501"):
        with pytest.raises(IsolationError):
            verify_isolation_env({**env, "TILERT_WHOLE_LAYER_ENDPOINT": bad_ep})
    with pytest.raises(IsolationError):
        verify_isolation_env({**env, "VLLM_CACHE_ROOT": "/dev/shm/agy_tp8/vllm_cache"})
    with pytest.raises(IsolationError):
        verify_isolation_env({**env, "ATB_SHARE_MEMORY_NAME_SUFFIX": "  glm52_run_n3_ok  "})
    peer_bad = allocate_isolation_env("run_n3_peer", port_base=35900)
    peer_bad["TILERT_RUN_DIR"] = "/dev/shm/agy_tp8"
    with pytest.raises(IsolationError):
        verify_isolation_env(env, peer_envs=[peer_bad])

    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "n3_rep",
        case_root=case_root,
        port_base=36200,
    )
    rep = _load_json(bundle / "e2e_result.json")
    rep["isolation"]["whole_layer_endpoint"] = "tcp://10.83.185.129:36201"
    _write_json(bundle / "e2e_result.json", rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_root / "case-01-coding")

    for bad_row in (
        "7.150.99.25 A3 Ascend910_9392 ========·· 84% ········ 8/8 84% 空闲 长期",
        (
            "7.150.99.25 A3 Ascend910_9392 ========·· 84% "
            "################················ 16/16 84% 空闲 长期"
        ),
        "7.150.99.25 A3 Ascend910_9392 ========·· 84% ????????········ 8/16 84% 空闲 长期",
        "7.150.99.25 A3 Ascend910_9392 ========·· 84% ················ ١٦/١٦ ٨٤% 空闲 长期",
    ):
        with pytest.raises(IsolationError):
            select_a3_host(parse_kbenv_output(bad_row), min_free_dies=8)
    with pytest.raises(IsolationError):
        select_free_die_slice(tuple(range(16, 24)), 8)

    glyph_ok = "7.150.99.26 A3 Ascend910_9392 ====······ 84% +=×#·········### 9/16 84% 部分 否"
    parsed_ok = parse_kbenv_output(glyph_ok)
    assert len(parsed_ok) == 1
    assert select_a3_host(parsed_ok, min_free_dies=8).ip == "7.150.99.26"

    slow_kbenv = tmp_path / "slow_kbenv.py"
    slow_kbenv.write_text("import time; time.sleep(5)\n", encoding="utf-8")
    with pytest.raises(IsolationError, match="timed out"):
        run_kbenv_precheck(
            kbenv_bin=sys.executable,
            extra_args=[str(slow_kbenv)],
            timeout=0.1,
        )


def test_t2_fb_trial_order_dup_idx_and_non_contiguous_abba_rejected(
    tmp_path: Path,
    case_root: Path,
) -> None:
    single_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fb_single",
        case_root=case_root,
        port_base=36500,
    )
    single_rep = _load_json(single_dir / "e2e_result.json")
    warms = [t for t in single_rep["trials"] if t["is_warmup"]]
    forms = [t for t in single_rep["trials"] if not t["is_warmup"]]
    reordered = deepcopy(single_rep)
    reordered["trials"] = forms + warms
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(reordered)

    dup_idx = deepcopy(single_rep)
    for trial in dup_idx["trials"]:
        trial["trial_idx"] = 0
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(dup_idx)

    abba_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fb_abba",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        port_base=36800,
    )
    abba_rep = _load_json(abba_dir / "e2e_result.json")
    abba_warm = [t for t in abba_rep["trials"] if t["is_warmup"]]
    abba_form = [t for t in abba_rep["trials"] if not t["is_warmup"]]
    form_first = deepcopy(abba_rep)
    form_first["trials"] = abba_form + abba_warm
    for idx, trial in enumerate(form_first["trials"]):
        trial["trial_idx"] = idx
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(form_first, require_ab_ba=True)

    split_pairs = deepcopy(abba_rep)
    split_pairs["trials"][6], split_pairs["trials"][8] = (
        split_pairs["trials"][8],
        split_pairs["trials"][6],
    )
    for idx, trial in enumerate(split_pairs["trials"]):
        trial["trial_idx"] = idx
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(split_pairs, require_ab_ba=True)


def test_t2_fd_fe_ff_cache_shape_rank_identity_dt_ms_and_head_width(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fd_fe_ff",
        case_root=case_root,
        port_base=37100,
    )
    orig_rep = _load_json(bundle / "e2e_result.json")

    bad_head = deepcopy(orig_rep)
    bad_head["protocol"]["head_execution_before_timing"] = {
        "wrapped": False,
        "active_graph_width": 12,
        "graph_widths": [12],
    }
    _write_json(bundle / "e2e_result.json", bad_head)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)
    _write_json(bundle / "e2e_result.json", orig_rep)

    bad_shape_rep = deepcopy(orig_rep)
    for r in range(WORLD_SIZE):
        rank_file = bundle / f"rank-{r}.json"
        row = _load_json(rank_file)
        for layer in row["main_state"]["layers"]:
            for meta in layer.values():
                meta["shape"] = [1, 1]
        for layer in row["draft_state"]["layers"]:
            for meta in layer.values():
                meta["shape"] = [1, 1]
        _write_json(rank_file, row)
        bad_shape_rep["ranks"][r]["report_sha256"] = file_sha256(rank_file)
    _write_json(bundle / "e2e_result.json", bad_shape_rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir)

    bundle2 = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fd_fe_2",
        case_root=case_root,
        port_base=37400,
    )
    orig2 = _load_json(bundle2 / "e2e_result.json")
    slow_ranks_rep = deepcopy(orig2)
    for r in range(WORLD_SIZE):
        rank_file = bundle2 / f"rank-{r}.json"
        row = _load_json(rank_file)
        row["dt_ms"] = 50000.0
        row["tpot_ms"] = 50000.0 / float(DECODE_TOKENS)
        _write_json(rank_file, row)
        slow_ranks_rep["ranks"][r]["report_sha256"] = file_sha256(rank_file)
    _write_json(bundle2 / "e2e_result.json", slow_ranks_rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle2, case_dir)

    bundle3 = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fe_swap",
        case_root=case_root,
        port_base=37700,
    )
    r0_path = bundle3 / "rank-0.json"
    r1_path = bundle3 / "rank-1.json"
    r0_bytes = r0_path.read_bytes()
    r1_bytes = r1_path.read_bytes()
    r0_path.write_bytes(r1_bytes)
    r1_path.write_bytes(r0_bytes)
    swap_rep = _load_json(bundle3 / "e2e_result.json")
    swap_rep["ranks"][0]["report_sha256"] = file_sha256(r0_path)
    swap_rep["ranks"][1]["report_sha256"] = file_sha256(r1_path)
    _write_json(bundle3 / "e2e_result.json", swap_rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle3, case_dir)

    r0_path.write_text("[1, 2, 3]\n", encoding="utf-8")
    swap_rep["ranks"][0]["report_sha256"] = file_sha256(r0_path)
    _write_json(bundle3 / "e2e_result.json", swap_rep)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle3, case_dir)


def test_t2_fc_fg_fa_speculative_bounds_and_strict_isolation_typing(
    tmp_path: Path,
    case_root: Path,
) -> None:
    case_dir = case_root / "case-01-coding"
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fc_fg_fa",
        case_root=case_root,
        port_base=38000,
    )
    orig = _load_json(bundle / "e2e_result.json")
    inv = _load_json(bundle / "rank-0.json")["runtime_invariants"]

    absurd_trace = deepcopy(orig["round_traces"][:1])
    absurd_trace[0]["accepted"] = 1023
    absurd_trace[0]["done"] = True
    absurd_trace[0]["final_pos"] = FINAL_POS
    absurd_trace[0]["drain_q_len"] = 0
    absurd_trace[0]["horizon_level"] = 2
    with pytest.raises(VerificationError):
        verify_speculative_invariants(absurd_trace, inv, width=16)

    tight_inv = deepcopy(inv)
    tight_inv["collective_split_max_rows"] = 4
    with pytest.raises(VerificationError):
        verify_speculative_invariants(orig["round_traces"], tight_inv, width=16)

    tight_l78 = deepcopy(inv)
    tight_l78["l78_gemm_chunk_max_rows"] = 2
    with pytest.raises(VerificationError):
        verify_speculative_invariants(orig["round_traces"], tight_l78, width=16)

    neg_q = deepcopy(orig["round_traces"])
    neg_q[0]["kv_commit_q_len"] = -5
    with pytest.raises(VerificationError):
        verify_speculative_invariants(neg_q, inv, width=16)

    neg_pos = deepcopy(orig["round_traces"])
    neg_pos[0]["final_pos"] = -999
    with pytest.raises(VerificationError):
        verify_speculative_invariants(neg_pos, inv, width=16)

    regressed_pos = deepcopy(orig["round_traces"])
    regressed_pos[1]["final_pos"] = regressed_pos[0]["final_pos"] - 1
    with pytest.raises(VerificationError):
        verify_speculative_invariants(regressed_pos, inv, width=16)

    for bad_id in (None, True, 1234):
        bad_run = deepcopy(orig)
        bad_run["run_id"] = bad_id
        bad_run["isolation"]["shm_dir"] = f"/dev/shm/{bad_id}"
        bad_run["isolation"]["atb_share_memory_suffix"] = f"glm52_{bad_id}"
        _write_json(bundle / "e2e_result.json", bad_run)
        with pytest.raises(VerificationError):
            verify_e2e_run(bundle, case_dir)

    for iso_key in ("shm_dir", "whole_layer_endpoint", "onesided_endpoint", "hccl_port_range"):
        ws_rep = deepcopy(orig)
        ws_rep["isolation"][iso_key] = f" {ws_rep['isolation'][iso_key]} "
        _write_json(bundle / "e2e_result.json", ws_rep)
        with pytest.raises(VerificationError):
            verify_e2e_run(bundle, case_dir)

    for bad_so in ("/tmp/shared_unisolated", "/dev/shm/other_run/builds", ""):
        so_rep = deepcopy(orig)
        so_rep["runtime_hygiene"]["compiled_so_dir"] = bad_so
        _write_json(bundle / "e2e_result.json", so_rep)
        with pytest.raises(VerificationError):
            verify_e2e_run(bundle, case_dir)

    ok_so = deepcopy(orig)
    ok_so["runtime_hygiene"]["compiled_so_dir"] = "tests/.artifacts/builds/m5"
    _write_json(bundle / "e2e_result.json", ok_so)
    assert verify_e2e_run(bundle, case_dir)["case_id"] == "case-01-coding"


def test_t2_r1_r2_r4_r6_visible_devices_cpp_splice_kbenv_and_py_ast(
    tmp_path: Path,
) -> None:
    with pytest.raises(IsolationError):
        allocate_isolation_env("run_r1_oob", port_base=38300, visible_devices=range(16, 24))

    env_a = allocate_isolation_env(
        "run_r1_a",
        port_base=38600,
        target_host="7.150.99.10",
        visible_devices=range(8),
    )
    env_b_same = allocate_isolation_env(
        "run_r1_b",
        port_base=38900,
        target_host="7.150.99.10",
        visible_devices=range(8),
    )
    with pytest.raises(IsolationError):
        verify_isolation_env(env_a, peer_envs=[env_b_same], target_host="7.150.99.10")

    env_b_split = allocate_isolation_env(
        "run_r1_c",
        port_base=39200,
        target_host="7.150.99.10",
        visible_devices=range(8, 16),
    )
    verify_isolation_env(env_a, peer_envs=[env_b_split], target_host="7.150.99.10")

    env_no_host_1 = allocate_isolation_env("run_r1_d", port_base=39500, visible_devices=range(8))
    env_no_host_2 = allocate_isolation_env("run_r1_e", port_base=39800, visible_devices=range(8))
    with pytest.raises(IsolationError):
        verify_isolation_env(env_no_host_1, peer_envs=[env_no_host_2])

    for bad_vis in ("16,17,18,19,20,21,22,23", "0,1,2,3,4,5,6", "0, 1,2,3,4,5,6,7", ""):
        with pytest.raises(IsolationError):
            verify_isolation_env({**env_a, "ASCEND_RT_VISIBLE_DEVICES": bad_vis})
    no_vis = dict(env_a)
    no_vis.pop("ASCEND_RT_VISIBLE_DEVICES")
    with pytest.raises(IsolationError):
        verify_isolation_env(no_vis)

    for key, padded in (
        ("TILERT_RUN_DIR", f"  {env_a['TILERT_RUN_DIR']}\n"),
        ("MASTER_PORT", f" {env_a['MASTER_PORT']}\n"),
        ("TILERT_WHOLE_LAYER_ENDPOINT", f" {env_a['TILERT_WHOLE_LAYER_ENDPOINT']} "),
        ("TILERT_ONESIDED_ENDPOINT", f" {env_a['TILERT_ONESIDED_ENDPOINT']} "),
        ("HCCL_NPU_SOCKET_PORT_RANGE", f" {env_a['HCCL_NPU_SOCKET_PORT_RANGE']} "),
        ("TILERT_TARGET_HOST", " 7.150.99.10 "),
    ):
        with pytest.raises(IsolationError):
            verify_isolation_env({**env_a, key: padded})

    raw_splice = tmp_path / "raw_splice.cpp"
    raw_splice.write_text(
        'const char* s = R"(a)\\\n" )"; // REAL COMMENT\nconst char* t = R"(x)";\n',
        encoding="utf-8",
    )
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([raw_splice])

    unclosed_str = tmp_path / "unclosed.cpp"
    unclosed_str.write_text('const char* s = "unclosed; // HIDDEN\n', encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([unclosed_str])

    unclosed_eof = tmp_path / "unclosed_eof.cpp"
    unclosed_eof.write_text('const char* s = "unclosed', encoding="utf-8")
    with pytest.raises(HygieneError):
        verify_cpp_no_comments([unclosed_eof])

    for bad_timeout in (math.nan, math.inf, -math.inf):
        with pytest.raises(IsolationError):
            run_kbenv_precheck(kbenv_bin=sys.executable, timeout=bad_timeout)

    empty_map_row = "7.150.99.25 A3 Ascend910_9392 ========·· 84% 0/16 84% 空闲 长期"
    with pytest.raises(IsolationError):
        select_a3_host(parse_kbenv_output(empty_map_row), min_free_dies=0)
    with pytest.raises(IsolationError):
        select_a3_host(parse_kbenv_output(empty_map_row), min_free_dies=17)

    for name, src in (
        ("dict_str.py", '{"note": "hidden comment"}\n'),
        ("sub_str.py", '("hidden",)[0]\n'),
        ("ifexp_str.py", '("hidden" if True else "x")\n'),
        ("boolop_str.py", '"a" or "b"\n'),
        ("cmp_str.py", '"a" == "b"\n'),
    ):
        py_file = tmp_path / name
        py_file.write_text(src, encoding="utf-8")
        with pytest.raises(HygieneError):
            verify_python_source_hygiene([py_file])

    bare_cr = tmp_path / "bare_cr.py"
    bare_cr.write_bytes(b"x = 1\ry = 2\n")
    with pytest.raises(HygieneError):
        verify_python_source_hygiene([bare_cr])

    call_ok = tmp_path / "call_ok.py"
    call_ok.write_text('print("ok")\n', encoding="utf-8")
    verify_python_source_hygiene([call_ok])


def test_t2_m21_fast_parity_rejected_by_formal_gate_and_protocol_helper(
    tmp_path: Path,
    case_root: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    from .cli import main as cli_main

    monkeypatch.syspath_prepend(str(repo_root))
    runner_mod = importlib.import_module("tests.glm52.run_m2_case01")
    assert runner_mod.loaded_modules_has_vllm() is False
    assert runner_mod.resolve_trial_protocol(None, None, True, False) == (0, 1, True)
    assert runner_mod.resolve_trial_protocol(None, None, False, False) == (2, 5, False)
    assert runner_mod.resolve_trial_protocol(1, 2, True, False) == (1, 2, True)
    with pytest.raises(ValueError):
        runner_mod.resolve_trial_protocol(-1, 1, True, False)
    with pytest.raises(ValueError):
        runner_mod.resolve_trial_protocol(0, 0, True, False)

    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "fast_parity_bundle",
        case_root=case_root,
        warmups=2,
        formals=5,
        port_base=39100,
    )
    report = _load_json(bundle / "e2e_result.json")
    formal_trial = deepcopy(report["trials"][2])
    formal_trial["trial_idx"] = 0
    formal_trial["is_warmup"] = False
    report["protocol"]["warmups"] = 0
    report["protocol"]["formals"] = 1
    report["trials"] = [formal_trial]
    report["summary"]["warmups_ms"] = []
    report["summary"]["formal_ms"] = [formal_trial["max_tpot_ms"]]
    report["summary"]["median_tpot_ms"] = formal_trial["max_tpot_ms"]
    report["summary"]["min_tpot_ms"] = formal_trial["max_tpot_ms"]
    report["summary"]["p90_tpot_ms"] = formal_trial["max_tpot_ms"]
    _write_json(bundle / "e2e_result.json", report)

    with pytest.raises(VerificationError):
        verify_timing_and_protocol(report)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_root / "case-01-coding")
    rc = cli_main(
        [
            "--mode",
            "verify-report",
            "--case",
            "case-01-coding",
            "--case-root",
            str(case_root),
            "--report-dir",
            str(bundle),
        ]
    )
    assert rc == 1


def test_t2_m21_a3_container_isolation_fleet_prestage_and_hygiene(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(repo_root))
    from tests.glm52 import a3_container as ac

    hosts = parse_kbenv_output(
        "\n".join(
            [
                (
                    "  7.150.11.91 A3 Ascend910_9392 ====······ 41% "
                    "················ 16/16 41% 空闲 否"
                ),
                (
                    "  7.150.14.84 A3 Ascend910_9392 =====····· 51% "
                    "················ 16/16 51% 占用 否"
                ),
                (
                    "  10.83.185.129 A3 Ascend910_9392 =========· 89% "
                    "················ 16/16 89% 空闲 长期"
                ),
            ]
        )
    )
    sel, dies = ac.select_host_and_dies(hosts, die_slice="8..15")
    assert sel.ip == "7.150.11.91"
    assert dies == tuple(range(8, 16))
    slots = ac.select_fleet_slots(hosts, dual_slot=True)
    assert slots == [
        ("7.150.11.91", tuple(range(8))),
        ("7.150.11.91", tuple(range(8, 16))),
    ]

    ok, _ = ac.check_prestage_capacity(
        shm_free_bytes=500 * 1024**3,
        mem_total_bytes=1000 * 1024**3,
        mem_avail_bytes=450 * 1024**3,
        payload_bytes=100 * 1024**3,
    )
    assert ok is True
    low_ram, reason_ram = ac.check_prestage_capacity(
        shm_free_bytes=500 * 1024**3,
        mem_total_bytes=1000 * 1024**3,
        mem_avail_bytes=200 * 1024**3,
        payload_bytes=100 * 1024**3,
    )
    assert low_ram is False and "RAM" in reason_ram
    low_shm, reason_shm = ac.check_prestage_capacity(
        shm_free_bytes=50 * 1024**3,
        mem_total_bytes=1000 * 1024**3,
        mem_avail_bytes=450 * 1024**3,
        payload_bytes=30 * 1024**3,
    )
    assert low_shm is False and "/dev/shm" in reason_shm

    run_cmd = ac.build_docker_run_cmd("run_test_m21", tuple(range(8, 16)))
    assert "--init" in run_cmd
    assert "--network=host" in run_cmd
    assert "--ipc=host" in run_cmd
    assert "--cgroupns=host" in run_cmd
    assert "--device=/dev/davinci8" in run_cmd
    assert "--device=/dev/davinci15" in run_cmd
    assert "--device=/dev/davinci0" not in run_cmd
    assert ac.DEFAULT_IMAGE in run_cmd

    exec_cmd = ac.build_docker_exec_cmd("run_test_m21", "echo hi")
    assert exec_cmd[:4] == ["docker", "exec", "run_test_m21", "bash"]
    assert "unset PROFILER_SAMPLECONFIG PROFILING_MODE ASCEND_LAUNCH_BLOCKING" in exec_cmd[5]
    assert "cd /dev/shm/run_test_m21/tmp" in exec_cmd[5]
    assert "timeout 5 test -e /work/model-raw" in exec_cmd[5]

    for bad_host in FORBIDDEN_HOSTS:
        with pytest.raises(ac.IsolationError):
            ac.select_host_and_dies(hosts, target_host=bad_host)
    for bad_id in ("agy_tp8", "../oops"):
        with pytest.raises(ac.IsolationError):
            ac.build_docker_run_cmd(bad_id, tuple(range(8)))
        with pytest.raises(ac.IsolationError):
            ac.build_teardown_cmd(bad_id)

    iso_env = allocate_isolation_env("run_test_m21", port_base=29900, target_host="7.150.11.91")
    env_sh = ac.build_container_env_script(iso_env)
    assert "export TILERT_BUILD_DIR=/dev/shm/run_test_m21/builds" in env_sh
    assert "export WORLD_SIZE=8" in env_sh
    assert "unset PROFILER_SAMPLECONFIG PROFILING_MODE ASCEND_LAUNCH_BLOCKING" in env_sh

    verify_python_source_hygiene(
        [
            repo_root / "tests/glm52/a3_container.py",
            repo_root / "tests/glm52/run_m1_case01.py",
            repo_root / "tests/glm52/run_m2_case01.py",
            Path(__file__),
        ],
        max_line_length=100,
    )


def _setup_tp16_synth_bundle(
    tmp_path: Path,
    case_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tag: str,
    port_base: int,
) -> tuple[Path, Path]:
    synth_tokens = [13534] + [11] * (OSL - 1)
    synth_sha = default_tokens_sha256(synth_tokens)
    tspec = get_track_spec("a3-tp16-ep16")
    monkeypatch.setitem(
        GOLDEN_TRACKS,
        "a3-tp16-ep16",
        TrackSpec(
            track="a3-tp16-ep16",
            chip=tspec.chip,
            world_size=16,
            ep_size=16,
            reduction_tree=(8, 4, 2, 1),
            output_sha256_by_case={"case-01-coding": synth_sha},
        ),
    )
    cases_tmp = tmp_path / f"{tag}_cases"
    c1 = cases_tmp / "case-01-coding"
    golden_p = golden_result_path(c1, "a3-tp16-ep16")
    golden_p.parent.mkdir(parents=True)
    shutil.copy2(case_root / "case-01-coding" / "prompt.json", c1 / "prompt.json")
    prompt_sha = get_case_spec("case-01-coding", track="a3-tp16-ep16").prompt_sha256
    _write_json(
        golden_p,
        {
            "prompt_sha256": prompt_sha,
            "first_token_id": 13534,
            "output_sha256": synth_sha,
            "output_token_ids": synth_tokens,
        },
    )
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / f"{tag}_bundle",
        case_root=cases_tmp,
        milestone="M5",
        track="a3-tp16-ep16",
        port_base=port_base,
    )
    return bundle, c1


def test_t2_f1_f6_tp16_rejects_tp8_sha_and_tampered_tp16_golden(
    tmp_path: Path,
    case_root: Path,
) -> None:
    tp8_bench = _load_json(golden_result_path(case_root / "case-01-coding", "a3-tp8-ep8"))
    spec16 = get_case_spec("case-01-coding", track="a3-tp16-ep16")
    with pytest.raises(VerificationError):
        verify_output_tokens(
            tp8_bench["output_token_ids"],
            expected_sha256=spec16.output_sha256,
            first_token_id=13534,
        )

    c1 = tmp_path / "c1_tp8_in_tp16"
    c1_golden = golden_result_path(c1, "a3-tp16-ep16")
    c1_golden.parent.mkdir(parents=True)
    shutil.copy2(case_root / "case-01-coding" / "prompt.json", c1 / "prompt.json")
    _write_json(c1_golden, tp8_bench)
    with pytest.raises(VerificationError):
        verify_prompt_case(c1, spec16, track="a3-tp16-ep16")

    bad_first = deepcopy(tp8_bench)
    bad_first["first_token_id"] = 13535
    bad_first["output_token_ids"][0] = 13535
    _write_json(c1_golden, bad_first)
    with pytest.raises(VerificationError):
        verify_prompt_case(c1, spec16, track="a3-tp16-ep16")

    for oos_case in ("case-02-review", "case-03-refactor", "case-04-math"):
        with pytest.raises(VerificationError):
            get_case_spec(oos_case, track="a3-tp16-ep16")
    with pytest.raises(VerificationError):
        resolve_track(None, world_size=32)


def test_t2_f2_tp16_rejects_15_or_17_ranks_dup_rank_and_rank15_divergence(
    tmp_path: Path,
    case_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _ = _setup_tp16_synth_bundle(tmp_path, case_root, monkeypatch, "r16", 32100)
    ranks = [_load_json(bundle / f"rank-{r}.json") for r in range(16)]
    spec16 = get_case_spec("case-01-coding", track="a3-tp16-ep16")

    with pytest.raises(VerificationError):
        verify_8rank_states(ranks[:15], expected_spec=spec16, world_size=16)
    with pytest.raises(VerificationError):
        verify_8rank_states(ranks + [deepcopy(ranks[0])], expected_spec=spec16, world_size=16)

    dup_ranks = deepcopy(ranks)
    dup_ranks[15]["rank"] = 9
    with pytest.raises(VerificationError):
        verify_8rank_states(dup_ranks, expected_spec=spec16, world_size=16)

    ws_ranks = deepcopy(ranks)
    ws_ranks[15]["world_size"] = 8
    with pytest.raises(VerificationError):
        verify_8rank_states(ws_ranks, expected_spec=spec16, world_size=16)

    sdp_ranks = deepcopy(ranks)
    sdp_ranks[15]["serial_draft_match"] = False
    with pytest.raises(VerificationError):
        verify_8rank_states(sdp_ranks, expected_spec=spec16, world_size=16)

    ekv_ranks = deepcopy(ranks)
    ekv_ranks[15]["ekv_serial_diffs"] = {"layer_77": {"max_abs": 1e-3}}
    with pytest.raises(VerificationError):
        verify_8rank_states(ekv_ranks, expected_spec=spec16, world_size=16)

    sha_ranks = deepcopy(ranks)
    old_sha = sha_ranks[15]["main_state"]["layers"][77]["latent"]["sha256"]
    sha_ranks[15]["main_state"]["layers"][77]["latent"]["sha256"] = (
        ("0" if old_sha[0] != "0" else "1") + old_sha[1:]
    )
    with pytest.raises(VerificationError):
        verify_8rank_states(sha_ranks, expected_spec=spec16, world_size=16)

    tok_ranks = deepcopy(ranks)
    tok_ranks[15]["output_token_ids"][10] ^= 1
    with pytest.raises(VerificationError):
        verify_8rank_states(tok_ranks, expected_spec=spec16, world_size=16)


def test_t2_f3_tp16_rejects_rank15_straggler_exact_3ms_and_wrong_formula(
    tmp_path: Path,
    case_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, case_dir = _setup_tp16_synth_bundle(tmp_path, case_root, monkeypatch, "t16", 32400)
    orig = _load_json(bundle / "e2e_result.json")

    exact3 = deepcopy(orig)
    for trial in exact3["trials"]:
        if not trial["is_warmup"]:
            trial["rank_dt_ms"] = [3.0 * float(DECODE_TOKENS)] * 16
            trial["max_tpot_ms"] = 3.0
    exact3["summary"]["formal_ms"] = [3.0] * 5
    exact3["summary"]["median_tpot_ms"] = 3.0
    exact3["summary"]["min_tpot_ms"] = 3.0
    exact3["summary"]["p90_tpot_ms"] = 3.0
    for ms in ("M4", "M5"):
        with pytest.raises(VerificationError):
            verify_timing_and_protocol(exact3, max_tpot_ms=3.0, milestone=ms, track="a3-tp16-ep16")

    near3 = deepcopy(orig)
    for trial in near3["trials"]:
        if not trial["is_warmup"]:
            trial["rank_dt_ms"] = [2.9999 * float(DECODE_TOKENS)] * 16
            trial["max_tpot_ms"] = 2.9999
    near3["summary"]["formal_ms"] = [2.9999] * 5
    near3["summary"]["median_tpot_ms"] = 2.9999
    near3["summary"]["min_tpot_ms"] = 2.9999
    near3["summary"]["p90_tpot_ms"] = 2.9999
    assert (
        verify_timing_and_protocol(
            near3, max_tpot_ms=3.0, milestone="M5", track="a3-tp16-ep16"
        )["median_tpot_ms"]
        < 3.0
    )

    straggler = deepcopy(orig)
    for trial in straggler["trials"]:
        if not trial["is_warmup"]:
            dts = [2.4 * float(DECODE_TOKENS)] * 15 + [3.05 * float(DECODE_TOKENS)]
            trial["rank_dt_ms"] = dts
            trial["max_tpot_ms"] = 3.05
    straggler["summary"]["formal_ms"] = [3.05] * 5
    straggler["summary"]["median_tpot_ms"] = 3.05
    straggler["summary"]["min_tpot_ms"] = 3.05
    straggler["summary"]["p90_tpot_ms"] = 3.05
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(
            straggler, max_tpot_ms=3.0, milestone="M5", track="a3-tp16-ep16"
        )
    _write_json(bundle / "e2e_result.json", straggler)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir, max_tpot_ms=3.0, milestone="M5", track="a3-tp16-ep16")

    bad_len = deepcopy(orig)
    bad_len["trials"][2]["rank_dt_ms"] = [2.5 * float(DECODE_TOKENS)] * 8
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(bad_len, milestone="M5", track="a3-tp16-ep16")

    bad_formula = deepcopy(orig)
    bad_formula["protocol"]["tpot_formula"] = "max(8-rank dt_ms) / 1023.0"
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(bad_formula, milestone="M5", track="a3-tp16-ep16")

    bad_vllm = deepcopy(orig)
    bad_vllm["runtime_hygiene"]["loaded_modules_has_vllm"] = True
    for ms in ("M2", "M3", "M4", "M5"):
        with pytest.raises(VerificationError):
            verify_timing_and_protocol(bad_vllm, milestone=ms, track="a3-tp16-ep16")


def test_t2_f5_tp16_rejects_missing_or_tampered_rank15_json_and_8_manifest(
    tmp_path: Path,
    case_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, case_dir = _setup_tp16_synth_bundle(tmp_path, case_root, monkeypatch, "m16", 32700)
    orig = _load_json(bundle / "e2e_result.json")

    trunc = deepcopy(orig)
    trunc["ranks"] = trunc["ranks"][:8]
    _write_json(bundle / "e2e_result.json", trunc)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir, milestone="M5", track="a3-tp16-ep16")

    _write_json(bundle / "e2e_result.json", orig)
    r15_path = bundle / "rank-15.json"
    r15_data = _load_json(r15_path)
    r15_tampered = deepcopy(r15_data)
    r15_tampered["dt_ms"] = 1.0
    _write_json(r15_path, r15_tampered)
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir, milestone="M5", track="a3-tp16-ep16")

    r15_path.unlink()
    with pytest.raises(VerificationError):
        verify_e2e_run(bundle, case_dir, milestone="M5", track="a3-tp16-ep16")


def test_t2_f7_tp16_rejects_partial_15_die_host_and_8die_env() -> None:
    partial_kbenv = (
        "  7.156.98.198 A3 Ascend910_9392 ========·· 84% "
        "··#············· 15/16 84% 部分 否\n"
    )
    hosts = parse_kbenv_output(partial_kbenv)
    assert select_a3_host(hosts, min_free_dies=8).ip == "7.156.98.198"
    with pytest.raises(IsolationError):
        select_a3_host(hosts, min_free_dies=16)
    with pytest.raises(IsolationError):
        select_free_die_slice(tuple(range(15)), count=16)
    with pytest.raises(IsolationError):
        allocate_isolation_env(
            "run_bad16", port_base=33000, world_size=16, visible_devices=range(8)
        )
    env8 = allocate_isolation_env("run_ok8", port_base=33300, world_size=8)
    with pytest.raises(IsolationError):
        verify_isolation_env(env8, world_size=16)


def test_t2_f5_run_m2_case01_tp16_runtime_config_and_flag_conflicts(
    tmp_path: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(repo_root))
    rm2 = importlib.import_module("tests.glm52.run_m2_case01")
    p = rm2.build_parser()

    a_nf = p.parse_args(["--mtp-prefix", "m", "--ckpt", "k", "--fast-parity", "--no-fast-parity"])
    assert a_nf.fast_parity is False
    assert rm2.resolve_trial_protocol(
        a_nf.warmups,
        a_nf.formals,
        a_nf.fast_parity,
        a_nf.serial_draft_parity,
        report_dir=a_nf.report_dir,
        strict_formal=True,
    ) == (2, 5, True)

    a_fast_nosdp = p.parse_args(
        ["--mtp-prefix", "m", "--ckpt", "k", "--fast-parity", "--no-serial-draft-parity"]
    )
    assert rm2.resolve_trial_protocol(
        a_fast_nosdp.warmups,
        a_fast_nosdp.formals,
        a_fast_nosdp.fast_parity,
        a_fast_nosdp.serial_draft_parity,
        report_dir=a_fast_nosdp.report_dir,
        strict_formal=True,
    ) == (0, 1, False)

    a_rep = p.parse_args(
        [
            "--mtp-prefix",
            "m",
            "--ckpt",
            "k",
            "--fast-parity",
            "--no-serial-draft-parity",
            "--report-dir",
            str(tmp_path / "rep"),
        ]
    )
    assert rm2.resolve_trial_protocol(
        a_rep.warmups,
        a_rep.formals,
        a_rep.fast_parity,
        a_rep.serial_draft_parity,
        report_dir=a_rep.report_dir,
        strict_formal=True,
    ) == (2, 5, True)

    for bad_args in (
        (-1, 1, True, True, None, True),
        (0, 0, True, True, None, True),
        (1, 5, False, True, None, True),
        (2, 4, False, True, None, True),
        (1, 5, True, False, str(tmp_path), True),
    ):
        with pytest.raises(ValueError):
            rm2.resolve_trial_protocol(
                bad_args[0],
                bad_args[1],
                bad_args[2],
                bad_args[3],
                report_dir=bad_args[4],
                strict_formal=bad_args[5],
            )

    if "torch" not in sys.modules:
        fake_torch: Any = ModuleType("torch")
        fake_torch.__file__ = "/opt/fake_site/torch/__init__.py"
        fake_torch.Tensor = object
        fake_torch.bfloat16 = "bf16"
        fake_torch.float16 = "f16"
        fake_torch.float32 = "f32"
        fake_torch.int8 = "i8"
        fake_torch.int32 = "i32"
        fake_torch.int64 = "i64"
        fake_torch.uint8 = "u8"
        fake_torch.nn = SimpleNamespace(Module=object)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.delenv("TILERT_BUILD_DIR", raising=False)
    bm = importlib.import_module("tilert.models.glm_5_2_ascend.ops.build_megakernel")
    ops_mod = importlib.import_module("tilert.models.glm_5_2_ascend.dspine.build_dspine_ops")
    dspine_pkg = importlib.import_module("tilert.models.glm_5_2_ascend.dspine")
    seen_ws: list[int] = []

    def fake_build_chain(
        *,
        out_dir: Path | None = None,
        union_sfa: bool = False,
        enable_kernel_prof: bool = False,
        world_size: int = 8,
    ) -> tuple[Path, Path]:
        del union_sfa, enable_kernel_prof
        seen_ws.append(world_size)
        d = bm.default_build_dir(world_size) if out_dir is None else Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        so = d / "chain.so"
        so.write_bytes(b"tp16")
        return so, d / "chain.manifest.json"

    monkeypatch.setattr(bm, "build_chain", fake_build_chain)
    monkeypatch.setattr(
        ops_mod,
        "build_dspine_ops",
        lambda out, world_size=8: (Path(out) / f"op_{world_size}.so", None),
    )
    monkeypatch.setattr(
        dspine_pkg,
        "build_draft_l78_k110419",
        lambda out_dir=None, k_sfa=1024, world_size=8: (
            Path(out_dir) / f"k19_{world_size}_{k_sfa}.so",
            Path(out_dir) / "k20.so",
        ),
    )
    wl_base = tmp_path / "base16.json"
    _write_json(wl_base, {"tree": "orig"})
    build_dir, wl_json, _, _ = rm2._prepare_build_artifacts(
        SimpleNamespace(
            build_dir=None,
            union_sfa=False,
            enable_kernel_prof=False,
            whole_layer_json=None,
            whole_layer_base=str(wl_base),
            k_sfa=1024,
        ),
        rank=0,
    )
    assert seen_ws == [16]
    assert build_dir.name == "megakernel_tp16_ep16"
    assert _load_json(wl_json)["tree"].endswith("megakernel_tp16_ep16/chain.so")


def test_t2_f6_a3_container_tp16_rejects_8die_slice_and_partial_host(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(repo_root))
    ac = importlib.import_module("tests.glm52.a3_container")

    with pytest.raises(ac.IsolationError):
        ac.parse_die_slice("0..7", world_size=16)
    with pytest.raises(ac.IsolationError):
        ac.parse_die_slice("0..15", world_size=8)
    with pytest.raises(ac.IsolationError):
        ac.build_docker_run_cmd("run_bad_tp16", tuple(range(8)), world_size=16)

    hosts_15 = parse_kbenv_output(
        "  7.156.98.198 A3 Ascend910_9392 ========·· 84% ··#············· 15/16 84% 部分 否\n"
    )
    with pytest.raises(ac.IsolationError):
        ac.select_host_and_dies(hosts_15, world_size=16)

    hosts_8 = parse_kbenv_output(
        "  7.150.12.20 A3 Ascend910_9392 ====······ 40% ········######## 8/16 84% 部分 否\n"
    )
    assert ac.select_fleet_slots(hosts_8, world_size=16) == []


def _fake_npu_smi(
    hbm_by_die: dict[int, tuple[int, int]],
    busy_npus: tuple[tuple[int, int], ...] = (),
) -> str:
    lines = [
        "| NPU Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |",
        "| Chip Phy-ID | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |",
    ]
    for d in range(16):
        npu_id, chip_id = divmod(d, 2)
        used, tot = hbm_by_die.get(d, (3000, 65536))
        health = "Alarm" if d == 0 else "OK"
        lines.append(f"| {npu_id} Ascend910 | {health} | 165.0 35 0 / 0 |")
        lines.append(f"| {chip_id} {d} | 0000:9D:00.0 | 0 0 / 0 {used} / {tot} |")
    lines.append("| NPU Chip | Process id | Process name | Process memory(MB) |")
    for npu_id, chip_id in busy_npus:
        lines.append(f"| {npu_id} {chip_id} | 12345 | python3 | 4096 |")
    if not busy_npus:
        lines.append("| No running processes found in NPU 0 |")
    return "\n".join(lines)


def test_t2_alarm_dies_opt_in_guardrails(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(repo_root))
    ac = importlib.import_module("tests.glm52.a3_container")
    hosts_alarm = parse_kbenv_output(
        "  7.150.10.78 A3 Ascend910_9392 ·········· 1% ×··············· 15/16 46% 占用 否\n"
    )
    with pytest.raises(ac.IsolationError):
        ac.select_host_and_dies(hosts_alarm, target_host="7.150.10.78", world_size=16)
    with pytest.raises(ac.IsolationError):
        ac.select_host_and_dies(hosts_alarm, world_size=16, allow_alarm_dies=True)

    hosts_busy = parse_kbenv_output(
        "  7.150.10.79 A3 Ascend910_9392 ·········· 1% #··············· 15/16 46% 占用 否\n"
    )
    with pytest.raises(ac.IsolationError):
        ac.select_host_and_dies(
            hosts_busy,
            target_host="7.150.10.79",
            world_size=16,
            allow_alarm_dies=True,
        )

    sel, dies = ac.select_host_and_dies(
        hosts_alarm,
        target_host="7.150.10.78",
        world_size=16,
        allow_alarm_dies=True,
    )
    assert sel.ip == "7.150.10.78"
    assert dies == tuple(range(16))

    ok_smi = _fake_npu_smi({0: (5242, 65536)})
    pcts = ac.verify_npu_smi_dies(ok_smi, dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT)
    assert pcts[0] <= 8.0

    high_hbm_smi = _fake_npu_smi({0: (5244, 65536)})
    with pytest.raises(ac.IsolationError, match="HBM usage"):
        ac.verify_npu_smi_dies(high_hbm_smi, dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT)

    busy_proc_smi = _fake_npu_smi({0: (3000, 65536)}, busy_npus=((0, 0),))
    with pytest.raises(ac.IsolationError, match="running NPU processes"):
        ac.verify_npu_smi_dies(busy_proc_smi, dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT)

    missing_die_smi = _fake_npu_smi({}).replace("| 1 15 |", "| 1 99 |").splitlines()
    missing_die_smi = "\n".join(ln for ln in missing_die_smi if "| 1 99 |" not in ln)
    with pytest.raises(ac.IsolationError, match="telemetry"):
        ac.verify_npu_smi_dies(missing_die_smi, dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT)

    class _Res:
        def __init__(self, out: str) -> None:
            self.stdout = out.encode("utf-8")

    monkeypatch.setattr(ac, "_kbenv_hosts", lambda _bin: hosts_alarm)

    cur_smi = ok_smi

    def _fake_ssh(ip: str, cmd: str, check: bool = True) -> _Res:
        if cmd == "npu-smi info":
            return _Res(cur_smi)
        return _Res(
            "---DF---\nFilesystem 1B-blocks Used Available Use% Mounted on\n"
            "tmpfs 200000000000 1000 1000000000000 1% /dev/shm\n"
            "---MEM---\nMemTotal: 1000 kB\nMemAvailable: 900 kB\n---PS---\n"
        )

    monkeypatch.setattr(ac, "remote_ssh", _fake_ssh)
    probe_out = ac.probe_host("7.150.10.78", world_size=16, allow_alarm_dies=True)
    assert probe_out["selected_dies"] == list(range(16))

    cur_smi = high_hbm_smi
    with pytest.raises(ac.IsolationError, match="HBM usage"):
        ac.probe_host("7.150.10.78", world_size=16, allow_alarm_dies=True)

    with pytest.raises(ac.IsolationError):
        ac.probe_host("7.150.10.78", world_size=16, allow_alarm_dies=False)

    for dmap in ("+···············", "×+··············"):
        hosts_occ = parse_kbenv_output(
            f"  7.150.10.79 A3 Ascend910_9392 ·········· 1% {dmap} 15/16 46% 占用 否\n"
        )
        with pytest.raises(ac.IsolationError):
            ac.select_host_and_dies(
                hosts_occ,
                target_host="7.150.10.79",
                world_size=16,
                allow_alarm_dies=True,
            )

    no_proc_tbl = ok_smi.split("| NPU Chip | Process id |")[0]
    with pytest.raises(ac.IsolationError, match="Process id"):
        ac.verify_npu_smi_dies(no_proc_tbl, dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT)

    na_hbm_smi = ok_smi.replace("0 0 / 0 5242 / 65536", "0 128 / 4096 NA / NA")
    with pytest.raises(ac.IsolationError, match="telemetry"):
        ac.verify_npu_smi_dies(na_hbm_smi, dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT)

    dual_lines = [
        "| NPU Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |",
        "| Chip Device | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |",
    ]
    for npu_id in range(8):
        health = "Alarm" if npu_id == 0 else "OK"
        dual_lines.append(f"| {npu_id} Ascend910 | {health} | 165.0 35 0 / 0 |")
        for chip_id in (0, 1):
            d = npu_id * 2 + chip_id
            dual_lines.append(f"| {chip_id} {d} | 0000:9D:00.0 | 0 0 / 0 3000 / 65536 |")
    dual_lines.extend(
        [
            "| NPU Chip | Process id | Process name | Process memory(MB) |",
            "| No running processes found in NPU 0 |",
        ]
    )
    dual_pcts = ac.verify_npu_smi_dies(
        "\n".join(dual_lines), dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT
    )
    assert set(dual_pcts) == set(range(16))

    tiny_over_smi = _fake_npu_smi({0: (800004, 10000000)})
    with pytest.raises(ac.IsolationError, match="HBM usage"):
        ac.verify_npu_smi_dies(tiny_over_smi, dies, max_hbm_pct=ac.ALARM_DIE_MAX_HBM_PCT)
