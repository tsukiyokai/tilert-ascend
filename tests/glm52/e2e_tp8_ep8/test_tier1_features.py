import json
from pathlib import Path
import subprocess
import sys

from typing import Any

import pytest

from .a3_runner import build_dry_run_bundle
from .contracts import (
    ALLOW_RELAXED_PARITY,
    DECODE_TOKENS,
    DEFAULT_KBENV_BIN,
    FINAL_POS,
    FORBIDDEN_HOSTS,
    GOLDEN_CASES,
    ISL,
    OSL,
    SCHEMA_VERSION,
    WORLD_SIZE,
    VerificationError,
    compact_tokens_sha256,
    default_tokens_sha256,
    get_case_spec,
)
from .hygiene_gate import (
    KbenvHost,
    allocate_isolation_env,
    parse_kbenv_output,
    select_a3_host,
    verify_isolation_env,
    verify_python_source_hygiene,
    verify_root_hygiene,
)
from .parity_gate import (
    ZeroLagHorizonRef,
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


def _load_ranks(bundle_dir: Path) -> list[dict[str, Any]]:
    return [_load_json(bundle_dir / f"rank-{rank}.json") for rank in range(WORLD_SIZE)]


def test_t1_f1_case01_prompt_len_and_compact_sha(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    prompt_data = verify_prompt_case(case_root / "case-01-coding", spec)
    assert len(prompt_data["prompt_token_ids"]) == ISL
    assert compact_tokens_sha256(prompt_data["prompt_token_ids"]) == spec.prompt_sha256


def test_t1_f1_case01_first_token_id(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(case_root / "case-01-coding" / "bench_result.json")
    assert bench["first_token_id"] == 13534
    assert bench["output_token_ids"][0] == spec.first_token_id


def test_t1_f1_case01_output_len_and_accepted_sum(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    formal_trials = [t for t in report["trials"] if not t["is_warmup"]]
    assert formal_trials
    for trial in formal_trials:
        assert len(trial["output_token_ids"]) == OSL
        assert sum(trial["accepted_per_round"]) == DECODE_TOKENS


def test_t1_f1_case01_output_default_json_sha(case_root: Path) -> None:
    spec = get_case_spec("case-01-coding")
    bench = _load_json(case_root / "case-01-coding" / "bench_result.json")
    tokens = bench["output_token_ids"]
    actual = verify_output_tokens(
        tokens,
        expected_sha256=spec.output_sha256,
        first_token_id=spec.first_token_id,
    )
    assert actual == "2f751372fc82cf290c8e35d3e7cc3cad0296fde4d8d188fba102ac34f82c2962"
    assert default_tokens_sha256(tokens) == spec.output_sha256


def test_t1_f1_schema_world_size_and_ep(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["world_size"] == WORLD_SIZE
    assert report["expert_parallel"] is True
    assert "torch" not in sys.modules


def test_t1_f2_all_8_ranks_present_and_complete(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)
    assert [row["rank"] for row in ranks] == list(range(WORLD_SIZE))
    verify_8rank_states(ranks, expected_spec=get_case_spec("case-01-coding"))


def test_t1_f2_main_state_pos_9215_and_78_layers_match(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)
    ref_main = ranks[0]["main_state"]
    assert ref_main["position"] == FINAL_POS
    assert len(ref_main["layers"]) == 78
    verify_8rank_states(ranks, main_ref=ref_main)


def test_t1_f2_draft_state_pos_9215_and_1_layer_match(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)
    ref_draft = ranks[0]["draft_state"]
    assert ref_draft["position"] == FINAL_POS
    assert len(ref_draft["layers"]) == 1
    verify_8rank_states(ranks, serial_draft_ref=ref_draft)


def test_t1_f2_serial_draft_match_true_all_ranks(case01_bundle: Path) -> None:
    ranks = _load_ranks(case01_bundle)
    assert all(row["serial_draft_match"] is True for row in ranks)
    assert len({tuple(row["accepted_per_round"]) for row in ranks}) == 1


def test_t1_f2_draft_head_distinct_from_lm_head(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    assert report["runtime_hygiene"]["draft_head_distinct_from_lm_head"] is True


def test_t1_f3_warmups_head_warmed_and_prefix_restored(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    proto = report["protocol"]
    assert proto["warmups"] >= 2
    assert proto["head_warmed"] is True
    assert proto["prefix_restored"] is True
    assert proto["synchronized_start"] is True


def test_t1_f3_profiler_and_sync_debug_off(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    proto = report["protocol"]
    assert proto["profiler_enabled"] is False
    assert proto["sync_debug_flags"] == []


def test_t1_f3_tpot_uses_max_rank_over_1023(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    for trial in report["trials"]:
        expected = max(trial["rank_dt_ms"]) / float(DECODE_TOKENS)
        assert abs(trial["max_tpot_ms"] - expected) < 1e-6


def test_t1_f3_formal_count_and_summary_stats_recomputed(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    metrics = verify_timing_and_protocol(report, milestone=report["milestone"])
    assert len(report["summary"]["formal_ms"]) >= 5
    assert metrics["min_tpot_ms"] <= metrics["median_tpot_ms"] <= metrics["p90_tpot_ms"]


def test_t1_f3_ab_ba_interleaved_and_tpot_target(
    tmp_path: Path,
    case_root: Path,
) -> None:
    bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "abba",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        milestone="M5",
    )
    report = _load_json(bundle / "e2e_result.json")
    metrics = verify_timing_and_protocol(
        report,
        max_tpot_ms=3.0,
        require_warm_head=True,
        require_ab_ba=True,
        milestone="M5",
    )
    assert metrics["median_tpot_ms"] < 3.0


def test_t1_f4_isl_8192_decode_1023_final_pos_9215(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    traces = report["round_traces"]
    assert traces[-1]["done"] is True
    assert traces[-1]["final_pos"] == ISL + DECODE_TOKENS == FINAL_POS


def test_t1_f4_select_optimal_w_act_w16_w24_and_mask() -> None:
    lps_w16 = [0.0 if i == 0 else -0.45 * float(i) for i in range(16)]
    w16 = select_optimal_w_act(lps_w16, width=16, min_w_act=10, t_base_ms=46.5, c_w_ms=0.68)
    assert w16 in (10, 12, 14, 16)
    lps_w24 = [0.0 if i == 0 else -0.15 * float(i) for i in range(24)]
    w24 = select_optimal_w_act(lps_w24, width=24, min_w_act=12, t_base_ms=48.0, c_w_ms=0.78)
    assert w24 in (12, 14, 16, 18, 20, 22, 24)
    assert select_optimal_w_act(lps_w16, width=16, min_w_act=0) == 16
    assert select_optimal_w_act(lps_w16, width=16, min_w_act=16) == 16


def test_t1_f4_collective_split_max_rows_le_16(case01_bundle: Path) -> None:
    rank0 = _load_json(case01_bundle / "rank-0.json")
    inv = rank0["runtime_invariants"]
    assert 1 <= inv["collective_split_max_rows"] <= 16


def test_t1_f4_l78_gemm_chunk_max_rows_le_8(case01_bundle: Path) -> None:
    rank0 = _load_json(case01_bundle / "rank-0.json")
    inv = rank0["runtime_invariants"]
    assert 1 <= inv["l78_gemm_chunk_max_rows"] <= 8


def test_t1_f4_horizon_controller_kv_commit_and_inflight_drain(case01_bundle: Path) -> None:
    ctrl = ZeroLagHorizonRef()
    assert ctrl.update(2) == 0
    assert ctrl.update(9) == 2
    assert ctrl.update(7) == 2
    report = _load_json(case01_bundle / "e2e_result.json")
    rank0 = _load_json(case01_bundle / "rank-0.json")
    verify_speculative_invariants(report["round_traces"], rank0["runtime_invariants"], width=16)


def test_t1_f5_strict_guard_passes_valid_bundle(case01_bundle: Path, case_root: Path) -> None:
    summary = verify_e2e_run(case01_bundle, case_root / "case-01-coding", milestone="M5")
    assert summary["case_id"] == "case-01-coding"


def test_t1_f5_allow_relaxed_parity_cannot_be_enabled(case01_bundle: Path, case_root: Path) -> None:
    assert ALLOW_RELAXED_PARITY is False
    with pytest.raises(VerificationError):
        verify_e2e_run(
            case01_bundle,
            case_root / "case-01-coding",
            allow_relaxed_parity=True,
        )


def test_t1_f5_1023_divisor_enforced_in_e2e(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    assert report["protocol"]["tpot_formula"] == "max(8-rank dt_ms) / 1023.0"


def test_t1_f5_rejects_unknown_schema_version(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    report["schema_version"] = "glm52.e2e.v0"
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(report)


def test_t1_f5_rejects_missing_runtime_hygiene(case01_bundle: Path) -> None:
    report = _load_json(case01_bundle / "e2e_result.json")
    report.pop("runtime_hygiene")
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(report)


def test_t1_f6_cases_index_matches_all_4_prompts(case_root: Path) -> None:
    index_rows = json.loads((case_root / "cases-index.json").read_text(encoding="utf-8"))
    by_id = {row["case_id"]: row for row in index_rows}
    assert set(by_id) == set(GOLDEN_CASES)
    for case_id, spec in GOLDEN_CASES.items():
        assert by_id[case_id]["prompt_sha256"] == spec.prompt_sha256
        assert by_id[case_id]["seq_len"] == spec.seq_len


def test_t1_f6_case02_review_golden_parity(case_root: Path) -> None:
    spec = get_case_spec("case-02-review")
    verify_prompt_case(case_root / "case-02-review", spec)
    bench = _load_json(case_root / "case-02-review" / "bench_result.json")
    assert verify_output_tokens(
        bench["output_token_ids"],
        expected_sha256=spec.output_sha256,
        first_token_id=334,
    ) == spec.output_sha256


def test_t1_f6_case03_refactor_golden_parity(case_root: Path) -> None:
    spec = get_case_spec("case-03-refactor")
    verify_prompt_case(case_root / "case-03-refactor", spec)
    bench = _load_json(case_root / "case-03-refactor" / "bench_result.json")
    assert verify_output_tokens(
        bench["output_token_ids"],
        expected_sha256=spec.output_sha256,
        first_token_id=1499,
    ) == spec.output_sha256


def test_t1_f6_case04_math_golden_parity(case_root: Path) -> None:
    spec = get_case_spec("case-04-math")
    verify_prompt_case(case_root / "case-04-math", spec)
    bench = _load_json(case_root / "case-04-math" / "bench_result.json")
    assert verify_output_tokens(
        bench["output_token_ids"],
        expected_sha256=spec.output_sha256,
        first_token_id=8419,
    ) == spec.output_sha256


def test_t1_f6_all_oos_cases_verify_e2e_run(tmp_path: Path, case_root: Path) -> None:
    for idx, case_id in enumerate(("case-02-review", "case-03-refactor", "case-04-math")):
        out_dir = build_dry_run_bundle(
            case_id=case_id,
            output_dir=tmp_path / case_id,
            case_root=case_root,
            port_base=31000 + idx * 300,
        )
        res = verify_e2e_run(out_dir, case_root / case_id)
        assert res["case_id"] == case_id


def test_t1_f7_project_py_check_and_root_layout(repo_root: Path) -> None:
    result = verify_root_hygiene(repo_root)
    assert result["pass"] is True


def test_t1_f7_e2e_package_zero_comments_and_docstrings(e2e_pkg_dir: Path) -> None:
    py_files = sorted(e2e_pkg_dir.glob("*.py"))
    assert len(py_files) == 10
    verify_python_source_hygiene(py_files, max_line_length=100)


def test_t1_f7_kbenv_parser_selects_eligible_a3_host() -> None:
    proc = subprocess.run(
        [DEFAULT_KBENV_BIN, "--demo", "--once"],
        capture_output=True,
        text=True,
        check=True,
    )
    hosts = parse_kbenv_output(proc.stdout)
    selected = select_a3_host(hosts, min_free_dies=8)
    assert selected.ip == "7.208.149.157"
    assert selected.chip == "Ascend910_9392"
    assert selected.free_dies == 16


def test_t1_f7_forbidden_hosts_excluded_even_if_a3_idle() -> None:
    hosts = [
        KbenvHost(
            ip=FORBIDDEN_HOSTS[0],
            generation="A3",
            chip="Ascend910_9392",
            free_dies=16,
            total_dies=16,
            ram_pct=85,
            state="空闲",
            linger="长期",
            free_die_indices=tuple(range(16)),
            die_map="·" * 16,
        ),
        KbenvHost(
            ip=FORBIDDEN_HOSTS[1],
            generation="A3",
            chip="Ascend910_9392",
            free_dies=16,
            total_dies=16,
            ram_pct=85,
            state="空闲",
            linger="长期",
            free_die_indices=tuple(range(16)),
            die_map="·" * 16,
        ),
        KbenvHost(
            ip="7.150.12.20",
            generation="A3",
            chip="Ascend910_9392",
            free_dies=8,
            total_dies=16,
            ram_pct=80,
            state="部分",
            linger="长期",
            free_die_indices=tuple(range(8)),
            die_map="·" * 8 + "#" * 8,
        ),
    ]
    selected = select_a3_host(hosts, min_free_dies=8)
    assert selected.ip == "7.150.12.20"


def test_t1_f7_isolation_allocator_unique_shm_ports_atb() -> None:
    env_a = allocate_isolation_env("run_alpha", port_base=29100, target_host="7.150.12.20")
    env_b = allocate_isolation_env("run_beta", port_base=29400, target_host="7.150.13.51")
    verify_isolation_env(env_a, peer_envs=[env_b], target_host="7.150.12.20")
    verify_isolation_env(env_b, peer_envs=[env_a], target_host="7.150.13.51")
