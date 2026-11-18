import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from .a3_runner import build_dry_run_bundle, compute_live_cache_digest, run_a3_formal
from .cli import main as cli_main
from .conftest import resolve_case01_bundle
from .contracts import (
    DEFAULT_KBENV_BIN,
    FINAL_POS,
    FORBIDDEN_HOSTS,
    GOLDEN_CASES,
    IsolationError,
    VerificationError,
    get_case_spec,
)
from .hygiene_gate import (
    allocate_isolation_env,
    parse_kbenv_output,
    select_a3_host,
    verify_isolation_env,
    verify_python_source_hygiene,
    verify_root_hygiene,
)
from .parity_gate import verify_e2e_run, verify_speculative_invariants


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_t3_01_f1_x_f2_case01_golden_and_8rank_digests_w16_w24(
    tmp_path: Path,
    case_root: Path,
) -> None:
    for width, min_w, port in ((16, 10, 29100), (24, 12, 29400)):
        out_dir = build_dry_run_bundle(
            case_id="case-01-coding",
            output_dir=tmp_path / f"w{width}",
            case_root=case_root,
            width=width,
            min_w_act=min_w,
            port_base=port,
        )
        res = verify_e2e_run(out_dir, case_root / "case-01-coding")
        assert res["output_sha256"] == get_case_spec("case-01-coding").output_sha256


def test_t3_02_f1_x_f3_case01_ab_ba_warm_head_and_sha_every_trial(
    tmp_path: Path,
    case_root: Path,
) -> None:
    out_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "abba_f1_f3",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        milestone="M5",
    )
    res = verify_e2e_run(
        out_dir,
        case_root / "case-01-coding",
        max_tpot_ms=3.0,
        require_warm_head=True,
        require_ab_ba=True,
        milestone="M5",
    )
    assert res["median_tpot_ms"] < 3.0


def test_t3_03_f2_x_f4_dynamic_w_act_split_and_chunk_with_8rank_parity(
    tmp_path: Path,
    case_root: Path,
) -> None:
    out_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "dyn_w",
        case_root=case_root,
        width=16,
        min_w_act=10,
    )
    report = _load_json(out_dir / "e2e_result.json")
    rank0 = _load_json(out_dir / "rank-0.json")
    verify_speculative_invariants(report["round_traces"], rank0["runtime_invariants"], width=16)
    res = verify_e2e_run(out_dir, case_root / "case-01-coding")
    assert res["case_id"] == "case-01-coding"


def test_t3_04_f3_x_f4_inflight2_pipelining_warmups_and_clean_pos9215_drain(
    tmp_path: Path,
    case_root: Path,
) -> None:
    out_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "inflight2",
        case_root=case_root,
        warmups=2,
        formals=5,
    )
    report = _load_json(out_dir / "e2e_result.json")
    rank0 = _load_json(out_dir / "rank-0.json")
    assert rank0["runtime_invariants"]["in_flight_depth"] == 2
    assert report["round_traces"][-1]["drain_q_len"] == 0
    assert report["round_traces"][-1]["final_pos"] == FINAL_POS
    verify_e2e_run(out_dir, case_root / "case-01-coding")


def test_t3_05_f3_x_f6_oos_cases_ab_ba_and_horizon_controller(
    tmp_path: Path,
    case_root: Path,
) -> None:
    for idx, case_id in enumerate(("case-02-review", "case-03-refactor", "case-04-math")):
        out_dir = build_dry_run_bundle(
            case_id=case_id,
            output_dir=tmp_path / f"oos_abba_{case_id}",
            case_root=case_root,
            warmups=3,
            formals=5,
            ab_ba=True,
            port_base=30000 + idx * 300,
        )
        res = verify_e2e_run(
            out_dir,
            case_root / case_id,
            require_ab_ba=True,
            milestone="M5",
        )
        assert res["median_tpot_ms"] < 3.0


def test_t3_06_f4_x_f6_distinct_oos_tau_regimes_without_sha_drift(
    tmp_path: Path,
    case_root: Path,
) -> None:
    dir_c2 = build_dry_run_bundle(
        case_id="case-02-review",
        output_dir=tmp_path / "c2",
        case_root=case_root,
        port_base=30100,
    )
    dir_c3 = build_dry_run_bundle(
        case_id="case-03-refactor",
        output_dir=tmp_path / "c3",
        case_root=case_root,
        port_base=30400,
    )
    r2 = _load_json(dir_c2 / "e2e_result.json")
    r3 = _load_json(dir_c3 / "e2e_result.json")
    assert r2["round_traces"][-1]["horizon_level"] != r3["round_traces"][-1]["horizon_level"]
    verify_e2e_run(dir_c2, case_root / "case-02-review")
    verify_e2e_run(dir_c3, case_root / "case-03-refactor")


def test_t3_07_f5_x_f7_preflight_aborts_before_launching_bad_runner(
    tmp_path: Path,
    case_root: Path,
) -> None:
    marker = tmp_path / "should_not_exist.txt"
    bad_runner = f"python3 -c \"from pathlib import Path; Path('{marker}').write_text('ran')\""
    with pytest.raises(IsolationError):
        run_a3_formal(
            case_id="case-01-coding",
            output_dir=tmp_path / "aborted",
            case_root=case_root,
            runner_cmd=bad_runner,
            target_host=FORBIDDEN_HOSTS[0],
        )
    assert not marker.exists()


def test_t3_08_f1_x_f7_concurrent_dual_runs_disjoint_shm_ports_and_hygiene(
    tmp_path: Path,
    case_root: Path,
    repo_root: Path,
) -> None:
    env_a = allocate_isolation_env("run_concurrent_a", port_base=28100, target_host="7.150.12.20")
    env_b = allocate_isolation_env("run_concurrent_b", port_base=28500, target_host="7.150.13.51")
    verify_isolation_env(env_a, peer_envs=[env_b], target_host="7.150.12.20")
    verify_isolation_env(env_b, peer_envs=[env_a], target_host="7.150.13.51")
    dir_a = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "run_a",
        case_root=case_root,
        run_id="run_concurrent_a",
        port_base=28100,
    )
    dir_b = build_dry_run_bundle(
        case_id="case-02-review",
        output_dir=tmp_path / "run_b",
        case_root=case_root,
        run_id="run_concurrent_b",
        port_base=28500,
    )
    verify_e2e_run(dir_a, case_root / "case-01-coding")
    verify_e2e_run(dir_b, case_root / "case-02-review")
    assert verify_root_hygiene(repo_root)["pass"] is True


def test_t4_01_case01_formal_acceptance_end_to_end(
    tmp_path: Path,
    case_root: Path,
    repo_root: Path,
) -> None:
    summary_path = tmp_path / "t4_01" / "summary.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.glm52.e2e_tp8_ep8.cli",
            "--mode",
            "dry-run",
            "--case",
            "case-01-coding",
            "--case-root",
            str(case_root),
            "--output",
            str(summary_path),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = _load_json(summary_path)
    assert payload["pass"] is True
    assert payload["cases"][0]["median_tpot_ms"] < 3.0
    assert (
        payload["cases"][0]["output_sha256"]
        == "2f751372fc82cf290c8e35d3e7cc3cad0296fde4d8d188fba102ac34f82c2962"
    )
    verify_code = cli_main(
        [
            "--mode",
            "verify-report",
            "--case",
            "case-01-coding",
            "--case-root",
            str(case_root),
            "--report-dir",
            str(summary_path.parent / "case-01-coding"),
        ]
    )
    assert verify_code == 0


def test_t4_02_case01_ab_ba_3plus5_regression_gate(
    tmp_path: Path,
    case_root: Path,
) -> None:
    summary_path = tmp_path / "t4_02" / "summary.json"
    rc = cli_main(
        [
            "--mode",
            "dry-run",
            "--case",
            "case-01-coding",
            "--case-root",
            str(case_root),
            "--warmups",
            "3",
            "--formals",
            "5",
            "--ab-ba",
            "--output",
            str(summary_path),
        ]
    )
    assert rc == 0
    report = _load_json(summary_path.parent / "case-01-coding" / "e2e_result.json")
    assert report["protocol"]["paired_order"] == "ab_ba"
    assert report["summary"]["ab_ba_delta_ms"] < 0.0


def test_t4_03_oos_coding_acceptance_cases_02_03_04(
    tmp_path: Path,
    case_root: Path,
) -> None:
    summary_path = tmp_path / "t4_03" / "summary.json"
    rc = cli_main(
        [
            "--mode",
            "dry-run",
            "--case",
            "all",
            "--case-root",
            str(case_root),
            "--output",
            str(summary_path),
        ]
    )
    assert rc == 0
    payload = _load_json(summary_path)
    by_case = {item["case_id"]: item for item in payload["cases"]}
    assert set(by_case) == set(GOLDEN_CASES)
    for case_id, spec in GOLDEN_CASES.items():
        assert by_case[case_id]["output_sha256"] == spec.output_sha256


def test_t4_04_sequential_multicase_cache_isolation(
    tmp_path: Path,
    case_root: Path,
) -> None:
    sequence = (
        "case-01-coding",
        "case-02-review",
        "case-03-refactor",
        "case-04-math",
        "case-01-coding",
    )
    verified_digests: list[dict[str, Any]] = []
    for step_idx, case_id in enumerate(sequence):
        step_dir = build_dry_run_bundle(
            case_id=case_id,
            output_dir=tmp_path / f"seq_{step_idx}_{case_id}",
            case_root=case_root,
            port_base=32000 + step_idx * 300,
        )
        res = verify_e2e_run(step_dir, case_root / case_id)
        rank0 = _load_json(step_dir / "rank-0.json")
        verified_digests.append(
            {
                "case_id": case_id,
                "output_sha256": res["output_sha256"],
                "main_state": rank0["main_state"],
                "draft_state": rank0["draft_state"],
            }
        )
    assert verified_digests[0]["output_sha256"] == verified_digests[-1]["output_sha256"]
    assert verified_digests[0]["main_state"] == verified_digests[-1]["main_state"]
    assert verified_digests[0]["draft_state"] == verified_digests[-1]["draft_state"]
    assert verified_digests[0]["main_state"] != verified_digests[1]["main_state"]


def test_t4_05_milestone_m2_plus_vllm_free_and_tpot_policy_gate(
    tmp_path: Path,
    case_root: Path,
) -> None:
    for idx, (ms, max_tpot) in enumerate(
        (("M1", 6.8), ("M2", 6.0), ("M3", 4.0), ("M4", 3.0), ("M5", 3.0))
    ):
        out_dir = build_dry_run_bundle(
            case_id="case-01-coding",
            output_dir=tmp_path / f"ms_{ms}",
            case_root=case_root,
            milestone=ms,
            port_base=33500 + idx * 300,
        )
        res = verify_e2e_run(
            out_dir,
            case_root / "case-01-coding",
            max_tpot_ms=max_tpot,
            milestone=ms,
        )
        report = _load_json(out_dir / "e2e_result.json")
        if ms == "M1":
            assert report["runtime_hygiene"]["loaded_modules_has_vllm"] is True
        else:
            assert report["runtime_hygiene"]["loaded_modules_has_vllm"] is False
        assert res["milestone"] == ms


def test_t4_06_e2e_hygiene_kbenv_and_r3_r4_r5_audit(
    tmp_path: Path,
    repo_root: Path,
    e2e_pkg_dir: Path,
    case_root: Path,
) -> None:
    assert verify_root_hygiene(repo_root)["pass"] is True
    verify_python_source_hygiene(sorted(e2e_pkg_dir.glob("*.py")), max_line_length=100)
    kbenv_proc = subprocess.run(
        [DEFAULT_KBENV_BIN, "--demo", "--once"],
        capture_output=True,
        text=True,
        check=True,
    )
    selected = select_a3_host(parse_kbenv_output(kbenv_proc.stdout), min_free_dies=8)
    assert selected.ip not in FORBIDDEN_HOSTS

    tampered_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "tampered_cli",
        case_root=case_root,
    )
    res_file = tampered_dir / "e2e_result.json"
    bad_report = _load_json(res_file)
    bad_report["summary"]["output_sha256"] = "0" * 64
    res_file.write_text(json.dumps(bad_report) + "\n", encoding="utf-8")
    rc = cli_main(
        [
            "--mode",
            "verify-report",
            "--case",
            "case-01-coding",
            "--case-root",
            str(case_root),
            "--report-dir",
            str(tampered_dir),
        ]
    )
    assert rc == 1
    with pytest.raises(VerificationError):
        verify_e2e_run(tampered_dir, case_root / "case-01-coding")


def test_t4_07_partial_node_free_die_slice_and_target_host_propagation(
    tmp_path: Path,
    case_root: Path,
    repo_root: Path,
) -> None:
    fake_kbenv = tmp_path / "fake_kbenv.py"
    fake_kbenv.write_text(
        "print('  7.156.98.198 A3 Ascend910_9392 ========·· 84% "
        "··#············· 15/16 84% 部分 否')\n",
        encoding="utf-8",
    )
    mock_runner = tmp_path / "mock_runner.py"
    mock_runner.write_text(
        "import argparse, json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from tests.glm52.e2e_tp8_ep8.a3_runner import build_dry_run_bundle\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--case-dir', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "p.add_argument('--warmups', type=int, required=True)\n"
        "p.add_argument('--trials', type=int, required=True)\n"
        "p.add_argument('--protocol', required=True)\n"
        "p.add_argument('--milestone', required=True)\n"
        "p.add_argument('--width', type=int, required=True)\n"
        "a, _ = p.parse_known_args()\n"
        "out = Path(a.output_dir)\n"
        "build_dry_run_bundle(\n"
        "    case_id=Path(a.case_dir).name,\n"
        "    output_dir=out,\n"
        "    case_root=Path(a.case_dir).parent,\n"
        "    warmups=a.warmups,\n"
        "    formals=a.trials,\n"
        "    ab_ba=(a.protocol == 'ab_ba'),\n"
        "    milestone=a.milestone,\n"
        "    width=a.width,\n"
        "    run_id=os.environ['TILERT_E2E_RUN_ID'],\n"
        "    port_base=int(os.environ['MASTER_PORT']),\n"
        ")\n"
        "(out / 'env_seen.json').write_text(json.dumps({\n"
        "    'TILERT_TARGET_HOST': os.environ.get('TILERT_TARGET_HOST'),\n"
        "    'ASCEND_RT_VISIBLE_DEVICES': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),\n"
        "    'MASTER_PORT': os.environ.get('MASTER_PORT'),\n"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "partial_run"
    res = run_a3_formal(
        case_id="case-01-coding",
        output_dir=out_dir,
        case_root=case_root,
        runner_cmd=f"{sys.executable} {mock_runner}",
        kbenv_bin=f"{sys.executable}",
        kbenv_args=[str(fake_kbenv)],
    )
    seen = _load_json(out_dir / "env_seen.json")
    assert seen["TILERT_TARGET_HOST"] == "7.156.98.198"
    assert seen["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15"
    assert res["host"] == "7.156.98.198"
    assert res["visible_devices"] == "8,9,10,11,12,13,14,15"


def test_t4_08_a3_live_requires_report_dir_or_runner_cmd(
    tmp_path: Path,
    case_root: Path,
) -> None:
    with pytest.raises(pytest.fail.Exception):
        resolve_case01_bundle(
            tmp_path=tmp_path,
            case_root=case_root,
            e2e_milestone="M5",
            is_a3_live=True,
            live_report_dir=None,
            live_runner_cmd=None,
        )
    existing = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "live_dir",
        case_root=case_root,
        port_base=34100,
    )
    resolved = resolve_case01_bundle(
        tmp_path=tmp_path,
        case_root=case_root,
        e2e_milestone="M5",
        is_a3_live=True,
        live_report_dir=existing,
        live_runner_cmd=None,
    )
    assert resolved == existing


def test_t4_09_a3_formal_ephemeral_ports_fresh_dir_timeout_and_live_digest(
    tmp_path: Path,
    case_root: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_runner = tmp_path / "env_runner.py"
    mock_runner.write_text(
        "import argparse, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from tests.glm52.e2e_tp8_ep8.a3_runner import build_dry_run_bundle\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--case-dir', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "a, _ = p.parse_known_args()\n"
        "out = Path(a.output_dir)\n"
        "build_dry_run_bundle(\n"
        "    case_id=Path(a.case_dir).name,\n"
        "    output_dir=out,\n"
        "    case_root=Path(a.case_dir).parent,\n"
        "    run_id=os.environ['TILERT_E2E_RUN_ID'],\n"
        "    port_base=int(os.environ['MASTER_PORT']),\n"
        ")\n",
        encoding="utf-8",
    )
    summary0 = tmp_path / "cli_a3_0" / "summary.json"
    rc0 = cli_main(
        [
            "--mode",
            "a3-formal",
            "--case",
            "case-01-coding",
            "--case-root",
            str(case_root),
            "--target-host",
            "7.150.12.20",
            "--runner-cmd",
            f"{sys.executable} {mock_runner}",
            "--output",
            str(summary0),
        ]
    )
    assert rc0 == 0
    rep0 = _load_json(summary0.parent / "case-01-coding" / "e2e_result.json")
    port0 = int(rep0["isolation"]["master_port"])

    summary1 = tmp_path / "cli_a3_1" / "summary.json"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy_sock:
        busy_sock.bind(("127.0.0.1", port0))
        rc1 = cli_main(
            [
                "--mode",
                "a3-formal",
                "--case",
                "case-01-coding",
                "--case-root",
                str(case_root),
                "--target-host",
                "7.150.12.20",
                "--runner-cmd",
                f"{sys.executable} {mock_runner}",
                "--output",
                str(summary1),
            ]
        )
    assert rc1 == 0
    rep1 = _load_json(summary1.parent / "case-01-coding" / "e2e_result.json")
    port1 = int(rep1["isolation"]["master_port"])
    assert abs(port0 - port1) >= 300

    stale_dir = tmp_path / "cli_a3_0" / "case-01-coding"
    with pytest.raises(VerificationError):
        run_a3_formal(
            case_id="case-01-coding",
            output_dir=stale_dir,
            case_root=case_root,
            runner_cmd="true",
            target_host="7.150.12.20",
        )

    sleeper = f"{sys.executable} -c 'import time; time.sleep(5)'"
    with pytest.raises(VerificationError, match="timed out"):
        run_a3_formal(
            case_id="case-01-coding",
            output_dir=tmp_path / "timeout_run",
            case_root=case_root,
            runner_cmd=sleeper,
            target_host="7.150.12.20",
            timeout_s=0.1,
        )

    class _FakeTensor:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self.dtype = "torch.bfloat16"
            self.shape = [FINAL_POS, 8]

        def __getitem__(self, _item: slice) -> "_FakeTensor":
            return self

        def contiguous(self) -> "_FakeTensor":
            return self

        def view(self, _dtype: object) -> "_FakeTensor":
            return self

        def cpu(self) -> "_FakeTensor":
            return self

        def numpy(self) -> SimpleNamespace:
            return SimpleNamespace(tobytes=lambda: self._payload)

    fake_torch = SimpleNamespace(uint8="uint8")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    digest = compute_live_cache_digest([{"latent": _FakeTensor(b"abc")}], FINAL_POS)
    assert digest["position"] == FINAL_POS
    assert len(digest["layers"][0]["latent"]["sha256"]) == 64


def test_t4_10_cli_rejects_low_warmups_formals_and_nan_tpot(
    tmp_path: Path,
    case_root: Path,
) -> None:
    for extra in (
        ["--warmups", "1"],
        ["--formals", "4"],
        ["--ab-ba", "--warmups", "2"],
        ["--max-tpot-ms", "nan"],
        ["--max-tpot-ms", "0"],
    ):
        rc = cli_main(
            [
                "--mode",
                "dry-run",
                "--case",
                "case-01-coding",
                "--case-root",
                str(case_root),
                "--output",
                str(tmp_path / "bad_cli" / "summary.json"),
                *extra,
            ]
        )
        assert rc == 1


def test_t4_11_n2_n5_runner_contract_match_fresh_dir_and_digest_validation(
    tmp_path: Path,
    case_root: Path,
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cheat_runner = tmp_path / "cheat_runner.py"
    cheat_runner.write_text(
        "import argparse, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(repo_root)!r})\n"
        "from tests.glm52.e2e_tp8_ep8.a3_runner import build_dry_run_bundle\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--case-dir', required=True)\n"
        "p.add_argument('--output-dir', required=True)\n"
        "a, _ = p.parse_known_args()\n"
        "build_dry_run_bundle(\n"
        "    case_id=Path(a.case_dir).name,\n"
        "    output_dir=Path(a.output_dir),\n"
        "    case_root=Path(a.case_dir).parent,\n"
        "    warmups=2,\n"
        "    formals=5,\n"
        "    milestone='M1',\n"
        "    width=16,\n"
        "    run_id=os.environ['TILERT_E2E_RUN_ID'],\n"
        "    port_base=int(os.environ['MASTER_PORT']),\n"
        ")\n",
        encoding="utf-8",
    )
    for idx, kwargs in enumerate(
        (
            {"warmups": 4, "formals": 5, "width": 16, "milestone": "M1"},
            {"warmups": 2, "formals": 7, "width": 16, "milestone": "M1"},
            {"warmups": 2, "formals": 5, "width": 12, "milestone": "M1"},
            {"warmups": 2, "formals": 5, "width": 16, "milestone": "M5"},
        )
    ):
        with pytest.raises(VerificationError):
            run_a3_formal(
                case_id="case-01-coding",
                output_dir=tmp_path / f"cheat_{idx}",
                case_root=case_root,
                runner_cmd=f"{sys.executable} {cheat_runner}",
                target_host="7.150.12.20",
                **kwargs,
            )

    m1_bundle = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "m1_bundle",
        case_root=case_root,
        milestone="M1",
        port_base=36500,
    )
    rc_ms = cli_main(
        [
            "--mode",
            "verify-report",
            "--case",
            "case-01-coding",
            "--case-root",
            str(case_root),
            "--report-dir",
            str(m1_bundle),
            "--milestone",
            "M5",
        ]
    )
    assert rc_ms == 1

    polluted = tmp_path / "polluted_out"
    polluted.mkdir()
    (polluted / "old_trace.log").write_text("stale\n", encoding="utf-8")
    with pytest.raises(VerificationError):
        run_a3_formal(
            case_id="case-01-coding",
            output_dir=polluted,
            case_root=case_root,
            runner_cmd="true",
            target_host="7.150.12.20",
        )

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    with pytest.raises(VerificationError):
        compute_live_cache_digest([{"latent": SimpleNamespace(shape=[FINAL_POS, 8])}], 0)
    with pytest.raises(VerificationError):
        compute_live_cache_digest([], FINAL_POS)
    with pytest.raises(VerificationError):
        compute_live_cache_digest([{"latent": SimpleNamespace(shape=[10, 512])}], FINAL_POS)


def test_t4_12_r5_cli_verify_report_flags_and_non_dict_json(
    tmp_path: Path,
    case_root: Path,
) -> None:
    abba_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "r5_abba",
        case_root=case_root,
        warmups=3,
        formals=5,
        ab_ba=True,
        port_base=40100,
    )
    assert (
        cli_main(
            [
                "--mode",
                "verify-report",
                "--case",
                "case-01-coding",
                "--case-root",
                str(case_root),
                "--report-dir",
                str(abba_dir),
                "--ab-ba",
            ]
        )
        == 0
    )

    dry_dir = build_dry_run_bundle(
        case_id="case-01-coding",
        output_dir=tmp_path / "r5_dry",
        case_root=case_root,
        port_base=40400,
    )
    for bad_flags in (
        ["--warmups", "0", "--formals", "0", "--width", "999"],
        ["--warmups", "3"],
        ["--formals", "7"],
        ["--width", "24"],
    ):
        assert (
            cli_main(
                [
                    "--mode",
                    "verify-report",
                    "--case",
                    "case-01-coding",
                    "--case-root",
                    str(case_root),
                    "--report-dir",
                    str(dry_dir),
                    *bad_flags,
                ]
            )
            == 1
        )

    bad_json_dir = tmp_path / "r5_list_json"
    bad_json_dir.mkdir()
    (bad_json_dir / "e2e_result.json").write_text("[]\n", encoding="utf-8")
    summary_out = tmp_path / "r5_err_summary.json"
    assert (
        cli_main(
            [
                "--mode",
                "verify-report",
                "--case",
                "case-01-coding",
                "--case-root",
                str(case_root),
                "--report-dir",
                str(bad_json_dir),
                "--output",
                str(summary_out),
            ]
        )
        == 1
    )
    assert _load_json(summary_out)["pass"] is False
