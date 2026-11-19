import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.glm52 import a3_container as ac
from tests.glm52.e2e_tp8_ep8.contracts import IsolationError, VerificationError
from tests.glm52.e2e_tp8_ep8.hygiene_gate import verify_python_source_hygiene
from tests.glm52.e2e_tp8_ep8.parity_gate import verify_timing_and_protocol
from tests.glm52.warm_worker import (
    _manifest_has_kernel_prof,
    _pid_alive,
    aggregate_epoch_ranks,
    atomic_write_json,
    boot_daemon,
    classify_reload_tier,
    compute_boot_fingerprint,
    compute_hot_fingerprints,
    reset_all_graphs,
    stage_versioned_so,
    stop_daemon,
    submit_epoch,
    validate_job_dir,
)


def test_atomic_cmd_and_epoch_done_roundtrip(tmp_path: Path) -> None:
    job_dir = tmp_path / "control"
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "\n".join(
            [
                "import json, os, sys, time",
                "from pathlib import Path",
                "job_dir = Path(sys.argv[1])",
                "ack_dir = job_dir / 'worker_ack'",
                "ack_dir.mkdir(parents=True, exist_ok=True)",
                "(job_dir / 'daemon_ready.json').write_text("
                "json.dumps({'world_size': 8, 'boot_sha': 'b0', 'epoch': 0}))",
                "cmd_p = job_dir / 'worker_cmd.json'",
                "epoch = 1",
                "while True:",
                "    if cmd_p.is_file():",
                "        cmd = json.loads(cmd_p.read_text())",
                "        if cmd.get('epoch') == epoch:",
                "            if cmd.get('action') == 'exit':",
                "                (job_dir / f'epoch-{epoch}-done.json').write_text("
                "json.dumps({'epoch': epoch, 'ok': True}))",
                "                sys.exit(0)",
                "            for r in range(8):",
                "                (job_dir / f'epoch-{epoch}-rank-{r}.json').write_text(",
                "                    json.dumps({'epoch': epoch, 'rank': r, 'status': 'OK',",
                "                    'elapsed_seconds': 0.01, 'output_sha256': 'x'}))",
                "            (job_dir / f'epoch-{epoch}-done.json').write_text(",
                "                json.dumps({'epoch': epoch, 'ok': True, 'failed_ranks': []}))",
                "            epoch += 1",
                "    time.sleep(0.01)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    boot = boot_daemon(
        job_dir,
        model_dir="/work/model-raw",
        case_dir="c",
        mtp_prefix="m",
        ckpt="k",
        whole_layer_base="w",
        boot_timeout_s=5.0,
        worker_cmd_override=[sys.executable, str(script), str(job_dir)],
    )
    assert boot["world_size"] == 8
    r1 = submit_epoch(job_dir, action="eval", cmd_args={"k_sfa": 1024}, epoch_timeout_s=5.0)
    r2 = submit_epoch(job_dir, action="eval", cmd_args={"k_sfa": 1024}, epoch_timeout_s=5.0)
    assert r1["epoch"] == 1 and r1["ok"] is True
    assert r2["epoch"] == 2 and r2["ok"] is True
    stopped = stop_daemon(job_dir, timeout_s=5.0)
    assert stopped["stopped"] is True


def test_rank_error_aggregation(tmp_path: Path) -> None:
    job_dir = tmp_path / "ctrl"
    job_dir.mkdir()
    for r in range(8):
        payload = {"epoch": 1, "rank": r, "status": "OK", "elapsed_seconds": 0.02}
        if r == 3:
            payload["status"] = "FAIL"
            payload["error"] = "RuntimeError: rank 3 boom"
        atomic_write_json(job_dir / f"epoch-1-rank-{r}.json", payload)
    agg = aggregate_epoch_ranks(job_dir, 1, world_size=8)
    assert agg["ok"] is False
    assert agg["failed_ranks"] == [3]
    assert "rank 3 boom" in agg["errors"]["3"]


def test_dead_daemon_fails_fast(tmp_path: Path) -> None:
    job_dir = tmp_path / "dead"
    job_dir.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    atomic_write_json(job_dir / "daemon.pid", {"pid": proc.pid, "pgid": proc.pid})
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="dead"):
        submit_epoch(job_dir, epoch_timeout_s=30.0)
    assert time.monotonic() - t0 < 2.0


def test_timeout_kills_process_group(tmp_path: Path) -> None:
    job_dir = tmp_path / "hang"
    job_dir.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    atomic_write_json(job_dir / "daemon.pid", {"pid": proc.pid, "pgid": pgid})
    with pytest.raises(TimeoutError):
        submit_epoch(job_dir, epoch_timeout_s=0.3)
    proc.wait(timeout=3.0)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


def test_shm_blocklist_rejected() -> None:
    for bad in (
        "/dev/shm",
        "/dev/shm/",
        "/dev/shm/agy_tp8",
        "/dev/shm/agy_tp8/control",
        "/dev/shm/run_1/../../agy_tp8",
        "//dev/shm",
        "//dev/shm/agy_tp8",
        "//dev/shm/agy_tp8/control",
    ):
        with pytest.raises(IsolationError):
            validate_job_dir(bad)
    ok = validate_job_dir("/dev/shm/run_ok_123/control", run_id="run_ok_123")
    assert ok == Path("/dev/shm/run_ok_123/control")
    ok_dbl = validate_job_dir("//dev/shm/run_ok_123/control", run_id="run_ok_123")
    assert ok_dbl == Path("/dev/shm/run_ok_123/control")


def test_versioned_so_and_reload_tier(tmp_path: Path) -> None:
    so = tmp_path / "chain_K110419.so"
    so.write_bytes(b"v1-binary")
    v1 = stage_versioned_so(so)
    so.write_bytes(b"v2-binary")
    v2 = stage_versioned_so(so)
    assert v1 != v2
    assert v1.read_bytes() == b"v1-binary"
    assert v2.read_bytes() == b"v2-binary"

    base = {
        "boot_sha": "b1",
        "chain_sha": "c1",
        "dspine_py_sha": "p1",
        "dspine_ops_sha": "o1",
        "l78_klib_sha": "l1",
        "ckpt_sha": "k1",
    }
    assert classify_reload_tier(base, dict(base)) == "none"
    assert classify_reload_tier(base, {**base, "l78_klib_sha": "l2"}) == "dspine"
    assert classify_reload_tier(base, {**base, "chain_sha": "c2"}) == "chain_and_dspine"
    assert classify_reload_tier(base, {**base, "boot_sha": "b2"}) == "reboot"


def test_boot_and_hot_fingerprints_and_graph_reset(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    model_dir = repo / "tilert/models/glm_5_2_ascend"
    dspine_dir = model_dir / "dspine"
    dspine_dir.mkdir(parents=True)
    for name in ("end2end.py", "abi.py"):
        (model_dir / name).write_text("x = 1\n", encoding="utf-8")
    (dspine_dir / "tp_drafter.py").write_text("y = 1\n", encoding="utf-8")
    bdir = tmp_path / "builds"
    (bdir / "dspine_ops").mkdir(parents=True)
    (bdir / "dspine_l78").mkdir(parents=True)
    (bdir / "chain.so").write_bytes(b"chain1")
    (bdir / "dspine_ops/tilert_dspine_ops.so").write_bytes(b"ops1")
    (bdir / "dspine_l78/chain_K110419.so").write_bytes(b"k19")
    (bdir / "chain.manifest.json").write_text(json.dumps({"flags": ["-DF=1"]}), encoding="utf-8")
    wl_base = tmp_path / "whole_layer_base.json"
    wl_base.write_text('{"indexed": "a"}\n', encoding="utf-8")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"ckpt1")

    boot1 = compute_boot_fingerprint(wl_base, "/work/model-raw", bdir, repo)
    (dspine_dir / "tp_drafter.py").write_text("y = 2\n", encoding="utf-8")
    boot2 = compute_boot_fingerprint(wl_base, "/work/model-raw", bdir, repo)
    assert boot1 == boot2
    (model_dir / "end2end.py").write_text("x = 2\n", encoding="utf-8")
    boot3 = compute_boot_fingerprint(wl_base, "/work/model-raw", bdir, repo)
    assert boot1 != boot3

    hot = compute_hot_fingerprints(bdir, ckpt, repo, k_sfa=1024)
    assert len(hot["l78_klib_sha"]) == 64

    calls: list[str] = []

    class _Graph:
        def reset(self) -> None:
            calls.append("reset")

    class _Win:
        handle = 1

        def close(self) -> None:
            calls.append("win_close")

    layer = SimpleNamespace(entries={"a": 1}, reserves={"b": 2})
    target = SimpleNamespace(
        graph=_Graph(),
        _head_graphs={16: (_Graph(), None)},
        _rope_graphs={16: SimpleNamespace(graph=_Graph())},
        _native_attention_workspace_pool=SimpleNamespace(
            entries={"x": 1},
            capture_reserves={"y": 2},
        ),
    )
    draft = SimpleNamespace(_l78_win=_Win())
    mgr = SimpleNamespace(
        chains={"c": 1},
        chain_reserves={16: 1},
        pending_steps=[1],
        chain_active=True,
        capture_width=16,
        capture_seen={"l"},
        capture_owner=object(),
        capture_stream=object(),
        slot_refreshed={1},
        arena_pool={"p": 1},
        layers={"l0": layer},
    )
    reset_all_graphs(target, draft, mgr)
    assert calls == ["win_close", "reset", "reset", "reset"]
    assert target.graph is None
    assert draft._l78_win is None
    assert not mgr.chains and not layer.entries and not mgr.arena_pool


def test_boot_eval_stop_boot_eval_same_job_dir(tmp_path: Path) -> None:
    job_dir = tmp_path / "control"
    script = tmp_path / "fake_worker_reboot.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "from pathlib import Path",
                "job_dir = Path(sys.argv[1])",
                "ack_dir = job_dir / 'worker_ack'",
                "ack_dir.mkdir(parents=True, exist_ok=True)",
                "(job_dir / 'daemon_ready.json').write_text("
                "json.dumps({'world_size': 8, 'boot_sha': 'b0', 'epoch': 0}))",
                "cmd_p = job_dir / 'worker_cmd.json'",
                "epoch = 1",
                "while True:",
                "    if cmd_p.is_file():",
                "        cmd = json.loads(cmd_p.read_text())",
                "        if cmd.get('epoch') == epoch:",
                "            if cmd.get('action') == 'exit':",
                "                (job_dir / f'epoch-{epoch}-done.json').write_text("
                "json.dumps({'epoch': epoch, 'ok': True}))",
                "                sys.exit(0)",
                "            for r in range(8):",
                "                (job_dir / f'epoch-{epoch}-rank-{r}.json').write_text(",
                "                    json.dumps({'epoch': epoch, 'rank': r, 'status': 'OK',",
                "                    'elapsed_seconds': 0.01, 'output_sha256': 'x'}))",
                "            (job_dir / f'epoch-{epoch}-done.json').write_text(",
                "                json.dumps({'epoch': epoch, 'ok': True, 'failed_ranks': []}))",
                "            epoch += 1",
                "    time.sleep(0.01)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cmd = [sys.executable, str(script), str(job_dir)]
    boot_daemon(
        job_dir,
        model_dir="/work/model-raw",
        case_dir="c",
        mtp_prefix="m",
        ckpt="k",
        whole_layer_base="w",
        boot_timeout_s=5.0,
        worker_cmd_override=cmd,
    )
    r1 = submit_epoch(job_dir, action="eval", cmd_args={"k_sfa": 1024}, epoch_timeout_s=5.0)
    assert r1["epoch"] == 1 and r1["ok"] is True
    assert stop_daemon(job_dir, timeout_s=5.0)["stopped"] is True
    (job_dir / "worker_ack" / "stale_ack").write_text("stale\n", encoding="utf-8")

    boot_daemon(
        job_dir,
        model_dir="/work/model-raw",
        case_dir="c",
        mtp_prefix="m",
        ckpt="k",
        whole_layer_base="w",
        boot_timeout_s=5.0,
        worker_cmd_override=cmd,
    )
    assert not (job_dir / "worker_ack" / "stale_ack").exists()
    r2 = submit_epoch(job_dir, action="eval", cmd_args={"k_sfa": 1024}, epoch_timeout_s=5.0)
    assert r2["epoch"] == 1 and r2["ok"] is True
    assert stop_daemon(job_dir, timeout_s=5.0)["stopped"] is True


def test_dspine_unit_digest_and_skip_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    dspine_dir = repo / "tilert/models/glm_5_2_ascend/dspine"
    tests_dir = repo / "tests/glm52"
    dspine_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (dspine_dir / "tp_drafter.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tilert/models/glm_5_2_ascend/abi.py").write_text("a = 1\n", encoding="utf-8")
    (tests_dir / "test_dspine_unit.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    stamp_path = tmp_path / "checks/dspine_unit_stamp.json"

    exec_calls: list[str] = []
    next_rc = 0

    def fake_exec(host: str, run_id: str, inner_cmd: str, **_kw: object) -> int:
        exec_calls.append(f"{host}:{run_id}:{inner_cmd}")
        return next_rc

    monkeypatch.setattr(ac, "exec_in_container", fake_exec)
    d0 = ac.dspine_unit_digest(repo)
    res1 = ac.run_dspine_check(
        host="7.150.11.91",
        run_id="run_test_m21",
        stamp_path=stamp_path,
        repo_root=repo,
    )
    assert res1["status"] == "passed"
    assert res1["digest"] == d0
    assert len(exec_calls) == 1
    assert ac.read_dspine_unit_stamp(stamp_path)["digest"] == d0

    res2 = ac.run_dspine_check(
        host="7.150.11.91",
        run_id="run_test_m21",
        stamp_path=stamp_path,
        repo_root=repo,
    )
    assert res2["status"] == "skipped"
    assert len(exec_calls) == 1

    res3 = ac.run_dspine_check(
        host="7.150.11.91",
        run_id="run_test_m21",
        force=True,
        stamp_path=stamp_path,
        repo_root=repo,
    )
    assert res3["status"] == "passed"
    assert len(exec_calls) == 2

    (dspine_dir / "tp_drafter.py").write_text("x = 2\n", encoding="utf-8")
    next_rc = 1
    res4 = ac.run_dspine_check(
        host="7.150.11.91",
        run_id="run_test_m21",
        stamp_path=stamp_path,
        repo_root=repo,
    )
    assert res4["status"] == "failed"
    assert res4["returncode"] == 1
    assert len(exec_calls) == 3
    assert ac.read_dspine_unit_stamp(stamp_path)["digest"] == d0


def test_a3_container_sync_source_prestage_probe_and_du_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / "tilert/models/glm_5_2_ascend").mkdir(parents=True)
    (repo / "tests/glm52").mkdir(parents=True)
    (repo / "tilert/models/glm_5_2_ascend/end2end.py").write_text("e = 1\n", encoding="utf-8")
    (repo / "tests/glm52/warm_worker.py").write_text("ww = 1\n", encoding="utf-8")

    ssh_calls: list[tuple[str, str, bytes | None]] = []

    def fake_remote_ssh(
        host: str,
        remote_cmd: str,
        *,
        input_bytes: bytes | None = None,
        check: bool = True,
    ) -> SimpleNamespace:
        ssh_calls.append((host, remote_cmd, input_bytes))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(ac, "remote_ssh", fake_remote_ssh)
    out = ac.sync_source("7.150.11.91", "run_test_m21", repo_root=repo)
    assert out["paths"] == ["tilert", "tests/glm52"]
    assert len(ssh_calls) == 1
    assert "/dev/shm/run_test_m21/source" in ssh_calls[0][1]
    with tarfile.open(fileobj=io.BytesIO(ssh_calls[0][2]), mode="r:gz") as tf:
        names = set(tf.getnames())
    assert "tilert/models/glm_5_2_ascend/end2end.py" in names
    assert "tests/glm52/warm_worker.py" in names

    auto_cmd = ac.build_docker_exec_cmd("run_test_m21", "echo hi")[5]
    assert "test -e /prestage/cal_l78_v3_hybrid_best.pt ||" in auto_cmd
    skip_cmd = ac.build_docker_exec_cmd("run_test_m21", "echo hi", prestage_ckpt=True)[5]
    assert ac.CONTAINER_SFS_CKPT_PROBE not in skip_cmd
    assert "timeout 5 test -e /prestage/cal_l78_v3_hybrid_best.pt" in skip_cmd
    assert ac.du_timeout_for_item("model-raw") == 60
    assert ac.du_timeout_for_item("ckpt") == 15
    up_tar_cmd = ac.build_up_source_tar_cmd(repo)
    assert "--exclude=docs/evidence" in up_tar_cmd
    assert "--exclude=docs/assets" in up_tar_cmd


def test_formal_e2e_package_intact() -> None:
    e2e_dir = REPO_ROOT / "tests/glm52/e2e_tp8_ep8"
    assert len(list(e2e_dir.glob("*.py"))) == 10
    verify_python_source_hygiene(
        [
            REPO_ROOT / "tests/glm52/a3_container.py",
            REPO_ROOT / "tests/glm52/warm_worker.py",
            REPO_ROOT / "tests/glm52/run_m1_case01.py",
            REPO_ROOT / "tests/glm52/run_m2_case01.py",
            Path(__file__),
        ],
        max_line_length=100,
    )
    bad_report = {
        "schema_version": "glm52.e2e.v1",
        "milestone": "M2",
        "world_size": 8,
        "expert_parallel": True,
        "protocol": {"warmups": 0, "formals": 1},
        "runtime_hygiene": {"loaded_modules_has_vllm": False},
    }
    with pytest.raises(VerificationError):
        verify_timing_and_protocol(bad_report, milestone="M2")


def test_stop_daemon_waits_for_delayed_exit_before_boot(tmp_path: Path) -> None:
    job_dir = tmp_path / "ctrl_delay"
    lock_file = tmp_path / "resource.lock"
    script = tmp_path / "delayed_worker.py"
    script.write_text(
        "\n".join(
            [
                "import fcntl, json, sys, time",
                "from pathlib import Path",
                "job_dir, lock_path = Path(sys.argv[1]), Path(sys.argv[2])",
                "fh = lock_path.open('a+', encoding='utf-8')",
                "try:",
                "    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)",
                "except BlockingIOError:",
                "    sys.exit(9)",
                "(job_dir / 'daemon_ready.json').write_text("
                "json.dumps({'world_size': 8, 'boot_sha': 'b0', 'epoch': 0}))",
                "cmd_p = job_dir / 'worker_cmd.json'",
                "epoch = 1",
                "while True:",
                "    if cmd_p.is_file():",
                "        cmd = json.loads(cmd_p.read_text())",
                "        if cmd.get('epoch') == epoch and cmd.get('action') == 'exit':",
                "            (job_dir / f'epoch-{epoch}-done.json').write_text(",
                "                json.dumps({'epoch': epoch, 'ok': True}))",
                "            time.sleep(0.35)",
                "            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)",
                "            fh.close()",
                "            sys.exit(0)",
                "    time.sleep(0.01)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cmd = [sys.executable, str(script), str(job_dir), str(lock_file)]
    b1 = boot_daemon(
        job_dir,
        model_dir="/work/model-raw",
        case_dir="c",
        mtp_prefix="m",
        ckpt="k",
        whole_layer_base="w",
        boot_timeout_s=5.0,
        worker_cmd_override=cmd,
    )
    old_pid = int(b1["pid"])
    stop_res = stop_daemon(job_dir, timeout_s=5.0)
    assert stop_res["stopped"] is True
    assert not _pid_alive(old_pid)
    b2 = boot_daemon(
        job_dir,
        model_dir="/work/model-raw",
        case_dir="c",
        mtp_prefix="m",
        ckpt="k",
        whole_layer_base="w",
        boot_timeout_s=5.0,
        worker_cmd_override=cmd,
    )
    assert int(b2["pid"]) != old_pid
    assert stop_daemon(job_dir, timeout_s=5.0)["stopped"] is True


def test_stop_daemon_kills_hung_worker_after_exit_ack(tmp_path: Path) -> None:
    job_dir = tmp_path / "ctrl_hung"
    script = tmp_path / "hung_exit_worker.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys, time",
                "from pathlib import Path",
                "job_dir = Path(sys.argv[1])",
                "(job_dir / 'daemon_ready.json').write_text("
                "json.dumps({'world_size': 8, 'boot_sha': 'b0', 'epoch': 0}))",
                "cmd_p = job_dir / 'worker_cmd.json'",
                "while True:",
                "    if cmd_p.is_file():",
                "        cmd = json.loads(cmd_p.read_text())",
                "        if cmd.get('epoch') == 1 and cmd.get('action') == 'exit':",
                "            (job_dir / 'epoch-1-done.json').write_text("
                "json.dumps({'epoch': 1, 'ok': True}))",
                "            time.sleep(60)",
                "    time.sleep(0.01)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    boot = boot_daemon(
        job_dir,
        model_dir="/work/model-raw",
        case_dir="c",
        mtp_prefix="m",
        ckpt="k",
        whole_layer_base="w",
        boot_timeout_s=5.0,
        worker_cmd_override=[sys.executable, str(script), str(job_dir)],
    )
    pid = int(boot["pid"])
    stop_res = stop_daemon(job_dir, timeout_s=0.4)
    assert stop_res["stopped"] is True
    assert not _pid_alive(pid)


def test_warm_worker_preserves_kernel_prof_manifest(tmp_path: Path) -> None:
    bdir = tmp_path / "prof_build"
    bdir.mkdir()
    assert _manifest_has_kernel_prof(None) is False
    assert _manifest_has_kernel_prof(bdir) is False
    (bdir / "chain.manifest.json").write_text(
        json.dumps({"enable_kernel_prof": True, "flags": []}),
        encoding="utf-8",
    )
    assert _manifest_has_kernel_prof(bdir) is True
    (bdir / "chain.manifest.json").write_text(
        json.dumps({"enable_kernel_prof": False, "flags": ["-DENABLE_KERNEL_PROF=1"]}),
        encoding="utf-8",
    )
    assert _manifest_has_kernel_prof(bdir) is True
    (bdir / "chain.manifest.json").write_text(
        json.dumps({"enable_kernel_prof": False, "flags": ["-DF=1"]}),
        encoding="utf-8",
    )
    assert _manifest_has_kernel_prof(bdir) is False


def test_a3_container_exclude_forbidden_and_ssh_key_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ac.select_fleet_slots([], exclude_hosts=["10.83.185.129"]) == []
    hosts = ac.parse_kbenv_output(
        "  7.150.11.91 A3 Ascend910_9392 ====······ 41% "
        "················ 16/16 41% 空闲 否\n"
    )
    assert ac.select_fleet_slots(hosts, exclude_hosts=["10.83.185.129"]) == [
        ("7.150.11.91", tuple(range(8))),
        ("7.150.11.91", tuple(range(8, 16))),
    ]
    assert ac.select_fleet_slots(hosts, exclude_hosts=["10.83.185.129", "7.150.11.91"]) == []

    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    key_path = ssh_dir / "id_ed25519_a3_a5_codex"
    key_path.write_text("dummy-key\n", encoding="utf-8")
    monkeypatch.setattr(ac.Path, "home", lambda: tmp_path)

    seen_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: object) -> SimpleNamespace:
        seen_cmds.append(list(cmd))
        return SimpleNamespace(returncode=0 if str(key_path) in cmd else 255)

    monkeypatch.setattr(ac.subprocess, "run", fake_run)
    monkeypatch.delenv("TILERT_SSH_FORCE_PASSWORD", raising=False)
    ssh_cmd, env = ac._ssh_base_cmd("7.150.10.158")
    assert env is None
    assert str(key_path) in ssh_cmd
    assert "IdentitiesOnly=yes" in ssh_cmd
    assert len(seen_cmds) == 2

    monkeypatch.setenv("TILERT_SSH_FORCE_PASSWORD", "1")
    monkeypatch.setattr(ac, "_load_ssh_passwords", lambda: {"7.150.10.158": "pw"})
    pw_cmd, pw_env = ac._ssh_base_cmd("7.150.10.158")
    assert pw_cmd[0] == "sshpass"
    assert pw_env is not None and pw_env["SSHPASS"] == "pw"


def test_a3_container_8_15_logical_devices_prestage_and_shmem_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.glm52.e2e_tp8_ep8.hygiene_gate import allocate_isolation_env

    iso_8_15 = allocate_isolation_env(
        "run_test_815",
        port_base=31000,
        target_host="7.150.9.184",
        visible_devices=range(8, 16),
    )
    assert iso_8_15["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15"
    env_sh = ac.build_container_env_script(iso_8_15)
    assert "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in env_sh
    assert "export TILERT_HOST_VISIBLE_DEVICES=8,9,10,11,12,13,14,15" in env_sh

    shmem_script = ac.build_shmem_libs_script()
    assert "libtp_push_kernel.so" in shmem_script
    assert "tilert_shmem_bf16.so" in shmem_script
    assert "-Daclshmemx_sync_vec_all=aclshmemx_barrier_all_vec" in shmem_script

    fake_out = json.dumps(
        {
            "model": "/dev/shm/glm52_shared_ro/GLM-5.2-w4a8c8",
            "mounts": [
                [
                    "/dev/shm/glm52_shared_ro/dspine-glm52/checkpoints/cal_l78_v3_hybrid_best.pt",
                    "/prestage/cal_l78_v3_hybrid_best.pt",
                ]
            ],
        }
    ).encode("utf-8")
    monkeypatch.setattr(
        ac,
        "remote_ssh",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout=fake_out, stderr=b""),
    )
    model_path, mounts = ac._detect_remote_prestage_mounts("7.150.9.184")
    assert model_path == "/dev/shm/glm52_shared_ro/GLM-5.2-w4a8c8"
    assert mounts == [
        (
            "/dev/shm/glm52_shared_ro/dspine-glm52/checkpoints/cal_l78_v3_hybrid_best.pt",
            "/prestage/cal_l78_v3_hybrid_best.pt",
        )
    ]
