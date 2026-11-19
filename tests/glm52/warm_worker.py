import argparse
from collections.abc import Sequence
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GLM52_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GLM52_DIR) not in sys.path:
    sys.path.insert(0, str(GLM52_DIR))

try:
    from tests.glm52.e2e_tp8_ep8.contracts import IsolationError, SHARED_SHM_BLOCKLIST
    from tests.glm52.e2e_tp8_ep8.hygiene_gate import validate_run_id, verify_shm_isolation
except ModuleNotFoundError:
    from e2e_tp8_ep8.contracts import IsolationError, SHARED_SHM_BLOCKLIST
    from e2e_tp8_ep8.hygiene_gate import validate_run_id, verify_shm_isolation

CORE_BOOT_FILES: tuple[str, ...] = (
    "__init__.py",
    "reference.py",
)
M2_BOOT_FILES: tuple[str, ...] = (
    "end2end.py",
    "abi.py",
    "weights.py",
    "config.py",
    "spec.py",
)
DSPINE_HOT_FILES: tuple[str, ...] = (
    "__init__.py",
    "calibrated_l78.py",
    "tp_drafter.py",
    "build_dspine_ops.py",
)
L78_VER_BY_KSFA: dict[int, int] = {2048: 110415, 1024: 110419, 512: 110420}


def default_worker_cfg() -> dict[str, Any]:
    return {
        "b_scheds": [
            [4, 4, 4, 4, 4, 2, 2],
            [4, 4, 4, 4, 2, 2, 2, 2, 1, 1],
            [4, 4, 4, 4, 2, 2, 2, 2, 1, 1],
        ],
        "min_w_act": 10,
        "t_base_ms": 41.5,
        "c_w_ms": 0.68,
        "narrow_topk": True,
    }


def validate_job_dir(job_dir: Path | str, run_id: str | None = None) -> Path:
    raw = str(job_dir).strip()
    if not raw or ".." in Path(raw).parts or ".." in raw.split("/"):
        raise IsolationError(f"Invalid TILERT_JOB_DIR path traversal: {job_dir!r}")
    collapsed = re.sub(r"^/+", "/", raw)
    norm = os.path.normpath(collapsed)
    if norm in SHARED_SHM_BLOCKLIST or norm.startswith("/dev/shm/agy_tp8/"):
        raise IsolationError(f"Shared /dev/shm job dir forbidden: {norm!r}")
    if norm == "/dev/shm" or norm.startswith("/dev/shm/"):
        parts = Path(norm).parts
        if len(parts) < 4:
            raise IsolationError(f"Job dir under /dev/shm must isolate a run subdir: {norm!r}")
        inferred = parts[3] if run_id is None else run_id
        validate_run_id(inferred)
        verify_shm_isolation(norm, inferred)
    return Path(norm)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stage_versioned_so(so_path: Path) -> Path:
    sha10 = _file_sha256(so_path)[:10]
    ver_so = so_path.with_name(f"{so_path.stem}_{sha10}.so")
    if not ver_so.exists():
        tmp = ver_so.with_name(f".{ver_so.name}.{os.getpid()}.tmp")
        shutil.copy2(so_path, tmp)
        os.replace(tmp, ver_so)
    return ver_so


def compute_boot_fingerprint(
    whole_layer_base: Path,
    model_dir: str,
    build_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> str:
    h = hashlib.sha256()
    h.update(str(model_dir).encode("utf-8"))
    wl_path = Path(whole_layer_base)
    if wl_path.is_file():
        h.update(wl_path.read_bytes())
    model_pkg = Path(repo_root) / "tilert/models/glm_5_2_ascend"
    for name in (*CORE_BOOT_FILES, *M2_BOOT_FILES):
        fp = model_pkg / name
        h.update(name.encode("utf-8"))
        h.update(_file_sha256(fp).encode("utf-8"))
    return h.hexdigest()


def compute_hot_fingerprints(
    build_dir: Path,
    ckpt_path: Path,
    repo_root: Path = REPO_ROOT,
    k_sfa: int = 1024,
) -> dict[str, str]:
    bdir = Path(build_dir)
    ver = L78_VER_BY_KSFA.get(int(k_sfa), 110419)
    dspine_dir = Path(repo_root) / "tilert/models/glm_5_2_ascend/dspine"
    py_h = hashlib.sha256()
    for name in DSPINE_HOT_FILES:
        py_h.update(name.encode("utf-8"))
        py_h.update(_file_sha256(dspine_dir / name).encode("utf-8"))
    ckpt_p = Path(ckpt_path)
    if ckpt_p.exists():
        st = ckpt_p.stat()
        ckpt_sig = f"{os.path.realpath(ckpt_p)}:{st.st_size}:{st.st_mtime_ns}"
    else:
        ckpt_sig = "missing"
    return {
        "chain_sha": _file_sha256(bdir / "chain.so"),
        "dspine_ops_sha": _file_sha256(bdir / "dspine_ops/tilert_dspine_ops.so"),
        "l78_klib_sha": _file_sha256(bdir / "dspine_l78" / f"chain_K{ver}.so"),
        "dspine_py_sha": py_h.hexdigest(),
        "ckpt_sha": hashlib.sha256(ckpt_sig.encode("utf-8")).hexdigest(),
    }


def classify_reload_tier(prev_state: dict[str, Any], cur_state: dict[str, Any]) -> str:
    if prev_state.get("boot_sha") != cur_state.get("boot_sha"):
        return "reboot"
    if prev_state.get("chain_sha") != cur_state.get("chain_sha"):
        return "chain_and_dspine"
    for key in ("dspine_py_sha", "dspine_ops_sha", "l78_klib_sha", "ckpt_sha"):
        if prev_state.get(key) != cur_state.get(key):
            return "dspine"
    return "none"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    stat_p = Path(f"/proc/{pid}/stat")
    if stat_p.is_file():
        try:
            stat_txt = stat_p.read_text(encoding="utf-8", errors="replace")
            rparen = stat_txt.rfind(")")
            if rparen != -1:
                after = stat_txt[rparen + 2 :].split(None, 1)
                if after and after[0] == "Z":
                    return False
        except OSError:
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_process_group(pgid: int, grace_s: float = 2.0) -> None:
    if pgid <= 0:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _next_epoch(job_dir: Path) -> int:
    highest = 0
    for done_file in job_dir.glob("epoch-*-done.json"):
        parts = done_file.stem.split("-")
        if len(parts) == 3 and parts[1].isdigit():
            highest = max(highest, int(parts[1]))
    cmd_file = job_dir / "worker_cmd.json"
    if cmd_file.is_file():
        try:
            cmd = json.loads(cmd_file.read_text(encoding="utf-8"))
            if isinstance(cmd, dict) and isinstance(cmd.get("epoch"), int):
                highest = max(highest, int(cmd["epoch"]))
        except (OSError, json.JSONDecodeError):
            pass
    return highest + 1


def _read_log_tail(log_path: Path, max_lines: int = 40) -> str:
    if not log_path.is_file():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def aggregate_epoch_ranks(job_dir: Path, epoch: int, world_size: int = 8) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    failed_ranks: list[int] = []
    errors: dict[str, str] = {}
    for rank in range(world_size):
        rp = job_dir / f"epoch-{epoch}-rank-{rank}.json"
        if not rp.is_file():
            failed_ranks.append(rank)
            errors[str(rank)] = f"Missing rank report {rp.name}"
            continue
        data = json.loads(rp.read_text(encoding="utf-8"))
        reports.append(data)
        if data.get("status") != "OK":
            failed_ranks.append(rank)
            errors[str(rank)] = str(data.get("error", "unknown error"))
    reboot_required = any(bool(r.get("reboot_required")) for r in reports)
    ok = len(reports) == world_size and not failed_ranks and not reboot_required
    r0 = reports[0] if reports else {}
    return {
        "epoch": epoch,
        "ok": ok,
        "reboot_required": reboot_required,
        "failed_ranks": failed_ranks,
        "errors": errors,
        "dt_ms": max((float(r.get("elapsed_seconds", 0.0)) * 1000.0 for r in reports), default=0.0),
        "reload_tier": r0.get("reload_tier", "none"),
        "summary_path": r0.get("summary_path"),
        "output_sha256": r0.get("output_sha256"),
        "serial_draft_match": r0.get("serial_draft_match"),
        "ekv_serial_diffs": r0.get("ekv_serial_diffs"),
        "in_flight_depth": r0.get("in_flight_depth"),
        "drain_q_len": r0.get("drain_q_len"),
        "eos_drained_cleanly": r0.get("eos_drained_cleanly"),
        "median_tpot_ms": r0.get("median_tpot_ms"),
    }


def reset_all_graphs(target: Any, draft: Any, mgr: Any) -> None:
    if getattr(draft, "_l78_win", None) is not None:
        if getattr(draft._l78_win, "handle", 0) != 0:
            draft._l78_win.close()
        draft._l78_win = None
    if getattr(target, "graph", None) is not None:
        if hasattr(target.graph, "close"):
            target.graph.close()
        else:
            target.graph.reset()
        target.graph = None
    for graph, _ in getattr(target, "_head_graphs", {}).values():
        graph.reset()
    if hasattr(target, "_head_graphs"):
        target._head_graphs.clear()
    for captured in getattr(target, "_rope_graphs", {}).values():
        captured.graph.reset()
    if hasattr(target, "_rope_graphs"):
        target._rope_graphs.clear()
    acl_mod = sys.modules.get("vllm_ascend.compilation.acl_graph")
    if acl_mod is not None and hasattr(acl_mod, "_graph_params"):
        acl_mod._graph_params = None
    if mgr is not None:
        mgr.chains.clear()
        mgr.chain_reserves.clear()
        mgr.pending_steps.clear()
        mgr.chain_active = False
        mgr.capture_width = None
        mgr.capture_seen.clear()
        mgr.capture_owner = None
        mgr.capture_stream = None
        mgr.slot_refreshed.clear()
        if hasattr(mgr, "arena_pool"):
            mgr.arena_pool.clear()
        for layer in mgr.layers.values():
            layer.entries.clear()
            layer.reserves.clear()
    nat_pool = getattr(target, "_native_attention_workspace_pool", None)
    if nat_pool is not None:
        for attr in ("entries", "capture_reserves"):
            val = getattr(nat_pool, attr, None)
            if isinstance(val, dict):
                val.clear()


def _manifest_has_kernel_prof(build_dir: Path | None) -> bool:
    if build_dir is None:
        return False
    manifest_p = Path(build_dir) / "chain.manifest.json"
    if not manifest_p.is_file():
        return False
    try:
        data = json.loads(manifest_p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    flags = data.get("flags") or ()
    return bool(data.get("enable_kernel_prof", False) or "-DENABLE_KERNEL_PROF=1" in flags)


def _run_worker(args: argparse.Namespace) -> int:
    from datetime import timedelta
    import torch
    from run_m2_case01 import _prepare_build_artifacts, resolve_trial_protocol
    from tilert.models.glm_5_2_ascend.abi import (
        PreparedChain,
        SHMEM_HEAP_BYTES,
        TreeLibrary,
        Window,
    )
    from tilert.models.glm_5_2_ascend.config import RuntimeConfig
    import tilert.models.glm_5_2_ascend.dspine.calibrated_l78 as cal_mod
    import tilert.models.glm_5_2_ascend.dspine.tp_drafter as tp_mod
    from tilert.models.glm_5_2_ascend.end2end import Glm52AscendShowHands, MAIN_LAYERS, WIDTHS

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError(f"warm_worker requires WORLD_SIZE=8, got {world_size}")
    job_dir = validate_job_dir(
        args.job_dir or os.environ["TILERT_JOB_DIR"],
        os.environ.get("TILERT_E2E_RUN_ID"),
    )
    ack_dir = job_dir / "worker_ack"
    ack_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(rank)
    torch.npu.config.allow_internal_format = True
    torch.distributed.init_process_group(backend="gloo", timeout=timedelta(hours=24))
    tp_group = torch.distributed.group.WORLD

    raw_build_dir = Path(args.build_dir) if args.build_dir else None
    enable_kernel_prof = (
        bool(getattr(args, "enable_kernel_prof", False))
        or _manifest_has_kernel_prof(raw_build_dir)
    )
    args.enable_kernel_prof = enable_kernel_prof
    if not hasattr(args, "whole_layer_json"):
        args.whole_layer_json = None
    build_dir, wl_json, ops_dir, l78_dir = _prepare_build_artifacts(args, rank)
    torch.distributed.barrier(group=tp_group)
    args.build_dir = str(build_dir)

    case_dir = Path(args.case_dir)
    main_pf = case_dir / "main-prefix/rank-0.pt"
    cfg = RuntimeConfig(
        model_dir=Path(args.model_dir),
        cal_l78_path=Path(args.ckpt),
        whole_layer_json=wl_json,
        rank=rank,
        world_size=world_size,
        device=f"npu:{rank}",
    )
    runner = Glm52AscendShowHands(
        cfg,
        case_dir,
        Path(args.mtp_prefix),
        build_dir,
        k_sfa=int(args.k_sfa),
        width=16,
        main_prefix_path=main_pf,
        rank=rank,
        tp_group=tp_group,
    )
    runner.load()
    dev = runner.device
    tree_reload_idx = 0
    cur_cfg = default_worker_cfg()

    boot_sha = compute_boot_fingerprint(
        Path(args.whole_layer_base), args.model_dir, build_dir, REPO_ROOT
    )
    state: dict[str, Any] = {
        "boot_sha": boot_sha,
        **compute_hot_fingerprints(build_dir, Path(args.ckpt), REPO_ROOT, args.k_sfa),
        "k_sfa": int(args.k_sfa),
        "ckpt": str(args.ckpt),
    }

    ready_ack = ack_dir / f"ready_{rank}"
    ready_ack.write_text("ready\n", encoding="utf-8")
    ready_file = job_dir / "daemon_ready.json"
    if rank == 0:
        for r in range(world_size):
            r_ack = ack_dir / f"ready_{r}"
            while not r_ack.exists():
                time.sleep(0.02)
        for r in range(world_size):
            (ack_dir / f"ready_{r}").unlink(missing_ok=True)
        atomic_write_json(
            ready_file,
            {"world_size": world_size, "boot_sha": boot_sha, "epoch": 0, "pid": os.getpid()},
        )
    else:
        while not ready_file.is_file():
            time.sleep(0.02)

    cmd_file = job_dir / "worker_cmd.json"
    epoch = 1
    while True:
        cmd: dict[str, Any] | None = None
        while cmd is None:
            if cmd_file.is_file():
                try:
                    loaded = json.loads(cmd_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    loaded = None
                if isinstance(loaded, dict) and loaded.get("epoch") == epoch:
                    cmd = loaded
                    break
            time.sleep(0.02)

        action = str(cmd.get("action", "eval"))
        t0 = time.perf_counter()
        if action == "exit":
            atomic_write_json(
                job_dir / f"epoch-{epoch}-rank-{rank}.json",
                {"epoch": epoch, "rank": rank, "status": "OK", "elapsed_seconds": 0.0},
            )
            ack_f = ack_dir / f"rank_{rank}_{epoch}"
            ack_f.write_text("ok\n", encoding="utf-8")
            if rank == 0:
                for r in range(world_size):
                    while not (ack_dir / f"rank_{r}_{epoch}").exists():
                        time.sleep(0.02)
                for r in range(world_size):
                    (ack_dir / f"rank_{r}_{epoch}").unlink(missing_ok=True)
                done_p = aggregate_epoch_ranks(job_dir, epoch, world_size)
                atomic_write_json(job_dir / f"epoch-{epoch}-done.json", done_p)
            else:
                while ack_f.exists():
                    time.sleep(0.02)
            runner.close()
            return 0

        req_args = cmd.get("args", {})
        req_k_sfa = int(req_args.get("k_sfa") or state["k_sfa"])
        req_ckpt = str(req_args.get("ckpt") or state["ckpt"])
        req_union = bool(req_args.get("union_sfa", args.union_sfa))
        req_cfg = {
            "b_scheds": req_args.get("b_scheds", cur_cfg["b_scheds"]),
            "min_w_act": int(req_args.get("min_w_act", cur_cfg["min_w_act"])),
            "t_base_ms": float(req_args.get("t_base_ms", cur_cfg["t_base_ms"])),
            "c_w_ms": float(req_args.get("c_w_ms", cur_cfg["c_w_ms"])),
            "narrow_topk": bool(req_args.get("narrow_topk", cur_cfg["narrow_topk"])),
        }
        default_run_dir = Path(os.environ.get("TILERT_RUN_DIR", str(job_dir.parent)))
        req_out = str(req_args.get("out_json", str(default_run_dir / f"warm_epoch_{epoch}.json")))
        warmups, formals, serial_parity = resolve_trial_protocol(
            req_args.get("warmups"),
            req_args.get("formals"),
            bool(req_args.get("fast_parity", True)),
            bool(req_args.get("serial_draft_parity", True)),
        )
        rank_payload: dict[str, Any] = {"epoch": epoch, "rank": rank, "status": "OK"}
        try:
            cur_boot = compute_boot_fingerprint(
                Path(args.whole_layer_base), args.model_dir, build_dir, REPO_ROOT
            )
            if cur_boot != state["boot_sha"]:
                rank_payload["status"] = "REBOOT_REQUIRED"
                rank_payload["reboot_required"] = True
                rank_payload["reload_tier"] = "reboot"
            else:
                if rank == 0:
                    import tilert.models.glm_5_2_ascend.ops.build_megakernel as bm_mod

                    bm_mod = importlib.reload(bm_mod)
                    bm_mod.build_chain(
                        out_dir=build_dir,
                        union_sfa=req_union,
                        enable_kernel_prof=enable_kernel_prof,
                    )
                    cal_mod = importlib.reload(cal_mod)
                    cal_mod.build_draft_l78_k110419(out_dir=l78_dir, k_sfa=req_k_sfa)
                torch.distributed.barrier(group=tp_group)
                cur_hot = compute_hot_fingerprints(
                    build_dir, Path(req_ckpt), REPO_ROOT, req_k_sfa
                )
                cur_state = {"boot_sha": cur_boot, **cur_hot}
                tier = classify_reload_tier(state, cur_state)
                cfg_changed = req_cfg != cur_cfg or req_k_sfa != state["k_sfa"]
                if tier == "none" and cfg_changed:
                    tier = "dspine"
                rank_payload["reload_tier"] = tier
                if tier == "chain_and_dspine":
                    torch.npu.synchronize(dev)
                    torch.ops.tilert.set_active_tokens_ptr(runner.dyn_w, False, False)
                    if runner.g_round is not None:
                        runner.g_round.reset()
                        runner.g_round = None
                    if runner.drafter is not None:
                        runner.drafter.close()
                        runner.drafter = None
                    if runner.target_win is not None and runner.target_win.handle != 0:
                        runner.target_win.close()
                        runner.target_win = None
                    if runner.l78_win is not None and runner.l78_win.handle != 0:
                        runner.l78_win.close()
                        runner.l78_win = None
                    if runner.tree_lib is not None:
                        runner.tree_lib.finalize()
                        runner.tree_lib = None
                    runner.prepared_chain = None
                    runner._arena_pool.clear()
                    gc.collect()
                    ver_chain = stage_versioned_so(build_dir / "chain.so")
                    runner.tree_lib = TreeLibrary(ver_chain, None)
                    tree_reload_idx += 1
                    whole_ep = os.environ.get(
                        "TILERT_WHOLE_LAYER_ENDPOINT", "tcp://127.0.0.1:29590"
                    )
                    ep_host, ep_port = whole_ep.rsplit(":", 1)
                    runner.tree_lib.initialize(
                        runner.device.index,
                        runner.rank,
                        runner.world_size,
                        f"{ep_host}:{int(ep_port) + 20 + tree_reload_idx}",
                        SHMEM_HEAP_BYTES,
                    )
                    with torch.npu.stream(runner.cap_stream):
                        runner.target_win = Window(runner.tree_lib, max(WIDTHS))
                        runner.l78_win = Window(runner.tree_lib, 32)
                    steps_layers = runner._build_chain_steps(runner.l0_normed, runner.t_emb_out)
                    chain_steps_buf = runner.arena.bind_chain_step_buffers(MAIN_LAYERS)
                    runner.prepared_chain = PreparedChain(
                        steps_layers,
                        chain_steps_buf,
                        copy_stream=None,
                        mask_ptr=int(runner.arena.tree_mask_i32.data_ptr()),
                        peer_weights_ptr=int(runner.arena.peer_weights_ptr),
                    )
                    state["chain_sha"] = cur_hot["chain_sha"]

                need_drafter_rebuild = (
                    tier in ("chain_and_dspine", "dspine")
                    or cfg_changed
                    or runner.g_round is None
                )
                if need_drafter_rebuild:
                    if runner.g_round is not None:
                        torch.npu.synchronize(dev)
                        torch.ops.tilert.set_active_tokens_ptr(runner.dyn_w, False, False)
                        runner.g_round.reset()
                        runner.g_round = None
                    if runner.drafter is not None:
                        runner.drafter.close()
                        runner.drafter = None
                    if cur_hot["dspine_py_sha"] != state["dspine_py_sha"]:
                        cal_mod = importlib.reload(cal_mod)
                        tp_mod = importlib.reload(tp_mod)
                        state["dspine_py_sha"] = cur_hot["dspine_py_sha"]
                    if cur_hot["dspine_ops_sha"] != state["dspine_ops_sha"]:
                        _, runner.ops_cdll = tp_mod.load_dspine_ops(rank, out_dir=ops_dir)
                        if runner.arena.draft is not None:
                            runner.arena.draft.cal_wt.ops = runner.ops_cdll
                        state["dspine_ops_sha"] = cur_hot["dspine_ops_sha"]
                    if cur_hot["ckpt_sha"] != state["ckpt_sha"]:
                        cal_wt = cal_mod.TpCalibratedL78Weights.from_file(
                            req_ckpt,
                            rank=rank,
                            dev=dev,
                            ops_so_path=str(ops_dir / "tilert_dspine_ops.so"),
                        )
                        if runner.arena.draft is not None:
                            runner.arena.draft.cal_wt = cal_wt
                        state["ckpt_sha"] = cur_hot["ckpt_sha"]
                        state["ckpt"] = req_ckpt
                    runner.k_sfa = req_k_sfa
                    state["l78_klib_sha"] = cur_hot["l78_klib_sha"]
                    state["k_sfa"] = req_k_sfa
                    cur_cfg = dict(req_cfg)
                    runner.hctrl = cal_mod.ZeroLagHorizonController()
                    runner.drafter = tp_mod.CalibratedL78V3Runner(
                        runner.arena,
                        runner.prefix,
                        runner.metadata_lib,
                        runner.ffn_ops,
                        runner.l78_win,
                        runner.target_ctx,
                        runner.tp_group,
                        runner.prefix.first_token_id,
                        b_scheds=cur_cfg["b_scheds"],
                        horizon_ctrl=runner.hctrl,
                        inv_temp=runner.cfg.inv_temp,
                        beta_margin=0.0,
                        beta_depth=0.0,
                        k_sfa=req_k_sfa,
                        min_w_act=cur_cfg["min_w_act"],
                        t_base_ms=cur_cfg["t_base_ms"],
                        c_w_ms=cur_cfg["c_w_ms"],
                        narrow_topk=cur_cfg["narrow_topk"],
                        l78_build_dir=l78_dir,
                    )
                    runner._capture_g_round()

                report = runner.warmup_and_run(
                    warmups=warmups,
                    formals=formals,
                    max_steps=1023,
                    check_serial=serial_parity,
                )
                last_trial = report["trials"][-1]
                traces = report.get("round_traces") or []
                drain_q_len = traces[-1].get("drain_q_len") if traces else None
                in_flight_depth = report.get("runtime_invariants", {}).get("in_flight_depth")
                eos_drained = bool(report.get("eos_drained_cleanly", False)) and bool(
                    last_trial.get("eos_drained_cleanly", False)
                )
                summary = {
                    "epoch": epoch,
                    "reload_tier": tier,
                    "median_tpot_ms": round(float(report["median_tpot_ms"]), 4),
                    "formal_tpots_ms": [round(float(x), 4) for x in report["formal_ms"]],
                    "output_sha256": last_trial["output_sha256"],
                    "sum_accepted": report["sum_accepted"],
                    "mean_accepted": report["mean_accepted"],
                    "rounds": last_trial["rounds"],
                    "draft_committed_pos": report["draft_committed_pos"],
                    "all_ranks_target_exact": report["all_ranks_target_exact"],
                    "all_ranks_cache_parity": report["all_ranks_cache_parity"],
                    "serial_draft_match": report["serial_draft_match"],
                    "serial_draft_diffs": report["serial_draft_diffs"],
                    "ekv_serial_diffs": report["ekv_serial_diffs"],
                    "loaded_modules_has_vllm": report["loaded_modules_has_vllm"],
                    "in_flight_depth": in_flight_depth,
                    "drain_q_len": drain_q_len,
                    "eos_drained_cleanly": eos_drained,
                    "fast_parity": bool(req_args.get("fast_parity", True)),
                    "kernel_prof_acc": report.get("kernel_prof_acc"),
                    "cfg": cur_cfg,
                }
                if rank == 0:
                    atomic_write_json(Path(req_out), summary)
                    print("WARM_EPOCH_SUMMARY", json.dumps(summary), flush=True)
                parity_ok = (
                    (not serial_parity or report["serial_draft_match"] is True)
                    and not report["ekv_serial_diffs"]
                    and bool(report["all_ranks_target_exact"])
                    and bool(report["all_ranks_cache_parity"])
                    and not bool(report["loaded_modules_has_vllm"])
                    and in_flight_depth == 2
                    and drain_q_len == 0
                    and eos_drained is True
                )
                if not parity_ok:
                    rank_payload["status"] = "FAIL"
                    rank_payload["error"] = (
                        f"Parity mismatch: ekv={report['ekv_serial_diffs']} "
                        f"match={report['serial_draft_match']} "
                        f"depth={in_flight_depth} drain={drain_q_len} eos={eos_drained}"
                    )
                rank_payload["summary_path"] = req_out
                rank_payload["output_sha256"] = last_trial["output_sha256"]
                rank_payload["serial_draft_match"] = report["serial_draft_match"]
                rank_payload["ekv_serial_diffs"] = report["ekv_serial_diffs"]
                rank_payload["in_flight_depth"] = in_flight_depth
                rank_payload["drain_q_len"] = drain_q_len
                rank_payload["eos_drained_cleanly"] = eos_drained
                rank_payload["median_tpot_ms"] = round(float(report["median_tpot_ms"]), 4)
        except Exception as exc:
            rank_payload["status"] = "FAIL"
            rank_payload["error"] = f"{type(exc).__name__}: {exc}"
            sys.last_traceback = None
            if runner.g_round is not None:
                runner.g_round.reset()
                runner.g_round = None
        finally:
            gc.collect()
            torch.npu.empty_cache()

        rank_payload["elapsed_seconds"] = round(time.perf_counter() - t0, 4)
        atomic_write_json(job_dir / f"epoch-{epoch}-rank-{rank}.json", rank_payload)
        ack_f = ack_dir / f"rank_{rank}_{epoch}"
        ack_f.write_text("done\n", encoding="utf-8")
        if rank == 0:
            for r in range(world_size):
                while not (ack_dir / f"rank_{r}_{epoch}").exists():
                    time.sleep(0.02)
            for r in range(world_size):
                (ack_dir / f"rank_{r}_{epoch}").unlink(missing_ok=True)
            done_p = aggregate_epoch_ranks(job_dir, epoch, world_size)
            atomic_write_json(job_dir / f"epoch-{epoch}-done.json", done_p)
        else:
            while ack_f.exists():
                time.sleep(0.02)
        epoch += 1


def _clean_stale_ipc(job_dir: Path) -> None:
    (job_dir / "worker_cmd.json").unlink(missing_ok=True)
    (job_dir / "daemon_ready.json").unlink(missing_ok=True)
    for pattern in ("epoch-*-done.json", "epoch-*-rank-*.json"):
        for stale in job_dir.glob(pattern):
            stale.unlink(missing_ok=True)
    ack_dir = job_dir / "worker_ack"
    if ack_dir.is_dir():
        for ack in ack_dir.iterdir():
            if ack.is_file():
                ack.unlink(missing_ok=True)


def boot_daemon(
    job_dir: Path,
    *,
    model_dir: str,
    case_dir: str,
    mtp_prefix: str,
    ckpt: str,
    whole_layer_base: str,
    build_dir: str | None = None,
    k_sfa: int = 1024,
    union_sfa: bool = False,
    enable_kernel_prof: bool = False,
    boot_timeout_s: float = 600.0,
    worker_cmd_override: Sequence[str] | None = None,
) -> dict[str, Any]:
    job_dir = validate_job_dir(job_dir, os.environ.get("TILERT_E2E_RUN_ID"))
    job_dir.mkdir(parents=True, exist_ok=True)
    _clean_stale_ipc(job_dir)
    if build_dir:
        (Path(build_dir) / "whole_layer.ready").unlink(missing_ok=True)
    log_path = job_dir / "daemon.log"
    if worker_cmd_override is not None:
        cmd = list(worker_cmd_override)
    else:
        master_port = os.environ.get("MASTER_PORT", "29900")
        cmd = [
            "torchrun",
            "--nproc_per_node=8",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--job-dir",
            str(job_dir),
            "--model-dir",
            model_dir,
            "--case-dir",
            case_dir,
            "--mtp-prefix",
            mtp_prefix,
            "--ckpt",
            ckpt,
            "--whole-layer-base",
            whole_layer_base,
            "--k-sfa",
            str(k_sfa),
        ]
        if build_dir:
            cmd.extend(["--build-dir", str(build_dir)])
        if union_sfa:
            cmd.append("--union-sfa")
        if enable_kernel_prof:
            cmd.append("--enable-kernel-prof")
    log_fh = log_path.open("ab")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_fh.close()
    pgid = os.getpgid(proc.pid)
    pid_info = {"pid": proc.pid, "pgid": pgid, "job_dir": str(job_dir)}
    atomic_write_json(job_dir / "daemon.pid", pid_info)
    ready_file = job_dir / "daemon_ready.json"
    deadline = time.monotonic() + boot_timeout_s
    while time.monotonic() < deadline:
        if ready_file.is_file():
            ready_data = json.loads(ready_file.read_text(encoding="utf-8"))
            return {**pid_info, **ready_data}
        if not _pid_alive(proc.pid) or proc.poll() is not None:
            _kill_process_group(pgid)
            raise RuntimeError(
                f"Warm worker died during boot (rc={proc.poll()}):\n{_read_log_tail(log_path)}"
            )
        time.sleep(0.1)
    _kill_process_group(pgid)
    raise TimeoutError(
        f"Warm worker boot timed out after {boot_timeout_s}s:\n{_read_log_tail(log_path)}"
    )


def submit_epoch(
    job_dir: Path,
    *,
    action: str = "eval",
    cmd_args: dict[str, Any] | None = None,
    epoch_timeout_s: float = 600.0,
) -> dict[str, Any]:
    job_dir = validate_job_dir(job_dir, os.environ.get("TILERT_E2E_RUN_ID"))
    pid_file = job_dir / "daemon.pid"
    if not pid_file.is_file():
        raise RuntimeError(f"Daemon not running: missing {pid_file}")
    pid_info = json.loads(pid_file.read_text(encoding="utf-8"))
    pid = int(pid_info["pid"])
    pgid = int(pid_info.get("pgid", pid))
    if not _pid_alive(pid):
        raise RuntimeError(f"Daemon PID {pid} is dead:\n{_read_log_tail(job_dir / 'daemon.log')}")
    epoch = _next_epoch(job_dir)
    done_file = job_dir / f"epoch-{epoch}-done.json"
    atomic_write_json(
        job_dir / "worker_cmd.json",
        {"epoch": epoch, "action": action, "args": dict(cmd_args or {})},
    )
    deadline = time.monotonic() + epoch_timeout_s
    while time.monotonic() < deadline:
        if done_file.is_file():
            return json.loads(done_file.read_text(encoding="utf-8"))
        if not _pid_alive(pid):
            _kill_process_group(pgid)
            raise RuntimeError(
                f"Warm worker PID {pid} exited mid-epoch {epoch}:\n"
                f"{_read_log_tail(job_dir / 'daemon.log')}"
            )
        time.sleep(0.1)
    _kill_process_group(pgid)
    raise TimeoutError(
        f"Warm worker epoch {epoch} timed out after {epoch_timeout_s}s:\n"
        f"{_read_log_tail(job_dir / 'daemon.log')}"
    )


def stop_daemon(job_dir: Path, timeout_s: float = 30.0) -> dict[str, Any]:
    job_dir = validate_job_dir(job_dir, os.environ.get("TILERT_E2E_RUN_ID"))
    pid_file = job_dir / "daemon.pid"
    if not pid_file.is_file():
        return {"stopped": True, "was_running": False}
    pid_info = json.loads(pid_file.read_text(encoding="utf-8"))
    pid = int(pid_info["pid"])
    pgid = int(pid_info.get("pgid", pid))
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    if _pid_alive(pid):
        try:
            submit_epoch(job_dir, action="exit", epoch_timeout_s=timeout_s)
        except Exception:
            _kill_process_group(pgid)
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _pid_alive(pid):
            _kill_process_group(pgid)
            _pid_alive(pid)
    else:
        _kill_process_group(pgid)
    pid_file.unlink(missing_ok=True)
    (job_dir / "daemon_ready.json").unlink(missing_ok=True)
    return {"stopped": True, "was_running": True, "pid": pid}


def daemon_status(job_dir: Path) -> dict[str, Any]:
    job_dir = validate_job_dir(job_dir, os.environ.get("TILERT_E2E_RUN_ID"))
    pid_file = job_dir / "daemon.pid"
    ready_file = job_dir / "daemon_ready.json"
    pid_info = json.loads(pid_file.read_text(encoding="utf-8")) if pid_file.is_file() else {}
    ready_info = json.loads(ready_file.read_text(encoding="utf-8")) if ready_file.is_file() else {}
    pid = int(pid_info["pid"]) if "pid" in pid_info else 0
    return {
        "alive": _pid_alive(pid) if pid else False,
        "ready": ready_file.is_file(),
        "pid": pid or None,
        "boot_sha": ready_info.get("boot_sha"),
        "next_epoch": _next_epoch(job_dir),
    }


def boot_or_submit(args: argparse.Namespace) -> dict[str, Any]:
    job_dir = validate_job_dir(
        args.job_dir or os.environ["TILERT_JOB_DIR"],
        os.environ.get("TILERT_E2E_RUN_ID"),
    )
    raw_bdir = Path(args.build_dir) if args.build_dir else None
    enable_prof = (
        bool(getattr(args, "enable_kernel_prof", False))
        or _manifest_has_kernel_prof(raw_bdir)
    )
    if raw_bdir is not None:
        bdir = raw_bdir
    else:
        base_bdir = Path(os.environ.get("TILERT_BUILD_DIR", "."))
        bdir = base_bdir / "prof" if enable_prof else base_bdir
    cur_boot = compute_boot_fingerprint(
        Path(args.whole_layer_base), args.model_dir, bdir, REPO_ROOT
    )
    st = daemon_status(job_dir)
    if not st["alive"] or not st["ready"] or st.get("boot_sha") != cur_boot:
        stop_daemon(job_dir)
        boot_daemon(
            job_dir,
            model_dir=args.model_dir,
            case_dir=args.case_dir,
            mtp_prefix=args.mtp_prefix,
            ckpt=args.ckpt,
            whole_layer_base=args.whole_layer_base,
            build_dir=args.build_dir,
            k_sfa=args.k_sfa,
            union_sfa=args.union_sfa,
            enable_kernel_prof=enable_prof,
            boot_timeout_s=args.boot_timeout,
        )
    out_json = args.out_json or str(job_dir.parent / "warm_epoch_summary.json")
    cmd_args = {
        "k_sfa": args.k_sfa,
        "ckpt": args.ckpt,
        "warmups": args.warmups,
        "formals": args.formals,
        "fast_parity": bool(args.fast_parity),
        "serial_draft_parity": bool(args.serial_draft_parity),
        "out_json": out_json,
    }
    res = submit_epoch(
        job_dir,
        action="eval",
        cmd_args=cmd_args,
        epoch_timeout_s=args.epoch_timeout,
    )
    if res.get("reboot_required"):
        stop_daemon(job_dir)
        boot_daemon(
            job_dir,
            model_dir=args.model_dir,
            case_dir=args.case_dir,
            mtp_prefix=args.mtp_prefix,
            ckpt=args.ckpt,
            whole_layer_base=args.whole_layer_base,
            build_dir=args.build_dir,
            k_sfa=args.k_sfa,
            union_sfa=args.union_sfa,
            enable_kernel_prof=enable_prof,
            boot_timeout_s=args.boot_timeout,
        )
        res = submit_epoch(
            job_dir, action="eval", cmd_args=cmd_args, epoch_timeout_s=args.epoch_timeout
        )
    return res


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("worker", "submit", "status", "stop", "boot-or-submit"),
        required=True,
    )
    parser.add_argument("--job-dir", default=None)
    parser.add_argument("--model-dir", default="/work/model-raw")
    parser.add_argument("--case-dir", default="")
    parser.add_argument("--mtp-prefix", default="")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--whole-layer-base", default="")
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--k-sfa", type=int, default=1024)
    parser.add_argument("--union-sfa", action="store_true")
    parser.add_argument("--enable-kernel-prof", action="store_true")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--formals", type=int, default=1)
    parser.add_argument("--fast-parity", action="store_true", default=True)
    parser.add_argument("--no-fast-parity", dest="fast_parity", action="store_false")
    parser.add_argument("--serial-draft-parity", action="store_true", default=True)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--epoch-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    if args.mode == "worker":
        return _run_worker(args)
    job_dir = Path(args.job_dir or os.environ["TILERT_JOB_DIR"])
    if args.mode == "status":
        print(json.dumps(daemon_status(job_dir), indent=2))
        return 0
    if args.mode == "stop":
        print(json.dumps(stop_daemon(job_dir), indent=2))
        return 0
    if args.mode == "submit":
        out_json = args.out_json or str(job_dir.parent / "warm_epoch_summary.json")
        res = submit_epoch(
            job_dir,
            action="eval",
            cmd_args={
                "k_sfa": args.k_sfa,
                "ckpt": args.ckpt,
                "warmups": args.warmups,
                "formals": args.formals,
                "fast_parity": bool(args.fast_parity),
                "serial_draft_parity": bool(args.serial_draft_parity),
                "out_json": out_json,
            },
            epoch_timeout_s=args.epoch_timeout,
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1
    if args.mode == "boot-or-submit":
        res = boot_or_submit(args)
        print(json.dumps(res, indent=2))
        return 0 if res.get("ok") else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
