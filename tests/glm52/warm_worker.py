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
import statistics
import subprocess
import sys
import time
import types
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
    "cann_model.py",
    "cann_mtp.py",
    "cann_graph.py",
    "whole_layer.py",
    "whole_layer_abi.py",
    "onesided_reduce.py",
    "attention_native.py",
    "routed_shared.py",
    "tp_runtime.py",
)
DSPINE_HOT_FILES: tuple[str, ...] = (
    "__init__.py",
    "calibrated_l78.py",
    "tp_drafter.py",
    "build_dspine_ops.py",
)
L78_VER_BY_KSFA: dict[int, int] = {2048: 110415, 1024: 110419, 512: 110420}


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
    manifest_p = Path(build_dir) / "chain.manifest.json"
    if manifest_p.is_file():
        try:
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            h.update(json.dumps(manifest.get("flags", []), sort_keys=True).encode("utf-8"))
        except (OSError, json.JSONDecodeError):
            h.update(b"bad_manifest")
    model_pkg = Path(repo_root) / "tilert/models/glm_5_2_ascend"
    for name in CORE_BOOT_FILES:
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
    import torch
    from vllm.distributed import get_world_group
    from cache_digest import cache_digest
    from run_m1_case01 import (
        GOLDEN_SHA256,
        _apply_l78_chunked_gemm,
        _prepare_whole_layer_json,
        _serial_draft_digest,
        resolve_trial_protocol,
    )
    from tilert.models.glm_5_2_ascend.cann_model import CannModel
    from tilert.models.glm_5_2_ascend.cann_mtp import CannMtp
    import tilert.models.glm_5_2_ascend.dspine.calibrated_l78 as cal_mod
    import tilert.models.glm_5_2_ascend.dspine.tp_drafter as tp_mod
    from tilert.models.glm_5_2_ascend.ops.build_megakernel import build_chain, default_build_dir
    from tilert.models.glm_5_2_ascend.tp_runtime import TorchControlChannel
    import tilert.models.glm_5_2_ascend.whole_layer as wl
    from tilert.models.glm_5_2_ascend.whole_layer_abi import TreeLibrary
    from tilert.pd_vllm.ascend_cache import FullRankPrefix

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

    orig_install = CannModel._install_tp_sum

    def synced_install(self_m):
        torch.distributed.barrier(get_world_group().cpu_group)
        return orig_install(self_m)

    CannModel._install_tp_sum = synced_install

    raw_build_dir = Path(args.build_dir) if args.build_dir else None
    enable_kernel_prof = (
        bool(getattr(args, "enable_kernel_prof", False))
        or _manifest_has_kernel_prof(raw_build_dir)
    )
    args.enable_kernel_prof = enable_kernel_prof
    wl_json, ops_dir, l78_dir, _ = _prepare_whole_layer_json(args, rank)
    build_dir = wl_json.parent
    args.build_dir = str(build_dir)
    target = CannModel(
        Path(args.model_dir),
        f"npu:{rank}",
        rank,
        capacity=9344,
        gpu_memory_utilization=0.80,
        compile_target=False,
        super_kernel=False,
        attention_kernel="sfa",
        fp32_head=True,
        index_topk=None,
        expert_parallel=True,
        onesided_reduce=os.environ.get("TILERT_ONESIDED_ENDPOINT", "tcp://127.0.0.1:29797"),
        routed_kernel=False,
        routed_shared_kernel=False,
        mla_tail=False,
        native_attention=True,
        whole_layer=wl_json,
        aiv_collectives=True,
        rope_graph=True,
        gather_fia_max_width=None,
        mtp_tokens=0,
        layers=None,
        plain_moe_combine=False,
        graph_width=32,
        hoist_metadata=True,
        hoist_forward_context=True,
        packed_host_inputs=True,
    )
    channel = TorchControlChannel(get_world_group().cpu_group)
    ops_so, _ = tp_mod.load_dspine_ops(rank, out_dir=ops_dir)
    if rank == 0:
        cal_mod.build_draft_l78_k110419(out_dir=l78_dir, k_sfa=args.k_sfa)
    torch.distributed.barrier(group=channel.group)

    case_dir = Path(args.case_dir)
    pf_blob = torch.load(case_dir / "main-prefix/rank-0.pt", map_location="cpu", weights_only=False)
    mtp_blob = torch.load(args.mtp_prefix, map_location="cpu", weights_only=False)
    bench_meta = json.loads((case_dir / "bench_result.json").read_text())
    ref_state = json.loads((case_dir / "reference-state" / f"rank-{rank}.json").read_text())
    prompt = list(pf_blob["prompt_token_ids"])
    first_tok = int(pf_blob["first_token_id"])
    parent = pf_blob["parent_hidden"].to(target.device)
    prefix = FullRankPrefix(pf_blob["layers"], mtp_blob, parent, 8192, tuple(prompt))

    with torch.inference_mode():
        torch.ops.tilert.set_active_tokens_ptr(target._dyn_lens_dev[32], False, False)
    draft = CannMtp(target, shard_projections=True)
    draft._dynamic_mc2_mask = False
    draft._dynamic_sfa_len = False
    draft._dynamic_idx_max = 0
    _apply_l78_chunked_gemm(draft)
    wl.reclaim_hbm(target=target, draft=draft, mgr=target._whole_layer)

    dev = target.device
    raw_mask = torch.zeros(192, dtype=torch.uint8, device=dev)
    mask_off = (-int(raw_mask.data_ptr())) % 64
    tree_mask_i32 = raw_mask[mask_off : mask_off + 128].view(torch.int32)
    tree_mask_i32.zero_()
    torch.npu.synchronize()

    case_b = {
        "slug": "case-01-coding",
        "prefix": prefix,
        "mtp": mtp_blob,
        "first": first_tok,
        "prompt": prompt,
        "golden": bench_meta["output_token_ids"],
        "ref_sha": GOLDEN_SHA256,
        "ref_state": ref_state,
        "digest_fn": cache_digest,
    }
    shared_tgt = tp_mod.capture_shared_target_for_width(
        target, draft, 16, prefix.main, mtp_blob, case_b["golden"], tree_mask_i32, channel.group
    )
    cal_wt = cal_mod.TpCalibratedL78Weights.from_file(
        args.ckpt, rank=rank, dev=dev, ops_so_path=ops_so
    )
    hctrl = cal_mod.ZeroLagHorizonController()
    mgr = target._whole_layer
    mgr.set_tree_mask_ptr(0)
    tree_mask_i32.zero_()
    torch.npu.synchronize()

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
            target.shutdown()
            gc.collect()
            torch.npu.empty_cache()
            return 0

        req_args = cmd.get("args", {})
        req_k_sfa = int(req_args.get("k_sfa", state["k_sfa"]))
        req_ckpt = str(req_args.get("ckpt", state["ckpt"]))
        default_run_dir = Path(os.environ.get("TILERT_RUN_DIR", str(job_dir.parent)))
        req_out = str(req_args.get("out_json", str(default_run_dir / f"warm_epoch_{epoch}.json")))
        warmups, formals, serial_parity = resolve_trial_protocol(
            req_args.get("warmups"),
            req_args.get("formals"),
            bool(req_args.get("fast_parity", True)),
            bool(req_args.get("serial_draft_parity", True)),
        )
        rank_payload: dict[str, Any] = {"epoch": epoch, "rank": rank, "status": "OK"}
        runner = None
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
                    build_chain(
                        out_dir=build_dir,
                        union_sfa=bool(args.union_sfa),
                        enable_kernel_prof=enable_kernel_prof,
                    )
                    cal_mod.build_draft_l78_k110419(out_dir=l78_dir, k_sfa=req_k_sfa)
                torch.distributed.barrier(group=channel.group)
                cur_hot = compute_hot_fingerprints(build_dir, Path(req_ckpt), REPO_ROOT, req_k_sfa)
                cur_state = {"boot_sha": cur_boot, **cur_hot}
                tier = classify_reload_tier(state, cur_state)
                rank_payload["reload_tier"] = tier
                if tier == "chain_and_dspine":
                    reset_all_graphs(target, draft, mgr)
                    for win in reversed(tuple(mgr.windows.values())):
                        win.close()
                    mgr.windows.clear()
                    if mgr.initialized:
                        mgr.tree.finalize()
                    ver_chain = stage_versioned_so(build_dir / "chain.so")
                    mgr.tree = TreeLibrary(ver_chain, None)
                    mgr.tree.initialize(
                        mgr.device.index, mgr.rank, mgr.ranks, mgr.endpoint, mgr.heap_bytes
                    )
                    mgr.window(torch.npu.current_stream(mgr.device), False)
                    shared_tgt = tp_mod.capture_shared_target_for_width(
                        target,
                        draft,
                        16,
                        prefix.main,
                        mtp_blob,
                        case_b["golden"],
                        tree_mask_i32,
                        channel.group,
                    )
                    state["chain_sha"] = cur_hot["chain_sha"]
                if cur_hot["dspine_py_sha"] != state["dspine_py_sha"]:
                    cal_mod = importlib.reload(cal_mod)
                    tp_mod = importlib.reload(tp_mod)
                    hctrl = cal_mod.ZeroLagHorizonController()
                    state["dspine_py_sha"] = cur_hot["dspine_py_sha"]
                if cur_hot["dspine_ops_sha"] != state["dspine_ops_sha"]:
                    ops_so, ops_cdll = tp_mod.load_dspine_ops(rank, out_dir=ops_dir)
                else:
                    ops_cdll = None
                if cur_hot["ckpt_sha"] != state["ckpt_sha"]:
                    cal_wt = cal_mod.TpCalibratedL78Weights.from_file(
                        req_ckpt, rank=rank, dev=dev, ops_so_path=ops_so
                    )
                    state["ckpt_sha"] = cur_hot["ckpt_sha"]
                    state["ckpt"] = req_ckpt
                    state["dspine_ops_sha"] = cur_hot["dspine_ops_sha"]
                elif ops_cdll is not None:
                    cal_wt.ops = ops_cdll
                    state["dspine_ops_sha"] = cur_hot["dspine_ops_sha"]
                if cur_hot["l78_klib_sha"] != state["l78_klib_sha"] or req_k_sfa != state["k_sfa"]:
                    if hasattr(cal_wt, "_p2_klibs_by_ver"):
                        cal_wt._p2_klibs_by_ver.clear()
                    if hasattr(cal_wt, "_p2_klibs"):
                        delattr(cal_wt, "_p2_klibs")
                    state["l78_klib_sha"] = cur_hot["l78_klib_sha"]
                    state["k_sfa"] = req_k_sfa

                mgr.set_tree_mask_ptr(int(tree_mask_i32.data_ptr()))
                tree_mask_i32.zero_()
                runner = tp_mod.CalibratedL78V3Runner(
                    target,
                    draft,
                    shared_tgt,
                    cal_wt,
                    rank=rank,
                    channel_group=channel.group,
                    first_token=first_tok,
                    prefix_parent_hidden=parent,
                    b_scheds=[
                        [4, 4, 4, 4, 4, 2, 2],
                        [4, 4, 4, 4, 2, 2, 2, 2, 1, 1],
                        [4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1],
                    ],
                    horizon_ctrl=hctrl,
                    inv_temp=0.95,
                    beta_margin=0.0,
                    beta_depth=0.0,
                    k_sfa=req_k_sfa,
                    min_w_act=10,
                    t_base_ms=41.5,
                    c_w_ms=0.68,
                    p2_commit=True,
                    narrow_topk=False,
                    l78_build_dir=l78_dir,
                )
                samples = []
                total_runs = warmups + formals
                for idx in range(total_runs):
                    res = runner.decode_full(
                        case_b, "m21-warm-worker", check_digests=(idx == total_runs - 1)
                    )
                    samples.append(res)
                last = samples[-1]
                formal_tpots = [s["max_tpot_ms"] for s in samples[warmups:]]
                median_tpot = round(statistics.median(formal_tpots), 4)
                s_match = None
                s_diffs = None
                ekv_diffs = None
                if serial_parity:
                    runner.close()
                    runner = None
                    s_dig, s_diffs, ekv_diffs = _serial_draft_digest(
                        draft,
                        target,
                        case_b,
                        last["draft_committed_pos"],
                        last["seed_positions"],
                        cal_wt=cal_wt,
                        tree_mask_i32=tree_mask_i32,
                    )
                    s_match = s_dig == last["draft_digest"]
                summary = {
                    "epoch": epoch,
                    "reload_tier": tier,
                    "median_tpot_ms": median_tpot,
                    "formal_tpots_ms": formal_tpots,
                    "output_sha256": last["output_sha256"],
                    "sum_accepted": last["sum_accepted"],
                    "rounds": last["rounds"],
                    "draft_committed_pos": last["draft_committed_pos"],
                    "all_ranks_target_exact": last["all_ranks_target_exact"],
                    "all_ranks_cache_parity": last["all_ranks_cache_parity"],
                    "serial_draft_match": s_match,
                    "serial_draft_diffs": s_diffs,
                    "ekv_serial_diffs": ekv_diffs,
                    "fast_parity": bool(req_args.get("fast_parity", True)),
                }
                if rank == 0:
                    atomic_write_json(Path(req_out), summary)
                    print("WARM_EPOCH_SUMMARY", json.dumps(summary), flush=True)
                parity_ok = (
                    (not serial_parity or s_match is True)
                    and not ekv_diffs
                    and bool(last["all_ranks_target_exact"])
                    and bool(last["all_ranks_cache_parity"])
                )
                if not parity_ok:
                    rank_payload["status"] = "FAIL"
                    rank_payload["error"] = f"Parity mismatch: ekv={ekv_diffs} match={s_match}"
                rank_payload["summary_path"] = req_out
                rank_payload["output_sha256"] = last["output_sha256"]
                rank_payload["serial_draft_match"] = s_match
                rank_payload["ekv_serial_diffs"] = ekv_diffs
                rank_payload["median_tpot_ms"] = median_tpot
        except Exception as exc:
            rank_payload["status"] = "FAIL"
            rank_payload["error"] = f"{type(exc).__name__}: {exc}"
            sys.last_traceback = None
            reset_all_graphs(target, draft, mgr)
        finally:
            if runner is not None:
                runner.close()
            mgr.set_tree_mask_ptr(0)
            tree_mask_i32.zero_()
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
