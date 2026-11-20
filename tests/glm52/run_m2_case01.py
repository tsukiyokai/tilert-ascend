import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GLM52_DIR = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(GLM52_DIR) not in sys.path:
    sys.path.insert(0, str(GLM52_DIR))

try:
    from tests.glm52.e2e_tp8_ep8.contracts import (
        DEFAULT_CASE_ROOT,
        GOLDEN_CASES,
        GOLDEN_TRACKS,
        MILESTONES,
        WIDTHS,
        get_case_spec,
        milestone_tpot_limit,
        resolve_track,
        tpot_formula_for_world_size,
    )
    from tests.glm52.e2e_tp8_ep8.parity_gate import verify_e2e_run, verify_prompt_case
except ModuleNotFoundError:
    from e2e_tp8_ep8.contracts import (
        DEFAULT_CASE_ROOT,
        GOLDEN_CASES,
        GOLDEN_TRACKS,
        MILESTONES,
        WIDTHS,
        get_case_spec,
        milestone_tpot_limit,
        resolve_track,
        tpot_formula_for_world_size,
    )
    from e2e_tp8_ep8.parity_gate import verify_e2e_run, verify_prompt_case

FORBIDDEN_VLLM_PREFIXES: tuple[str, ...] = ("vllm", "vllm_ascend", "tilert.pd_vllm")
DEFAULT_WHOLE_LAYER_BASE: dict[str, str] = {
    "indexed": "/assets/runs/v3x58/build/indexed/whole_layer_indexed_attention_tp8.so",
    "shared": "/assets/runs/v3x58/build/full/whole_layer_full_attention_tp8.so",
    "ffn": "/sfs/shanshan/dspine-glm52/runtime_bundle_v5/agy_tp8/build/complete_ffn_tp8_ep8.so",
    "metadata": (
        "/sfs/shanshan/dspine-glm52/runtime_bundle_v5/agy_tp8/build/"
        "layer_local_chain_tp8_ep8.so"
    ),
}


def loaded_modules_has_vllm(modules: Any = None) -> bool:
    loaded = sys.modules if modules is None else modules
    return any(
        mod == prefix or mod.startswith(f"{prefix}.")
        for mod in loaded
        for prefix in FORBIDDEN_VLLM_PREFIXES
    )


def resolve_trial_protocol(
    warmups: int | None,
    formals: int | None,
    fast_parity: bool,
    serial_draft_parity: bool,
    *,
    report_dir: str | Path | None = None,
    strict_formal: bool = False,
) -> tuple[int, int, bool]:
    if report_dir is not None or (strict_formal and not fast_parity):
        w = 2 if warmups is None else int(warmups)
        f = 5 if formals is None else int(formals)
        if w < 2:
            raise ValueError(f"formal protocol requires warmups >= 2, got {w}")
        if f < 5:
            raise ValueError(f"formal protocol requires formals >= 5, got {f}")
        return w, f, True
    if fast_parity:
        w = 0 if warmups is None else int(warmups)
        f = 1 if formals is None else int(formals)
        sdp = bool(serial_draft_parity) if strict_formal else True
    else:
        w = 2 if warmups is None else int(warmups)
        f = 5 if formals is None else int(formals)
        sdp = bool(serial_draft_parity)
    if w < 0:
        raise ValueError(f"warmups must be >= 0, got {w}")
    if f < 1:
        raise ValueError(f"formals must be >= 1, got {f}")
    return w, f, sdp


def _resolve_case_paths(
    case_id: str,
    case_dir_arg: str | None,
    main_prefix_arg: str | None,
) -> tuple[Path, Path | None]:
    case_dir = Path(case_dir_arg) if case_dir_arg else (DEFAULT_CASE_ROOT / case_id)
    if main_prefix_arg:
        return case_dir, Path(main_prefix_arg)
    local_main = case_dir / "main-prefix" / "rank-0.pt"
    if local_main.is_file():
        return case_dir, local_main
    prestage_main = Path(f"/prestage/cases-8k1k/{case_id}/main-prefix/rank-0.pt")
    if prestage_main.is_file():
        return case_dir, prestage_main
    return case_dir, None


def _resolve_base_whole_layer(args: argparse.Namespace) -> dict[str, str]:
    for cand in (args.whole_layer_json, args.whole_layer_base):
        if cand and Path(cand).is_file():
            return json.loads(Path(cand).read_text(encoding="utf-8"))
    run_dir = os.environ.get("TILERT_RUN_DIR")
    if run_dir:
        cand_run = Path(run_dir) / "whole_layer_base.json"
        if cand_run.is_file():
            return json.loads(cand_run.read_text(encoding="utf-8"))
    return dict(DEFAULT_WHOLE_LAYER_BASE)


def _prepare_build_artifacts(
    args: argparse.Namespace,
    rank: int,
) -> tuple[Path, Path, Path, Path]:
    from tilert.models.glm_5_2_ascend.dspine import build_draft_l78_k110419
    from tilert.models.glm_5_2_ascend.dspine.build_dspine_ops import build_dspine_ops
    from tilert.models.glm_5_2_ascend.ops.build_megakernel import build_chain, default_build_dir

    enable_prof = bool(getattr(args, "enable_kernel_prof", False))
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    if args.build_dir:
        build_dir = Path(args.build_dir)
        chain_out_dir: Path | None = build_dir
    else:
        base_dir = default_build_dir() if world_size == 8 else default_build_dir(world_size)
        build_dir = base_dir / "prof" if enable_prof else base_dir
        chain_out_dir = None
    wl_json = build_dir / "whole_layer.json"
    ready = build_dir / "whole_layer.ready"
    ops_dir = build_dir / "dspine_ops"
    l78_dir = build_dir / "dspine_l78"
    if rank == 0:
        build_dir.mkdir(parents=True, exist_ok=True)
        if ready.exists():
            ready.unlink()
        with ThreadPoolExecutor(max_workers=3) as pool:
            if world_size == 8:
                f_chain = pool.submit(
                    build_chain,
                    out_dir=chain_out_dir,
                    union_sfa=bool(args.union_sfa),
                    enable_kernel_prof=enable_prof,
                )
                f_ops = pool.submit(build_dspine_ops, ops_dir)
                f_l78 = pool.submit(
                    build_draft_l78_k110419, out_dir=l78_dir, k_sfa=int(args.k_sfa)
                )
            else:
                f_chain = pool.submit(
                    build_chain,
                    out_dir=chain_out_dir,
                    union_sfa=bool(args.union_sfa),
                    enable_kernel_prof=enable_prof,
                    world_size=world_size,
                )
                f_ops = pool.submit(build_dspine_ops, ops_dir, world_size=world_size)
                f_l78 = pool.submit(
                    build_draft_l78_k110419,
                    out_dir=l78_dir,
                    k_sfa=int(args.k_sfa),
                    world_size=world_size,
                )
            chain_so, _ = f_chain.result()
            f_ops.result()
            f_l78.result()
        base_wl = _resolve_base_whole_layer(args)
        base_wl["tree"] = str(chain_so)
        wl_json.write_text(json.dumps(base_wl, indent=2) + "\n", encoding="utf-8")
        ready.write_text("ok\n", encoding="utf-8")
    else:
        while not ready.exists():
            time.sleep(0.05)
    return build_dir, wl_json, ops_dir, l78_dir


def _sync_debug_flags() -> list[str]:
    sync_vars = (
        "ASCEND_LAUNCH_BLOCKING",
        "ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE",
        "ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE",
        "ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE",
    )
    inactive = {"", "0", "false", "False"}
    return [k for k in sync_vars if os.environ.get(k, "") not in inactive]


def _profiler_enabled() -> bool:
    inactive = {"", "0", "false", "False"}
    return (
        os.environ.get("PROFILING_MODE", "0") not in inactive
        or os.environ.get("PROFILER_SAMPLECONFIG", "0") not in inactive
    )


def _isolation_block(run_id: str) -> dict[str, Any]:
    return {
        "shm_dir": os.environ.get("TILERT_RUN_DIR", f"/dev/shm/{run_id}"),
        "master_port": int(os.environ.get("MASTER_PORT", "29900")),
        "whole_layer_endpoint": os.environ.get(
            "TILERT_WHOLE_LAYER_ENDPOINT", "tcp://127.0.0.1:29901"
        ),
        "onesided_endpoint": os.environ.get(
            "TILERT_ONESIDED_ENDPOINT", "tcp://127.0.0.1:29902"
        ),
        "hccl_port_range": os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", "30000-30150"),
        "atb_share_memory_suffix": os.environ.get(
            "ATB_SHARE_MEMORY_NAME_SUFFIX", f"glm52_{run_id}"
        ),
    }


def build_rank_report(res: dict[str, Any], case_id: str) -> dict[str, Any]:
    return {
        "rank": int(res["rank"]),
        "world_size": int(res["world_size"]),
        "case_id": case_id,
        "prompt_sha256": res["prompt_sha256"],
        "first_token_id": int(res["first_token_id"]),
        "output_token_ids": res["output_token_ids"],
        "output_sha256": res["output_sha256"],
        "rounds": int(res["rounds"]),
        "accepted_per_round": res["accepted_per_round"],
        "horizon_levels": res["horizon_levels"],
        "dt_ms": float(res["dt_ms"]),
        "tpot_ms": float(res["tpot_ms"]),
        "main_state": res["main_state"],
        "draft_state": res["draft_state"],
        "serial_draft_match": bool(res["serial_draft_match"]),
        "ekv_serial_diffs": res.get("ekv_serial_diffs"),
        "runtime_invariants": res["runtime_invariants"],
    }


def build_e2e_report(
    res: dict[str, Any],
    *,
    case_id: str,
    milestone: str,
    width: int,
    warmups: int,
    formals: int,
    ranks_manifest: list[dict[str, Any]],
    run_id: str,
    track: str | None = None,
) -> dict[str, Any]:
    world_size = int(res["world_size"])
    resolved_track = resolve_track(track, world_size=world_size).track
    return {
        "schema_version": "glm52.e2e.v1",
        "milestone": milestone,
        "track": resolved_track,
        "run_id": run_id,
        "case_id": case_id,
        "width": int(width),
        "prompt_sha256": res["prompt_sha256"],
        "world_size": world_size,
        "expert_parallel": True,
        "protocol": {
            "warmups": int(warmups),
            "formals": int(formals),
            "paired_order": "single",
            "profiler_enabled": _profiler_enabled(),
            "sync_debug_flags": _sync_debug_flags(),
            "head_warmed": bool(res["head_warmed"]),
            "head_execution_before_timing": res["head_execution_before_timing"],
            "prefix_restored": True,
            "synchronized_start": True,
            "hccl_op_expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", "AIV"),
            "hccl_deterministic": os.environ.get("HCCL_DETERMINISTIC", "").lower() == "true",
            "tpot_formula": tpot_formula_for_world_size(world_size),
        },
        "isolation": _isolation_block(run_id),
        "runtime_hygiene": {
            "loaded_modules_has_vllm": bool(
                loaded_modules_has_vllm() or res["loaded_modules_has_vllm"]
            ),
            "compiled_so_dir": res["compiled_so_dir"],
            "draft_head_distinct_from_lm_head": bool(res["draft_head_distinct_from_lm_head"]),
        },
        "round_traces": res["round_traces"],
        "trials": res["trials"],
        "summary": {
            "warmups_ms": res["warmups_ms"],
            "formal_ms": res["formal_ms"],
            "median_tpot_ms": float(res["median_tpot_ms"]),
            "min_tpot_ms": float(res["min_tpot_ms"]),
            "p90_tpot_ms": float(res["p90_tpot_ms"]),
            "output_sha256": res["output_sha256"],
            "all_ranks_exact": bool(
                res["all_ranks_target_exact"] and res["all_ranks_cache_parity"]
            ),
            "ekv_serial_diffs": res.get("ekv_serial_diffs"),
            "ab_ba_delta_ms": None,
        },
        "ranks": ranks_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", default=None)
    parser.add_argument("--case-id", choices=tuple(GOLDEN_CASES), default="case-01-coding")
    parser.add_argument("--track", choices=tuple(GOLDEN_TRACKS), default=None)
    parser.add_argument("--main-prefix", default=None)
    parser.add_argument("--mtp-prefix", required=True)
    parser.add_argument("--model-dir", default="/work/model-raw")
    parser.add_argument("--rot-path", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--whole-layer-json", default=None)
    parser.add_argument("--whole-layer-base", default=None)
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--milestone", choices=MILESTONES, default="M2")
    parser.add_argument("--k-sfa", type=int, default=1024)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--formals", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1023)
    parser.add_argument(
        "--fast-parity", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--serial-draft-parity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--union-sfa", action="store_true")
    parser.add_argument("--enable-kernel-prof", action="store_true")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--out-json", default="/tmp/m2_case01_summary.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    import torch

    from tilert.models.glm_5_2_ascend.config import RuntimeConfig
    from tilert.models.glm_5_2_ascend.end2end import Glm52AscendShowHands

    args = build_parser().parse_args(argv)
    if args.width not in WIDTHS:
        raise ValueError(f"Unsupported width: {args.width} (expected in {WIDTHS})")
    args.warmups, args.formals, check_serial = resolve_trial_protocol(
        args.warmups,
        args.formals,
        args.fast_parity,
        args.serial_draft_parity,
        report_dir=args.report_dir,
        strict_formal=True,
    )

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "8"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", timeout=timedelta(hours=24))
    tp_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None

    case_dir, main_prefix_path = _resolve_case_paths(
        args.case_id, args.case_dir, args.main_prefix
    )
    track_spec = resolve_track(args.track, world_size=world_size)
    spec = get_case_spec(args.case_id, track=track_spec.track)
    if rank == 0:
        verify_prompt_case(case_dir, spec, track=track_spec.track)

    build_dir, wl_json, _, _ = _prepare_build_artifacts(args, rank)
    if tp_group is not None:
        torch.distributed.barrier(group=tp_group)

    cfg = RuntimeConfig(
        model_dir=Path(args.model_dir),
        cal_l78_path=Path(args.ckpt),
        whole_layer_json=wl_json,
        rank=rank,
        rot_path=Path(args.rot_path) if args.rot_path else None,
        world_size=world_size,
        ep_size=world_size,
        local_experts=256 // world_size,
        device=f"npu:{rank}",
    )
    runner = Glm52AscendShowHands(
        cfg,
        case_dir,
        Path(args.mtp_prefix),
        build_dir,
        k_sfa=int(args.k_sfa),
        width=int(args.width),
        main_prefix_path=main_prefix_path,
        rank=rank,
        tp_group=tp_group,
    )
    try:
        runner.load()
        res = runner.warmup_and_run(
            args.warmups,
            args.formals,
            args.max_steps,
            check_serial,
        )
        prof_acc = res.get("kernel_prof_acc")
        if args.enable_kernel_prof and prof_acc is None and runner.kernel_prof_enabled:
            prof_acc = runner.capture_prof_acc().tolist()
    finally:
        runner.close()

    if args.report_dir is not None:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        rank_file = report_dir / f"rank-{rank}.json"
        rank_payload = build_rank_report(res, args.case_id)
        rank_file.write_text(json.dumps(rank_payload) + "\n", encoding="utf-8")
        if tp_group is not None:
            torch.distributed.barrier(group=tp_group)
        if rank == 0:
            ranks_manifest = [
                {
                    "rank": r,
                    "report_path": f"rank-{r}.json",
                    "report_sha256": hashlib.sha256(
                        (report_dir / f"rank-{r}.json").read_bytes()
                    ).hexdigest(),
                }
                for r in range(world_size)
            ]
            run_id = os.environ.get("TILERT_E2E_RUN_ID", "m2_case01")
            e2e_payload = build_e2e_report(
                res,
                case_id=args.case_id,
                milestone=args.milestone,
                width=args.width,
                warmups=args.warmups,
                formals=args.formals,
                ranks_manifest=ranks_manifest,
                run_id=run_id,
                track=track_spec.track,
            )
            (report_dir / "e2e_result.json").write_text(
                json.dumps(e2e_payload, indent=2) + "\n",
                encoding="utf-8",
            )
            if args.warmups >= 2 and args.formals >= 5:
                verify_e2e_run(
                    report_dir,
                    case_dir,
                    max_tpot_ms=milestone_tpot_limit(args.milestone),
                    require_warm_head=True,
                    milestone=args.milestone,
                    track=track_spec.track,
                )

    vllm_loaded = loaded_modules_has_vllm() or bool(res["loaded_modules_has_vllm"])
    traces = res["round_traces"]
    drain_q_len = traces[-1].get("drain_q_len") if traces else None
    serial_ok = not check_serial or (
        res["serial_draft_match"] is True and not res["ekv_serial_diffs"]
    )
    parity_ok = (
        not vllm_loaded
        and res["output_sha256"] == spec.output_sha256
        and bool(res["all_ranks_target_exact"])
        and bool(res["all_ranks_cache_parity"])
        and serial_ok
        and res["runtime_invariants"].get("in_flight_depth") == 2
        and drain_q_len == 0
    )

    if rank == 0:
        summary = {
            "case_id": args.case_id,
            "track": track_spec.track,
            "world_size": world_size,
            "ep_size": world_size,
            "local_experts": 256 // world_size,
            "milestone": args.milestone,
            "width": int(args.width),
            "k_sfa": int(args.k_sfa),
            "median_tpot_ms": round(float(res["median_tpot_ms"]), 4),
            "min_tpot_ms": round(float(res["min_tpot_ms"]), 4),
            "p90_tpot_ms": round(float(res["p90_tpot_ms"]), 4),
            "formal_tpots_ms": res["formal_ms"],
            "warmup_tpots_ms": res["warmups_ms"],
            "output_sha256": res["output_sha256"],
            "sum_accepted": int(res["sum_accepted"]),
            "rounds": int(res["rounds"]),
            "mean_accepted": round(float(res["mean_accepted"]), 3),
            "target_pos": int(res["target_pos"]),
            "draft_committed_pos": int(res["draft_committed_pos"]),
            "all_ranks_target_exact": bool(res["all_ranks_target_exact"]),
            "all_ranks_cache_parity": bool(res["all_ranks_cache_parity"]),
            "serial_draft_match": res["serial_draft_match"],
            "serial_draft_diffs": res["serial_draft_diffs"],
            "ekv_serial_diffs": res["ekv_serial_diffs"],
            "loaded_modules_has_vllm": vllm_loaded,
            "in_flight_depth": res["runtime_invariants"].get("in_flight_depth"),
            "drain_q_len": drain_q_len,
            "fast_parity": bool(
                args.report_dir is None
                and (args.fast_parity or args.warmups < 2 or args.formals < 5)
            ),
            "kernel_prof_acc": prof_acc,
            "parity_ok": parity_ok,
        }
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("M2_CASE01_SUMMARY", json.dumps(summary), flush=True)

    if not parity_ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
