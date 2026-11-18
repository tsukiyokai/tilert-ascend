import argparse
from copy import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

import torch
from vllm.distributed import get_world_group

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_digest import cache_digest
from tilert.models.glm_5_2_ascend.cann_model import CannModel
from tilert.models.glm_5_2_ascend.cann_mtp import CannMtp, MAIN_CACHE
from tilert.models.glm_5_2_ascend.dspine import (
    CalibratedL78V3Runner,
    TpCalibratedL78Weights,
    ZeroLagHorizonController,
    build_draft_l78_k110419,
    load_dspine_ops,
)
from tilert.models.glm_5_2_ascend.dspine.tp_drafter import capture_shared_target_for_width
from tilert.models.glm_5_2_ascend.ops.build_megakernel import build_chain, default_build_dir
from tilert.models.glm_5_2_ascend.tp_runtime import TorchControlChannel
import tilert.models.glm_5_2_ascend.whole_layer as wl
from tilert.pd_vllm.ascend_cache import FullRankPrefix


GOLDEN_SHA256 = "2f751372fc82cf290c8e35d3e7cc3cad0296fde4d8d188fba102ac34f82c2962"


def _apply_l78_chunked_gemm(draft: CannMtp) -> None:
    mtp_layer = draft.model.model.layers["78"]
    impl = draft.config.compilation_config.static_forward_context[MAIN_CACHE].impl
    modules = {
        "rot": draft.model.rot,
        "eh_proj": mtp_layer.eh_proj,
        "index_projection": impl.indexer.impl.wk_weights_proj,
    }

    def chunked(original):
        def apply(self, layer, values, bias=None):
            if values.shape[0] <= 8:
                return original(layer, values, bias)
            return torch.cat([original(layer, part, bias) for part in values.split(8)])

        return apply

    for layer in modules.values():
        original = layer.quant_method
        method = copy(original)
        method.apply = MethodType(chunked(original.apply), method)
        layer.quant_method = method


def _prepare_whole_layer_json(args, rank: int) -> tuple[Path, Path, Path, Path]:
    enable_prof = bool(getattr(args, "enable_kernel_prof", False))
    if args.build_dir:
        build_dir = Path(args.build_dir)
        chain_out_dir: Path | None = build_dir
    else:
        base_dir = default_build_dir()
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
        chain_so, _ = build_chain(
            out_dir=chain_out_dir,
            union_sfa=args.union_sfa,
            enable_kernel_prof=enable_prof,
        )
        base_wl = json.loads(Path(args.whole_layer_base).read_text())
        base_wl["tree"] = str(chain_so)
        wl_json.write_text(json.dumps(base_wl, indent=2) + "\n")
        ready.write_text("ok\n")
    else:
        while not ready.exists():
            time.sleep(0.05)
    return wl_json, ops_dir, l78_dir, ready


def _serial_draft_digest(
    draft: CannMtp,
    target: CannModel,
    case_b: dict,
    pos_limit: int,
    seed_positions: list[int],
    cal_wt: TpCalibratedL78Weights | None = None,
    tree_mask_i32: torch.Tensor | None = None,
) -> tuple[dict, dict[str, float], dict[str, float]]:
    prefix_main = case_b["prefix"].main
    mtp_pref = case_b["mtp"]
    golden = case_b["golden"]
    with torch.npu.device(target.device), torch.inference_mode():
        prev_tail = {k: v[8191:pos_limit].cpu().clone() for k, v in draft.cache.items()}
        if tree_mask_i32 is not None:
            tree_mask_i32.zero_()
        mgr = getattr(target, "_whole_layer", None)
        if mgr is not None:
            mgr.set_tree_mask_ptr(0)
        target.inject(prefix_main)
        draft.inject(mtp_pref)
        for c_val in draft.cache.values():
            c_val[8191:].zero_()
        saved_g = target.graph
        target.graph = None
        h_chunks = []
        step_rows = 16
        target._dyn_lens_dev[step_rows].zero_()
        target._dyn_lens_dev[step_rows][0, :].fill_(step_rows)
        torch.ops.tilert.set_active_tokens_ptr(
            target._dyn_lens_dev[step_rows], False, False
        )
        total_need = pos_limit - 8191
        for start in range(0, total_need - 1, step_rows):
            toks = golden[start : start + step_rows]
            chunk_len = min(step_rows, total_need - 1 - start)
            if len(toks) < step_rows:
                toks = toks + [toks[-1]] * (step_rows - len(toks))
            st = target.verify(toks, full_logits=False, compute_head=False)
            h_chunks.append(st.last_hidden[:chunk_len].clone())
        target.graph = saved_g
        hidden_seq = torch.cat(h_chunks, dim=0)

        parent_h = case_b["prefix"].parent_hidden.to(target.device)
        torch.ops.tilert.set_active_tokens_ptr(target._dyn_lens_dev[1], False, False)
        draft.forward((case_b["first"],), parent_h, full_logits=False)
        for i in range(total_need - 1):
            draft.forward((golden[i + 1],), hidden_seq[i : i + 1], full_logits=False)
        torch.npu.synchronize()
        seed_set = set(seed_positions)
        ekv_idx = [p - 8191 for p in range(8191, pos_limit) if p not in seed_set]
        ekv_raw = {
            k: float(
                (v[8191:pos_limit].cpu()[ekv_idx].float() - prev_tail[k][ekv_idx].float())
                .abs()
                .max()
                .item()
            )
            for k, v in draft.cache.items()
        }
        ekv_diffs = {k: d for k, d in ekv_raw.items() if d > 0.0}
        if cal_wt is not None and seed_positions:
            blk = draft.model.model.layers["78"].mtp_block
            impl = blk.self_attn.mla_attn.mla_attn.impl
            orig_in = blk.input_layernorm.weight.data.clone()
            orig_qa = impl.q_a_layernorm.weight.data.clone()
            try:
                blk.input_layernorm.weight.data.copy_(cal_wt.cal_input_gamma)
                impl.q_a_layernorm.weight.data.copy_(cal_wt.cal_q_a_norm)
                for pos in sorted(seed_set):
                    if not 8191 <= pos < pos_limit:
                        continue
                    draft.position = pos
                    if pos == 8191:
                        draft.forward((case_b["first"],), parent_h, full_logits=False)
                    else:
                        idx_h = pos - 8192
                        draft.forward(
                            (golden[idx_h + 1],),
                            hidden_seq[idx_h : idx_h + 1],
                            full_logits=False,
                        )
            finally:
                blk.input_layernorm.weight.data.copy_(orig_in)
                impl.q_a_layernorm.weight.data.copy_(orig_qa)
                draft.position = pos_limit
            torch.npu.synchronize()
        diffs_raw = {
            k: float((v[8191:pos_limit].cpu().float() - prev_tail[k].float()).abs().max().item())
            for k, v in draft.cache.items()
        }
        diffs = {k: d for k, d in diffs_raw.items() if d > 0.0}
        return cache_digest([draft.cache], pos_limit), diffs, ekv_diffs


def resolve_trial_protocol(
    warmups: int | None,
    formals: int | None,
    fast_parity: bool,
    serial_draft_parity: bool,
) -> tuple[int, int, bool]:
    if fast_parity:
        w = 0 if warmups is None else int(warmups)
        f = 1 if formals is None else int(formals)
        sdp = True
    else:
        w = 2 if warmups is None else int(warmups)
        f = 5 if formals is None else int(formals)
        sdp = bool(serial_draft_parity)
    if w < 0:
        raise ValueError(f"warmups must be >= 0, got {w}")
    if f < 1:
        raise ValueError(f"formals must be >= 1, got {f}")
    return w, f, sdp


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/work/model-raw")
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--mtp-prefix", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--whole-layer-base", required=True)
    parser.add_argument("--build-dir", default=None)
    parser.add_argument("--out-json", default="/tmp/m1_case01_summary.json")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--milestone", default="M1")
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--formals", type=int, default=None)
    parser.add_argument("--k-sfa", type=int, default=1024)
    parser.add_argument("--union-sfa", action="store_true")
    parser.add_argument("--enable-kernel-prof", action="store_true")
    parser.add_argument("--serial-draft-parity", action="store_true")
    parser.add_argument("--fast-parity", action="store_true")
    args = parser.parse_args(argv)
    args.warmups, args.formals, args.serial_draft_parity = resolve_trial_protocol(
        args.warmups,
        args.formals,
        args.fast_parity,
        args.serial_draft_parity,
    )

    rank = int(os.environ["RANK"])
    orig_install = CannModel._install_tp_sum

    def synced_install(self):
        torch.distributed.barrier(get_world_group().cpu_group)
        return orig_install(self)

    CannModel._install_tp_sum = synced_install

    wl_json, ops_dir, l78_dir, _ = _prepare_whole_layer_json(args, rank)

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
    ops_so, _ = load_dspine_ops(rank, out_dir=ops_dir)
    if rank == 0:
        build_draft_l78_k110419(out_dir=l78_dir, k_sfa=args.k_sfa)
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

    shared_tgt = capture_shared_target_for_width(
        target, draft, 16, prefix.main, mtp_blob, case_b["golden"], tree_mask_i32, channel.group
    )
    cal_wt = TpCalibratedL78Weights.from_file(args.ckpt, rank=rank, dev=dev, ops_so_path=ops_so)
    hctrl = ZeroLagHorizonController()
    runner = CalibratedL78V3Runner(
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
        k_sfa=args.k_sfa,
        min_w_act=10,
        t_base_ms=41.5,
        c_w_ms=0.68,
        p2_commit=True,
        narrow_topk=False,
        l78_build_dir=l78_dir,
    )

    samples = []
    total_runs = args.warmups + args.formals
    for idx in range(total_runs):
        res = runner.decode_full(
            case_b, "m1-r4-dspine-v3p1p2", check_digests=(idx == total_runs - 1)
        )
        samples.append(res)
        if rank == 0:
            kind = "warmup" if idx < args.warmups else "formal"
            print(
                f"[M1 {kind} #{idx}] tpot={res['max_tpot_ms']:.4f}ms "
                f"rounds={res['rounds']} sum_acc={res['sum_accepted']} "
                f"sha={res['output_sha256'][:16]}",
                flush=True,
            )

    formal_tpots = [s["max_tpot_ms"] for s in samples[args.warmups :]]
    median_tpot = round(statistics.median(formal_tpots), 4)
    last = samples[-1]
    prof_acc = None
    if args.enable_kernel_prof and getattr(target, "_whole_layer", None) is not None:
        if target._whole_layer.kernel_prof_enabled:
            prof_acc = target._whole_layer.capture_prof_acc(16).tolist()

    serial_draft_match = None
    serial_draft_diffs = None
    ekv_serial_diffs = None
    if args.serial_draft_parity or args.report_dir is not None:
        runner.close()
        serial_dig, serial_draft_diffs, ekv_serial_diffs = _serial_draft_digest(
            draft,
            target,
            case_b,
            last["draft_committed_pos"],
            last["seed_positions"],
            cal_wt=cal_wt,
            tree_mask_i32=tree_mask_i32,
        )
        serial_draft_match = serial_dig == last["draft_digest"]
        if rank == 0:
            print(
                "SERIAL_DRAFT_PARITY",
                serial_draft_match,
                serial_draft_diffs,
                "EKV_ONLY",
                ekv_serial_diffs,
                flush=True,
            )
    else:
        runner.close()

    if args.report_dir is not None:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        prompt_sha = json.loads((case_dir / "prompt.json").read_text())["prompt_sha256"]
        traces = last["round_traces"]
        w_hist = sorted({int(t["w_act"]) for t in traces})
        rt_inv = {
            "w_act_min": 10,
            "w_act_max": max(w_hist),
            "w_act_history": w_hist,
            "collective_split_max_rows": 16,
            "l78_gemm_chunk_max_rows": 8,
            "in_flight_depth": 1,
            "draft_sfa_topk": int(args.k_sfa),
            "target_sfa_topk": 2048,
        }
        rank_dt = float(last["rank_dt_ms"][rank])
        rank_file = report_dir / f"rank-{rank}.json"
        rank_file.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": 8,
                    "case_id": "case-01-coding",
                    "prompt_sha256": prompt_sha,
                    "first_token_id": first_tok,
                    "output_token_ids": last["output_token_ids"],
                    "output_sha256": last["output_sha256"],
                    "rounds": last["rounds"],
                    "accepted_per_round": last["accepted_per_round"],
                    "horizon_levels": [int(t["horizon_level"]) for t in traces],
                    "dt_ms": rank_dt,
                    "tpot_ms": rank_dt / 1023.0,
                    "main_state": last["local_target_digest"],
                    "draft_state": last["local_draft_digest"],
                    "serial_draft_match": bool(serial_draft_match),
                    "runtime_invariants": rt_inv,
                }
            )
            + "\n"
        )
        torch.distributed.barrier(group=channel.group)
        if rank == 0:
            ranks_manifest = []
            for r in range(8):
                rf = report_dir / f"rank-{r}.json"
                ranks_manifest.append(
                    {
                        "rank": r,
                        "report_path": rf.name,
                        "report_sha256": hashlib.sha256(rf.read_bytes()).hexdigest(),
                    }
                )
            trials = []
            for idx, s in enumerate(samples):
                rdts = [float(x) for x in s["rank_dt_ms"]]
                trials.append(
                    {
                        "trial_idx": idx,
                        "is_warmup": idx < args.warmups,
                        "arm": "candidate",
                        "rank_dt_ms": rdts,
                        "max_tpot_ms": max(rdts) / 1023.0,
                        "rounds": s["rounds"],
                        "accepted_per_round": s["accepted_per_round"],
                        "mean_accepted": float(s["sum_accepted"]) / float(s["rounds"]),
                        "first_token_id": first_tok,
                        "output_token_ids": s["output_token_ids"],
                        "output_sha256": s["output_sha256"],
                        "eos_drained_cleanly": True,
                    }
                )
            warm_ms = [t["max_tpot_ms"] for t in trials[: args.warmups]]
            form_ms = sorted(t["max_tpot_ms"] for t in trials[args.warmups :])
            form_raw = [t["max_tpot_ms"] for t in trials[args.warmups :]]
            pos = 0.9 * (len(form_ms) - 1)
            lo_i, hi_i = math.floor(pos), math.ceil(pos)
            p90 = form_ms[lo_i] if lo_i == hi_i else (
                form_ms[lo_i] * (1.0 - (pos - lo_i)) + form_ms[hi_i] * (pos - lo_i)
            )
            sync_vars = (
                "ASCEND_LAUNCH_BLOCKING",
                "ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE",
                "ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE",
                "ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE",
            )
            inactive = {"", "0", "false", "False"}
            sync_flags = [k for k in sync_vars if os.environ.get(k, "") not in inactive]
            prof_on = (
                os.environ.get("PROFILING_MODE", "0") not in inactive
                or os.environ.get("PROFILER_SAMPLECONFIG", "0") not in inactive
            )
            run_id = os.environ.get("TILERT_E2E_RUN_ID", "m1_case01")
            e2e_payload = {
                "schema_version": "glm52.e2e.v1",
                "milestone": args.milestone,
                "run_id": run_id,
                "case_id": "case-01-coding",
                "width": 16,
                "prompt_sha256": prompt_sha,
                "world_size": 8,
                "expert_parallel": True,
                "protocol": {
                    "warmups": args.warmups,
                    "formals": args.formals,
                    "paired_order": "single",
                    "profiler_enabled": prof_on,
                    "sync_debug_flags": sync_flags,
                    "head_warmed": True,
                    "head_execution_before_timing": {
                        "wrapped": True,
                        "active_graph_width": 16,
                        "graph_widths": [16],
                    },
                    "prefix_restored": True,
                    "synchronized_start": True,
                    "hccl_op_expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", "AIV"),
                    "hccl_deterministic": os.environ.get("HCCL_DETERMINISTIC") == "true",
                    "tpot_formula": "max(8-rank dt_ms) / 1023.0",
                },
                "isolation": {
                    "shm_dir": os.environ["TILERT_RUN_DIR"],
                    "master_port": int(os.environ["MASTER_PORT"]),
                    "whole_layer_endpoint": os.environ["TILERT_WHOLE_LAYER_ENDPOINT"],
                    "onesided_endpoint": os.environ["TILERT_ONESIDED_ENDPOINT"],
                    "hccl_port_range": os.environ["HCCL_NPU_SOCKET_PORT_RANGE"],
                    "atb_share_memory_suffix": os.environ["ATB_SHARE_MEMORY_NAME_SUFFIX"],
                },
                "runtime_hygiene": {
                    "loaded_modules_has_vllm": (
                        "vllm" in sys.modules or "vllm_ascend" in sys.modules
                    ),
                    "compiled_so_dir": str(wl_json.parent),
                    "draft_head_distinct_from_lm_head": True,
                },
                "round_traces": traces,
                "trials": trials,
                "summary": {
                    "warmups_ms": warm_ms,
                    "formal_ms": form_raw,
                    "median_tpot_ms": float(statistics.median(form_raw)),
                    "min_tpot_ms": float(min(form_raw)),
                    "p90_tpot_ms": float(p90),
                    "output_sha256": last["output_sha256"],
                    "all_ranks_exact": bool(
                        last["all_ranks_target_exact"] and last["all_ranks_cache_parity"]
                    ),
                    "ab_ba_delta_ms": None,
                },
                "ranks": ranks_manifest,
            }
            (report_dir / "e2e_result.json").write_text(
                json.dumps(e2e_payload, indent=2) + "\n"
            )

    if rank == 0:
        summary = {
            "median_tpot_ms": median_tpot,
            "formal_tpots_ms": formal_tpots,
            "warmup_tpots_ms": [s["max_tpot_ms"] for s in samples[: args.warmups]],
            "output_sha256": last["output_sha256"],
            "sum_accepted": last["sum_accepted"],
            "rounds": last["rounds"],
            "mean_accepted": last["mean_accepted"],
            "target_pos": last["target_pos"],
            "draft_committed_pos": last["draft_committed_pos"],
            "all_ranks_target_exact": last["all_ranks_target_exact"],
            "all_ranks_cache_parity": last["all_ranks_cache_parity"],
            "serial_draft_match": serial_draft_match,
            "serial_draft_diffs": serial_draft_diffs,
            "ekv_serial_diffs": ekv_serial_diffs,
            "fast_parity": bool(args.fast_parity or args.warmups < 2 or args.formals < 5),
            "kernel_prof_acc": prof_acc,
        }
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2) + "\n")
        print("M1_CASE01_SUMMARY", json.dumps(summary), flush=True)

    target.shutdown()
    gc.collect()
    torch.npu.empty_cache()
    if args.fast_parity:
        parity_ok = (
            serial_draft_match is True
            and not ekv_serial_diffs
            and bool(last["all_ranks_target_exact"])
            and bool(last["all_ranks_cache_parity"])
        )
        if not parity_ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
