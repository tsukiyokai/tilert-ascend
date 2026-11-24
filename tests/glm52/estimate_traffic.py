"""Logical read scenarios for the current TP16 CANN model, not HBM traffic."""

import argparse
import hashlib
import json
from pathlib import Path


HIDDEN = 6144
LAYERS = 78
MOE_LAYERS = 75
LOCAL_INTERMEDIATE = 128
VOCAB_SHARD = 9680
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence"
RUNTIME_FILES = {
    "vllm-unified": (
        "vllm/model_executor/models/deepseek_v2.py",
        "vllm/model_executor/models/deepseek_mtp.py",
        "vllm/model_executor/layers/logits_processor.py",
    ),
    "vllm-ascend-unified": (
        "vllm_ascend/attention/sfa_v1.py",
        "vllm_ascend/attention/indexer.py",
        "vllm_ascend/ops/fused_moe/gate_linear.py",
        "vllm_ascend/quantization/methods/w4a8/w4a8.py",
        "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
    ),
}


def w8(outputs, inputs, scale_bytes=2):
    # Ordinary CANN W8 linears read BF16 scales; Prolog owns FP32 copies.
    return outputs * inputs + outputs * scale_bytes


def w4(outputs, inputs):
    # INT32 nibble packing preserves four bits per value. Scale carriers
    # are INT64 and MSD compensation is FP32 per output on this TP rank.
    return outputs * inputs // 2 + outputs * (8 + 4)


def ffn(intermediate, quantized):
    operation = w4 if quantized == 4 else w8
    return (operation(2 * intermediate, HIDDEN)
            + operation(HIDDEN, intermediate))


def components():
    attention = (
        w8(2048, HIDDEN, 4) + w8(576, HIDDEN, 4)
        + w8(1024, 2048, 4)
        + w8(HIDDEN, 1024) + 4 * 448 * 512 * 2
        + (HIDDEN * 2 + 2048 + 512) * 2
    )
    indexer = (w8(4096, 2048) + 128 * HIDDEN * 2
               + 32 * HIDDEN * 2 + 2 * 128 * 4)
    router = 256 * HIDDEN * 2 + 256 * 4
    return {
        "attention": attention, "indexer": indexer, "router": router,
        "shared_ffn": ffn(128, 8), "dense_ffn": ffn(768, 8),
        "one_expert": ffn(LOCAL_INTERMEDIATE, 4),
        "head": VOCAB_SHARD * HIDDEN * 2,
        "embedding_row": HIDDEN * 2, "final_norm": HIDDEN * 2,
    }


def main_weights(experts_per_layer=8, token_rows=1):
    parts = components()
    return {
        "attention": LAYERS * parts["attention"],
        "indexer": 21 * parts["indexer"],
        "router": MOE_LAYERS * parts["router"],
        "shared_ffn": MOE_LAYERS * parts["shared_ffn"],
        "dense_ffn": 3 * parts["dense_ffn"],
        "experts": MOE_LAYERS * experts_per_layer * parts["one_expert"],
        "head": parts["head"], "final_norm": parts["final_norm"],
        "embedding_rows": token_rows * parts["embedding_row"],
    }


def draft_weights(token_rows=1):
    parts = components()
    return {
        "attention": parts["attention"], "indexer": parts["indexer"],
        "router": parts["router"], "shared_ffn": parts["shared_ffn"],
        "experts": 8 * parts["one_expert"], "head": parts["head"],
        "embedding_rows": token_rows * parts["embedding_row"],
        "rot": HIDDEN * HIDDEN * 2,
        "eh_proj": HIDDEN * (HIDDEN * 2) * 2,
        # shared_head.norm is called for recycle and again for logits.
        "enorm_hnorm_head_norm": HIDDEN * 4 * 2,
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration():
    paths = {
        "hbm": EVIDENCE / "a3-hbm-20261017.json",
        "acceptance": EVIDENCE / "ascend-decode/cann-graph-20261017"
        / "mtp-long-bench-audit.json",
        "communication": EVIDENCE / "ascend-decode/tp-mode-20261017"
        / "audit.json",
    }
    records = {name: json.loads(path.read_text())
               for name, path in paths.items()}
    if any(record["status"] != "pass" for record in records.values()):
        raise ValueError("Calibration requires completed audited measurements")
    acceptance = records["acceptance"]["acceptance_per_request"]
    actual = acceptance["actual_sum"] / acceptance["rounds"]
    bandwidth = records["hbm"]["summary"]["read_gb_s"]["p50"]
    chains = {
        mode: [case for case in records["communication"]["groups"]
               ["graph-" + mode]["cases"]
               if case["dtype"] == "torch.float32" and case["calls"] == 156]
        for mode in ("ai_cpu", "aiv")
    }
    return {
        "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                          for path in paths.values()},
        "actual_committed_per_round": actual,
        "natural_accepted_per_round": acceptance["natural_mean"],
        "acceptance_scope": "One prompt, 3 measured 1024-output requests",
        "stream_read_gb_s": bandwidth,
        "bandwidth_scope": "Die 0 FP32 streaming read, event p50; not W4 "
                           "model bandwidth or the minimum across 16 dies",
        "round_budget_ms_at_2ms": 2 * actual,
        "max_round_bytes_at_stream_rate": 2 * actual * bandwidth * 1e6,
        "max_round_bytes_reserving_1ms":
            (2 * actual - 1) * bandwidth * 1e6,
        "measured_fp32_156_sum_graph_chains": chains,
        "communication_scope": (
            "Rank-MAX samples of 156 dependent sums; includes replay and "
            "rank skew, not bare alpha. Not added to the HBM scenarios "
            "as a model time"
        ),
    }


def estimate(runtime_root):
    main_kv = LAYERS * 2048 * (512 + 64) * 2
    main_index = 8192 * (16 * 130 + 5 * 256)
    draft_kv = 2048 * (512 + 64) * 2 + 8192 * 256
    one = main_weights()
    one_bytes = sum(one.values()) + main_kv + main_index
    draft = draft_weights()
    one_draft_bytes = sum(draft.values()) + draft_kv
    # Chain still executes fused wk_weights_proj, including its unused tail.
    skipped_index_query = w8(4096, 2048)
    one_chain_bytes = one_draft_bytes - skipped_index_query - 8192 * 256
    measured = calibration()
    accepted = measured["actual_committed_per_round"]
    bandwidth = measured["stream_read_gb_s"]
    scenarios = []
    for experts in (8, 16, 32):
        for cache_passes in (1, 4):
            # One target-conditioned MTP4 forward plus two chained MTP1
            # forwards per steady M4 round. Initial seeding is separate
            # here; whole-request TPOT includes it after the first callback.
            # The target-conditioned forward may read 8..32 MTP experts.
            conditioned_draft = (
                sum(draft_weights(4).values()) + draft_kv
                + (experts - 8) * components()["one_expert"]
            )
            traffic = (
                sum(main_weights(experts, 4).values())
                + cache_passes * (main_kv + main_index)
                + 2 * one_chain_bytes + conditioned_draft
                + (cache_passes - 1) * draft_kv
            )
            hbm_ms = traffic / (bandwidth * 1e6)
            scenarios.append({
                "verify_width": 4, "expert_union_per_layer": experts,
                "kv_read_passes": cache_passes,
                "round_bytes": traffic,
                "hbm_round_ms_at_stream_rate": hbm_ms,
                "hbm_tpot_ms_at_measured_acceptance": hbm_ms / accepted,
                "accepted_for_2ms_hbm_only": hbm_ms / 2,
            })
    sources = [Path(__file__), *(ROOT / "tilert/models/glm_5_2_ascend" / name
               for name in ("cann_model.py", "cann_mtp.py", "cann_graph.py",
                            "mtp_round.py"))]
    runtime_sources = [runtime_root / repo / name
                       for repo, names in RUNTIME_FILES.items()
                       for name in names]
    fixed_weights = (sum(main_weights(0, 4).values())
                     + 2 * one_chain_bytes
                     + sum(draft_weights(4).values())
                     - 8 * components()["one_expert"])
    return {
        "scope": "Logical weight/KV read scenarios, not measured HBM traffic",
        "layout": "TP16; INT4 packed W4, INT64 scale carriers and FP32 "
                  "compensation; replicated BF16 router with FP32 output; "
                  "BF16 ordinary W8 scales, FP32 Prolog scales; BF16 heads",
        "assumptions": [
            "Each weight is read once per participating expert group",
            "No L2 credit; excludes workspace, activation/copy traffic, MSD "
            "scratch, RoPE/Hadamard tables, zero norm biases and metadata",
            "KV reads bracket one shared pass and four independent passes",
            "M4 main and MTP verify use the same expert-union scenario",
            "qA/kvA and non-shared indexers are replicated on each rank",
            "MTP rot, eh_proj and router are replicated; head is TP-sharded",
            "Expert unions and actual repeated GM reads remain unmeasured",
            "MTP chain skips index q/selection but reads the fused K/weight "
            "projection in full",
            "Excludes Python, launch, host validation and output delivery",
            "Steady rounds exclude initial seed; request TPOT includes it",
            "The 8192 index scan is fixed here; a long request increases it",
            "No claim that the stream-read rate is achievable by this model",
        ],
        "main_m1_weights_by_component": one,
        "main_m1_weight_bytes": sum(one.values()),
        "main_m1_kv_bytes": main_kv + main_index,
        "main_m1_read_bytes": one_bytes,
        "main_m1_hbm_ms_at_stream_rate": one_bytes / (bandwidth * 1e6),
        "draft_m1_weights_by_component": draft,
        "draft_m1_read_bytes": one_draft_bytes,
        "draft_chain_read_bytes": one_chain_bytes,
        "initial_seed_read_bytes": one_draft_bytes,
        "round_formula": {
            "bytes": "fixed + E*(sum(main_layer_unions) + mtp_verify_union) "
                     "+ p*(main_KV_and_index + mtp_KV_and_index)",
            "fixed_bytes_including_two_chains": fixed_weights,
            "one_expert_bytes_E": components()["one_expert"],
            "main_KV_and_index_bytes": main_kv + main_index,
            "mtp_KV_and_index_bytes": draft_kv,
            "scope": "Two chained M1 calls each have eight experts; "
                     "main and conditioned-MTP unions may differ",
        },
        "m1_collectives": {
            "block_allreduce_fp32_24kib": 156,
            "embedding_allreduce_fp32_24kib": 1,
            "logits_allgather_bf16_bytes_per_rank": VOCAB_SHARD * 2,
        },
        "m4_collectives": {
            "block_allreduce_calls": 162,
            "embedding_allreduce_calls": 4,
            "fp32_96kib_calls": 160, "fp32_24kib_calls": 6,
            "logits_allgather_calls": 4,
            "logits_allgather_bf16_bytes_per_rank":
                [rows * VOCAB_SHARD * 2 for rows in (4, 1, 1, 4)],
            "scope": "compute_logits gathers the full vocabulary before "
                     "argmax, even when full_logits=False",
        },
        "calibration": measured,
        "m4_scenarios": scenarios,
        "source_sha256": {str(path.relative_to(ROOT)):
                          digest(path)
                          for path in sources},
        "runtime_source_sha256": {
            str(path.relative_to(runtime_root)): digest(path)
            for path in runtime_sources
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="Parent of vllm-unified and vllm-ascend-unified")
    args = parser.parse_args()
    report = estimate(args.runtime_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "main_m1_read_bytes", "main_m1_hbm_ms_at_stream_rate",
        "draft_m1_read_bytes", "draft_chain_read_bytes",
    )}))


if __name__ == "__main__":
    main()
