"""Time the CANN MegaMoE fused expert layer at one active-expert count.

MegaMoE fuses dispatch, both expert projections, the activation, the combine
and the shared expert into one call, so its cost is one fixed charge plus the
weights the routed experts actually read. Holding the token count fixed while
raising the number of distinct experts the routing selects scales that read
and separates the two, the same way the CANN grouped matmul was separated.

Launch with torch.distributed.run; every rank holds num_experts/ep_world_size
experts, so the operator needs ep_world_size >= 2. One shape per launch.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import torch

HIDDEN = 6144
INTERMEDIATE = 2048
TOP_K = 8


def expert_weights(local_experts, device):
    """One 2D tensor per local expert, the A2/A3 TensorList layout."""
    gate_up = [torch.randn(HIDDEN, 2 * INTERMEDIATE, device=device,
                           dtype=torch.bfloat16)
               for _ in range(local_experts)]
    down = [torch.randn(INTERMEDIATE, HIDDEN, device=device,
                        dtype=torch.bfloat16)
            for _ in range(local_experts)]
    return gate_up, down


def routing(tokens, distinct, num_experts, device, generator):
    """Select `distinct` experts per token, padding the rest by repetition."""
    chosen = torch.randperm(num_experts, generator=generator)[:distinct]
    columns = chosen.repeat((TOP_K + distinct - 1) // distinct)[:TOP_K]
    ids = columns.unsqueeze(0).repeat(tokens, 1).to(torch.int32)
    weights = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32)
    return ids.to(device), weights.to(device)


def batch_mean_us(call, calls):
    torch.npu.synchronize()
    start = perf_counter_ns()
    for _ in range(calls):
        call()
    torch.npu.synchronize()
    return (perf_counter_ns() - start) / calls / 1e3


def measure(args, report):
    import torch_npu
    from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch_npu.npu.set_device(rank)
    device = torch.device("npu", rank)
    torch.distributed.init_process_group("hccl", rank=rank, world_size=world)
    group = torch.distributed.group.WORLD

    local_experts = args.num_experts // world
    report.update({"rank": rank, "ep_world_size": world,
                   "local_experts": local_experts})
    sym_buffer = get_symm_buffer_for_mega_moe(
        group, args.num_experts, args.tokens, TOP_K, HIDDEN, INTERMEDIATE,
    )
    gate_up, down = expert_weights(local_experts, device)
    generator = torch.Generator().manual_seed(args.seed)
    ids, weights = routing(args.tokens, args.distinct, args.num_experts,
                           device, generator)
    x = torch.randn(args.tokens, HIDDEN, device=device, dtype=torch.bfloat16)

    def call():
        return mega_moe(x, ids, weights, gate_up, down, sym_buffer)

    for _ in range(args.warmup):
        call()
    samples = sorted(batch_mean_us(call, args.calls)
                     for _ in range(args.batches))

    median = statistics.median(samples)
    per_expert = (HIDDEN * 2 * INTERMEDIATE + INTERMEDIATE * HIDDEN) * 2
    read = min(args.distinct, local_experts) * per_expert
    report.update({
        "status": "pass", "weight_bytes": read, "median_us": median,
        "min_us": samples[0], "p90_us": samples[-1 - len(samples) // 10],
        "gb_per_s": read / (median / 1e6) / 1e9,
    })
    torch.distributed.destroy_process_group()
    return True


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--distinct", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = arguments()
    report = {
        "status": "error", "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "num_experts": args.num_experts, "distinct": args.distinct,
        "tokens": args.tokens, "top_k": TOP_K, "hidden": HIDDEN,
        "intermediate": INTERMEDIATE, "scene": "A16W16",
        "calls": args.calls, "warmup": args.warmup, "batches": args.batches,
        "scope": "one machine, ep_world_size ranks, no model",
    }
    try:
        passed = measure(args, report)
    except Exception as error:
        report["error"] = repr(error)[:2000]
        passed = False
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"rank-{os.environ.get('RANK', '0')}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": report.get("rank"), "distinct": args.distinct,
                      "passed": passed, "median_us": report.get("median_us"),
                      "gb_per_s": report.get("gb_per_s")}))
    return int(not passed)


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    raise SystemExit(main())
