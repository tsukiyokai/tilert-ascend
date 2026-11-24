"""Time one W4A8 expert-GMM path at a given active-expert and row count.

Each active group reads its own weight shard, so sweeping the number of
non-zero groups scales HBM traffic at a fixed kernel shape and separates the
call's fixed cost from its streaming slope; sweeping rows per group scales
the activation side instead. Values do not affect timing, so the bank is
filled with arbitrary bits. One shape per process. One device, one rank.
"""

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

import torch

EXPERTS = 256
HIDDEN = 6144
LOCAL_INTERMEDIATE = 128
PACK = 8
TOP_K = 8
L2_BYTES = 201326592
EVICTION_FACTOR = 4
PROJECTIONS = {"gate_up": (HIDDEN, 2 * LOCAL_INTERMEDIATE),
               "down": (LOCAL_INTERMEDIATE, HIDDEN)}


def rotation_banks(group_bytes, groups):
    """Rotate far enough that a bank leaves L2 before it is read again.

    A call reads `groups × group_bytes`, and the reuse distance is one bank
    short of the rotation, so a small group count needs a long rotation: at
    the model's 27 groups a single bank is a tenth of L2. Banks are always
    allocated in full, so the footprint grows as the measured shape shrinks
    and small group counts cost gigabytes of HBM.

    Four times L2 is the shortest reuse distance at which 256 groups stop
    reporting a marginal rate above the measured single-die HBM maximum.
    Rotating further than that raises the time again, so a longer rotation
    is not a safer one.
    """
    call_bytes = groups * group_bytes
    return 1 + -(-EVICTION_FACTOR * L2_BYTES // call_bytes)


def build_banks(inputs, outputs, device, count):
    """Allocate `count` independent banks so a rotation defeats L2 reuse.

    One bank of 256 `gate_up` groups is exactly the die's 192 MiB L2, so
    reading a single bank repeatedly measures cache reread rather than HBM.
    Rotating over several banks makes every call read memory the previous
    calls did not touch.
    """
    return [{
        "weight": torch.randint(-2 ** 31, 2 ** 31 - 1,
                                (EXPERTS, inputs, outputs // PACK),
                                dtype=torch.int32, device=device),
        "scale": torch.ones(EXPERTS, 1, outputs, dtype=torch.int64,
                            device=device),
        "scale_bias": torch.zeros(EXPERTS, outputs, dtype=torch.float32,
                                  device=device),
    } for _ in range(count)]


def build_routing(experts, rows, device):
    """Route as the model does: every token picks `TOP_K` experts.

    The matmul produces `experts × rows` permuted rows, so that row count
    is the token count times the fan-out, and `probs` carries one weight
    per pick as a two-dimensional tensor. The combine therefore gathers
    `tokens × TOP_K` rows, which is what the model pays at 27 groups over
    four tokens. A group count that is not a multiple of the fan-out leaves
    the last token's picks wrapping onto earlier rows.
    """
    permuted = experts * rows
    tokens = -(-permuted // TOP_K)
    return {
        "tokens": tokens,
        "sorted_indices": torch.arange(tokens * TOP_K, dtype=torch.int32,
                                       device=device).remainder(permuted),
        "row_index": torch.arange(permuted, dtype=torch.int64,
                                  device=device) // TOP_K,
        "probs": torch.ones(tokens, TOP_K, dtype=torch.bfloat16,
                            device=device),
        "logit": torch.ones(permuted, dtype=torch.float32, device=device),
    }


def build_activation(rows, inputs, device):
    return {
        "x": torch.ones(rows, inputs, dtype=torch.int8, device=device),
        "per_token_scale": torch.ones(rows, dtype=torch.float32,
                                      device=device),
    }


def grouped_matmul_call(banks, activation, counts, routing,
                        group_list_type=1):
    """Time the expert GMM, with the group list in either of its two forms.

    `group_list_type=1` passes per-group counts and `0` passes their running
    sum, and the vendor kernel treats them very differently: with counts it
    walks the list with scalar global loads, recomputing a prefix sum inside
    its workspace-split loop, where with a running sum the same quantity is
    one load of the last entry. The engine passes counts.
    """
    import torch_npu

    rotation = itertools.cycle(banks)
    group_list = counts if group_list_type else counts.cumsum(0)

    def call():
        bank = next(rotation)
        return torch_npu.npu_grouped_matmul(
            x=[activation["x"]], weight=[bank["weight"]],
            scale=[bank["scale"]], bias=[bank["scale_bias"]],
            per_token_scale=[activation["per_token_scale"]],
            group_list=group_list, split_item=2, group_type=0,
            group_list_type=group_list_type, output_dtype=torch.bfloat16,
        )[0]

    return call


def unpermuted_call(banks, activation, counts, routing):
    """Time the production pair: grouped matmul, then the routing combine."""
    import torch_npu

    matmul = grouped_matmul_call(banks, activation, counts, routing)

    def call():
        return torch_npu.npu_moe_token_unpermute(
            matmul(), routing["sorted_indices"], routing["probs"],
        )

    return call


def finalize_routing_call(banks, activation, counts, routing):
    """Time the fused alternative: grouped matmul with the combine inside.

    Float32 is this operator's only output dtype, it takes the W4A8 scale
    in its int64 carrier form and rejects a float one, and on A3 a null
    bias is an error, so the call cannot be made to mirror the unpermuted
    path's bfloat16 output. Its W4A8 mode also demands k > 128, which the
    `down` projection of a TP16 rank does not have, so that shape raises
    instead of measuring.
    """
    import torch_npu

    rotation = itertools.cycle(banks)

    def call():
        bank = next(rotation)
        return torch_npu.npu_grouped_matmul_finalize_routing(
            activation["x"], bank["weight"], counts,
            scale=bank["scale"], bias=bank["scale_bias"],
            pertoken_scale=activation["per_token_scale"],
            logit=routing["logit"], row_index=routing["row_index"],
            dtype=torch.float32, output_bs=routing["tokens"],
            group_list_type=1,
        )

    return call


PATHS = {"grouped": grouped_matmul_call, "unpermuted": unpermuted_call,
         "finalize_routing": finalize_routing_call}


def batch_mean_us(call, calls):
    torch.npu.synchronize()
    start = perf_counter_ns()
    for _ in range(calls):
        call()
    torch.npu.synchronize()
    return (perf_counter_ns() - start) / calls / 1e3


def measure(args, report):
    import torch_npu

    torch_npu.npu.set_device(args.device)
    device = torch.device("npu", args.device)
    inputs, outputs = PROJECTIONS[args.projection]
    group_bytes = inputs * (outputs // PACK) * 4
    rotation = args.banks or rotation_banks(group_bytes, args.experts)
    report["banks"] = rotation
    banks = build_banks(inputs, outputs, device, rotation)
    activation = build_activation(args.experts * args.rows, inputs, device)
    # Two group counts, not one. The kernel walks every group it is given and
    # reads a shard only for the non-empty ones, so a production call that
    # routes twelve rows over eighty-one experts still hands over all 256.
    # Holding the non-empty count fixed and shrinking the list prices the walk
    # itself, which the earlier `59.37 us + 0.709/expert` fit had folded whole
    # into its fixed term.
    configured = args.configured_experts or EXPERTS
    if configured < args.experts:
        raise ValueError("The list cannot be shorter than its non-empty part")
    counts = torch.zeros(configured, dtype=torch.int64, device=device)
    counts[:args.experts] = args.rows
    routing = build_routing(args.experts, args.rows, device)
    call = PATHS[args.path](banks, activation, counts, routing,
                            *([args.group_list_type]
                              if args.path == "grouped" else []))

    for _ in range(args.warmup):
        call()
    samples = sorted(batch_mean_us(call, args.calls)
                     for _ in range(args.batches))
    # A shape's time does not say which pipeline unit spent it, and the whole
    # question about this kernel is whether its scalar work follows the active
    # group count or the call. Profiling after the samples keeps the reported
    # median free of the profiler's own cost.
    if args.profile:
        capture = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(args.profile)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1))
        with capture:
            for _ in range(args.calls):
                call()
            torch.npu.synchronize()
        torch_npu.profiler.profiler.analyse(profiler_path=str(args.profile))
        report["profile"] = str(args.profile)
        report["profiled_calls"] = args.calls

    median = statistics.median(samples)
    read = args.experts * group_bytes
    report.update({
        "status": "pass", "weight_bytes": read,
        "configured_experts": configured,
        "group_list_type": args.group_list_type,
        "median_us": median, "min_us": samples[0],
        "p90_us": samples[-1 - len(samples) // 10],
        "gb_per_s": read / (median / 1e6) / 1e9,
    })
    return True


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--projection", choices=tuple(PROJECTIONS),
                        default="gate_up")
    parser.add_argument("--path", choices=tuple(PATHS), default="grouped")
    parser.add_argument("--banks", type=int,
                        help="override the rotation the shape derives")
    parser.add_argument("--experts", type=int, required=True,
                        help="groups holding rows")
    parser.add_argument("--configured-experts", type=int,
                        help="length of the group list, walked whole"
                             " by the kernel; defaults to 256")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--group-list-type", type=int, default=1,
                        choices=(0, 1),
                        help="0 passes a running sum, 1 per-group counts")
    parser.add_argument("--profile", type=Path,
                        help="write a Level1 trace of one batch")
    return parser.parse_args()


def main():
    args = arguments()
    report = {
        "status": "error", "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "hostname": platform.node(),
        "path": args.path, "projection": args.projection,
        "experts": args.experts, "rows": args.rows, "device": args.device,
        "calls": args.calls, "warmup": args.warmup, "batches": args.batches,
        "scope": "one die, one rank, shared with another tenant",
        "value_note": "arbitrary bits; timing does not depend on values",
        "bank_note": "counts select the active groups; the rotation puts "
                     f"{EVICTION_FACTOR} times L2 between two reads of one "
                     "bank, so a call streams from HBM",
    }
    try:
        passed = measure(args, report)
    except Exception as error:
        report["error"] = repr(error)[:2000]
        passed = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": args.path, "projection": args.projection,
                      "experts": args.experts,
                      "rows": args.rows, "passed": passed,
                      "median_us": report.get("median_us"),
                      "gb_per_s": report.get("gb_per_s")}))
    return int(not passed)


if __name__ == "__main__":
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    raise SystemExit(main())
