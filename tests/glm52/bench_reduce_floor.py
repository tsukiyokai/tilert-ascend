"""What does one TP16 reduction of a decode-sized message actually cost?

The round spends 9.974 ms on 157 reductions, 63.5 microseconds each for a
48 KiB message across sixteen ranks. That rate is 11 GB/s, two orders off any
link limit, so the cost is rounds and rendezvous rather than transfer, and a
one-shot push should reach single-digit microseconds. This measures the
baseline the push has to beat, under both expansion modes and at the exact
shape the o_proj and MoE reductions carry, so the later comparison is like
for like.

Launch with sixteen ranks:

    torchrun --nproc-per-node=16 bench_reduce_floor.py --report out.json
"""

import argparse
import json
import os

import torch
import torch.distributed as distributed
import torch_npu

ROWS = 4
HIDDEN = 6144


def timed(call, iterations):
    for _ in range(20):
        call()
    torch.npu.synchronize()
    start, end = torch.npu.Event(True), torch.npu.Event(True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def chained(call, length, iterations):
    """Marginal cost per reduction, so the launch floor is amortised."""
    single = timed(call, iterations)

    def chain():
        for _ in range(length):
            call()

    return single, (timed(chain, iterations) - single) / (length - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--chain", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--report")
    arguments = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl", rank=rank, world_size=world)
    group = distributed.distributed_c10d._get_default_group()

    message = torch.randn(arguments.rows, HIDDEN, dtype=torch.float32,
                          device=device)
    shard = torch.empty(arguments.rows * HIDDEN // world, dtype=torch.float32,
                        device=device)
    flat = message.view(-1)

    cases = {}
    cases["all_reduce"] = chained(
        lambda: distributed.all_reduce(message, group=group),
        arguments.chain, arguments.iterations)
    cases["reduce_scatter"] = chained(
        lambda: distributed.reduce_scatter_tensor(shard, flat, group=group),
        arguments.chain, arguments.iterations)
    cases["all_gather"] = chained(
        lambda: distributed.all_gather_into_tensor(flat, shard, group=group),
        arguments.chain, arguments.iterations)
    cases["barrier"] = chained(
        lambda: distributed.barrier(group=group),
        arguments.chain, arguments.iterations)

    gathered = [None] * world
    distributed.all_gather_object(gathered, {"rank": rank, "cases": cases},
                                  group=group)
    if rank == 0:
        bytes_moved = arguments.rows * HIDDEN * 4
        mode = os.environ.get("HCCL_OP_EXPANSION_MODE", "unset")
        print(f"world {world}, message {bytes_moved / 1024:.0f} KiB float32, "
              f"rows {arguments.rows}, expansion {mode}")
        print(f"{'collective':>16}{'single µs':>11}{'marginal µs':>13}"
              f"{'GB/s at marginal':>18}")
        for name, (single, marginal) in cases.items():
            rate = (bytes_moved / marginal / 1000.0 if marginal > 0 else 0.0)
            print(f"{name:>16}{single:11.2f}{marginal:13.2f}"
                  f"{rate:18.1f}")
        slowest = max(
            (max(item["cases"]["all_reduce"][1] for item in gathered),
             min(item["cases"]["all_reduce"][1] for item in gathered)))
        fastest = min(item["cases"]["all_reduce"][1] for item in gathered)
        print(f"  all_reduce marginal across ranks: "
              f"{fastest:.2f} to {slowest:.2f} µs")
        if arguments.report:
            with open(arguments.report, "w") as handle:
                json.dump({"world": world, "rows": arguments.rows,
                           "message_bytes": bytes_moved, "expansion": mode,
                           "ranks": gathered}, handle, indent=2)
                handle.write("\n")
    distributed.destroy_process_group()


main()
