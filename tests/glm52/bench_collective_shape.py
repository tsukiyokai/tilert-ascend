"""Where do a TP16 reduction's 126 microseconds go?

A reduction of four bytes costs as much as one of six mebibytes, so the figure
is rendezvous. Two measurements separate the candidate causes. Cost against
world size distinguishes an algorithm with O(N) steps from one whose single
rendezvous is simply expensive. Point-to-point latency from rank zero to every
other rank gives the per-hop cost and exposes the topology, since A3 pairs two
dies per module and a ring over sixteen dies crosses module boundaries.

Launch with sixteen ranks:

    torchrun --nproc-per-node=16 bench_collective_shape.py --report out.json
"""

import argparse
import json
import os

import torch
import torch.distributed as distributed
import torch_npu

ELEMENTS = 16


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


def subgroup_scaling(rank, world, device, iterations):
    """Every rank creates every subgroup; only members time theirs."""
    results = {}
    for size in (2, 4, 8, 16):
        if size > world:
            continue
        members = list(range(size))
        group = distributed.new_group(ranks=members, backend="hccl")
        if rank in members:
            buffer = torch.randn(ELEMENTS, dtype=torch.float32, device=device)
            distributed.all_reduce(buffer, group=group)
            results[size] = timed(
                lambda: distributed.all_reduce(buffer, group=group),
                iterations)
        distributed.barrier()
    return results


def pairwise_latency(rank, world, device, iterations):
    """Round-trip between rank zero and each peer, one pair at a time."""
    results = {}
    for peer in range(1, world):
        group = distributed.new_group(ranks=[0, peer], backend="hccl")
        if rank in (0, peer):
            buffer = torch.randn(ELEMENTS, dtype=torch.float32, device=device)
            distributed.all_reduce(buffer, group=group)
            micro = timed(
                lambda: distributed.all_reduce(buffer, group=group),
                iterations)
            if rank == 0:
                results[peer] = micro
        distributed.barrier()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--report")
    arguments = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl", rank=rank, world_size=world)

    scaling = subgroup_scaling(rank, world, device, arguments.iterations)
    pairs = pairwise_latency(rank, world, device, arguments.iterations)

    if rank == 0:
        mode = os.environ.get("HCCL_OP_EXPANSION_MODE", "unset")
        print(f"float32 all_reduce of {ELEMENTS} elements "
              f"({ELEMENTS * 4} B), expansion {mode}")
        print(f"\n{'ranks':>7}{'µs':>9}{'µs/rank':>10}{'vs 2 ranks':>12}")
        base = scaling.get(2)
        for size, micro in sorted(scaling.items()):
            print(f"{size:7d}{micro:9.2f}{micro / size:10.2f}"
                  f"{micro / base:11.2f}x")
        print(f"\n{'peer':>6}{'pair µs':>10}  (rank 0 to peer)")
        for peer, micro in sorted(pairs.items()):
            print(f"{peer:6d}{micro:10.2f}")
        if pairs:
            fastest = min(pairs.values())
            slowest = max(pairs.values())
            print(f"  pair latency {fastest:.2f} to {slowest:.2f} µs, "
                  f"ratio {slowest / fastest:.2f}")
        if arguments.report:
            with open(arguments.report, "w") as handle:
                json.dump({"world": world, "expansion": mode,
                           "elements": ELEMENTS, "scaling": scaling,
                           "pairs": pairs}, handle, indent=2)
                handle.write("\n")
    distributed.destroy_process_group()


main()
