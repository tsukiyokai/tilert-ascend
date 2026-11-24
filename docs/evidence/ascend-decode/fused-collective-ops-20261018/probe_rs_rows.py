"""Establish the row constraint on fused matmul-reduce-scatter.

Decode at batch one presents four rows to sixteen ranks, so whether the
operator demands rows divisible by the world size decides the route.
"""

import os

import torch
import torch.distributed as distributed
import torch_npu

INPUT_PER_RANK = 2048
OUTPUT = 6144


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    distributed.init_process_group("hccl", rank=rank, world_size=world)
    group = distributed.distributed_c10d._get_default_group()
    hcom = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    weight = torch.randn(INPUT_PER_RANK, OUTPUT, dtype=torch.bfloat16,
                         device=device)
    for rows in (2, 4, 5, 7, 16):
        activation = torch.randn(rows, INPUT_PER_RANK, dtype=torch.bfloat16,
                                 device=device)
        try:
            result = torch.ops.npu.npu_mm_reduce_scatter_base(
                activation, weight, hcom, world)
            torch.npu.synchronize()
            verdict = f"OK  out {tuple(result.shape)}"
        except Exception as error:
            verdict = "拒绝  " + str(error).split("\n")[0][-100:]
        if rank == 0:
            print(f"  rows={rows:3d} world={world}  {verdict}")
    distributed.destroy_process_group()


main()
