"""Can a symmetric-memory domain coexist with vLLM's HCCL group?

The decode path brings up HCCL first and keeps it for the whole session, so
the one-sided reduction can only reach production if aclshmem_init succeeds
afterwards and both remain usable. This brings up HCCL, then the symmetric
domain, then alternates reductions between them and checks they still agree —
the order the production path will actually use.

Launch with sixteen ranks:

    torchrun --nproc-per-node=16 check_coexist.py
"""

import ctypes
import os

import torch
import torch.distributed as distributed
import torch_npu

ROWS = 4
HIDDEN = 6144
LIBRARY_DIR = "/workspace/src/shmem-build/shmem/build/lib"
KERNEL_LIBRARY = f"{LIBRARY_DIR}/libtp_push_kernel.so"
LIBRARY = f"{LIBRARY_DIR}/aclshmem_torch.so"
HEAP_BYTES = 1024 * 1024 * 1024


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(torch.device(f"npu:{rank}"))

    # HCCL first, as the model's worker does.
    distributed.init_process_group("hccl", rank=rank, world_size=world)
    group = distributed.distributed_c10d._get_default_group()
    warm = torch.ones(ROWS * HIDDEN, dtype=torch.float32, device="npu")
    distributed.all_reduce(warm, group=group)
    torch.npu.synchronize()

    ctypes.CDLL(KERNEL_LIBRARY, mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(LIBRARY)
    manager = torch.classes.ShmemOps.Manager()
    status = manager.attr_init(rank, world, HEAP_BYTES, "tcp://127.0.0.1:9411")
    if rank == 0:
        print(f"attr_init after HCCL: status {status}")
    if status != 0:
        raise RuntimeError(f"rank {rank}: attr_init returned {status}")
    reducer = torch.classes.ShmemOps.AllReduce()

    # Alternate the two reductions, since a session will interleave them: the
    # one-sided path for the TP sum, HCCL for whatever else vLLM collectives.
    expected = float(world * (world - 1) // 2)
    wrong = 0
    disagree = 0
    for step in range(8):
        partial = torch.full((ROWS, HIDDEN), float(rank), dtype=torch.float32,
                             device="npu")
        mine = torch.empty_like(partial)
        reducer.compute(mine, partial)
        theirs = partial.clone()
        distributed.all_reduce(theirs, group=group)
        torch.npu.synchronize()
        wrong += int((mine != expected).sum().item())
        disagree += int((mine != theirs).sum().item())

    if rank == 0:
        print(f"8 interleaved rounds: wrong {wrong}, disagreeing with HCCL "
              f"{disagree}")
        print(f"both domains alive: hccl_world "
              f"{distributed.get_world_size(group)}, shmem_pes "
              f"{world}")

    manager.finalize()
    distributed.destroy_process_group()


main()
