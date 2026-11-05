"""Verify the one-sided all-reduce through its torch operator.

The decode path funnels every TP reduction through one override, so the
operator has to be callable on torch tensors and has to agree with HCCL
numerically before it can replace it. Each rank contributes its own index, so
every element of the sum must equal pe_size*(pe_size-1)/2 and a wrong
summation order cannot pass.

Launch with sixteen ranks:

    torchrun --nproc-per-node=16 check_torch_allreduce.py
"""

import ctypes
import os

import torch
import torch.distributed as distributed
import torch_npu

ROWS = 4
HIDDEN = 6144
LIBRARY_DIR = "/workspace/src/shmem-build/shmem/build/lib"
# The binding references the kernel's symbol without recording the kernel
# library as a dependency, so the kernel has to be resolved globally first.
KERNEL_LIBRARY = f"{LIBRARY_DIR}/libtp_push_kernel.so"
LIBRARY = f"{LIBRARY_DIR}/aclshmem_torch.so"
HEAP_BYTES = 1024 * 1024 * 1024


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


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(torch.device(f"npu:{rank}"))
    distributed.init_process_group("hccl", rank=rank, world_size=world)
    group = distributed.distributed_c10d._get_default_group()

    ctypes.CDLL(KERNEL_LIBRARY, mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(LIBRARY)
    manager = torch.classes.ShmemOps.Manager()
    status = manager.attr_init(rank, world, HEAP_BYTES, "tcp://127.0.0.1:9377")
    if status != 0:
        raise RuntimeError(f"rank {rank}: attr_init returned {status}")
    reducer = torch.classes.ShmemOps.AllReduce()

    partial = torch.full((ROWS, HIDDEN), float(rank), dtype=torch.float32,
                         device="npu")
    mine = torch.empty_like(partial)
    theirs = partial.clone()

    reducer.compute(mine, partial)
    distributed.all_reduce(theirs, group=group)
    torch.npu.synchronize()

    expected = float(world * (world - 1) // 2)
    wrong = int((mine != expected).sum().item())
    against_hccl = int((mine != theirs).sum().item())

    one_sided = timed(lambda: reducer.compute(mine, partial), 200)
    vendor = timed(lambda: distributed.all_reduce(theirs, group=group), 200)

    if rank == 0:
        print(f"world {world}  shape ({ROWS}, {HIDDEN}) float32  "
              f"{ROWS * HIDDEN * 4 / 1024:.0f} KiB")
        print(f"one-sided  {one_sided:7.3f} us   wrong {wrong} of "
              f"{mine.numel()}  expected {expected}  got {mine[0][0].item()}")
        print(f"HCCL       {vendor:7.3f} us   differs from one-sided in "
              f"{against_hccl} elements")
        print(f"speedup    {vendor / one_sided:.2f}x")

    manager.finalize()
    distributed.destroy_process_group()


main()
