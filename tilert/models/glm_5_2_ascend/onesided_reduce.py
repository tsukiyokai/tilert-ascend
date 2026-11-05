"""Replace the TP sum with a one-sided push and an ordered local reduction.

The vendor collective costs 63.5 microseconds in-graph for this round's 96 KiB
message and none of that is transfer: a four-byte reduction costs as much as a
six-megabyte one, so the figure is rendezvous. Writing straight into every
peer's symmetric window and summing locally in rank order measures 22.3
microseconds as a kernel and agrees with HCCL element for element. Over the
round's 157 reductions that is 9.97 milliseconds against 3.50.

The sum is taken in rank order, so the result is bit-reproducible — the
trajectory gate compares tokens bit for bit, and EP16 failed it precisely by
changing a summation's association.
"""

import ctypes
import os

LIBRARY_DIR = os.environ.get(
    "TILERT_SHMEM_LIB_DIR", "/workspace/src/shmem-build/shmem/build/lib",
)
HEAP_BYTES = 1024 * 1024 * 1024


# The binding's dependencies in the order the loader needs them. It records
# the kernels but not their directory, and `LD_LIBRARY_PATH` cannot carry it
# here because the arm's activate.sh resets the environment -- the same trap
# the launch script documents for ASCEND_CUSTOM_OPP_PATH. Resolving each by
# full path and globally is what makes the chain hold without it.
DEPENDENCIES = ("libshmem_utils.so", "libshmem.so", "libtp_push_kernel.so",
                "libkv_shuffle_kernel.so", "liballgather_kernel.so")


def _bind(device):
    """Bind this thread's device, so the runtime can report one."""
    import torch

    torch.npu.set_device(device)
    # `str` has an `index` method, so `hasattr(device, "index")` is true for
    # "npu:0" and hands back the method. The string form is what both a
    # `torch.device` and a plain string agree on.
    index = int(str(device).rsplit(":", 1)[-1])
    runtime = ctypes.CDLL("libascendcl.so")
    bound = runtime.aclrtSetDevice(ctypes.c_int(index))
    if bound != 0:
        raise RuntimeError(f"aclrtSetDevice({index}) returned {bound}")


def _load():
    """Resolve the binding's own libraries before torch sees the operators."""
    import torch

    for name in DEPENDENCIES:
        ctypes.CDLL(f"{LIBRARY_DIR}/{name}", mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(f"{LIBRARY_DIR}/aclshmem_torch.so")


class OneSidedReduce:
    """Own the symmetric domain for one rank and reduce FP32 tensors in it."""

    def __init__(self, rank, device, world_size, endpoint):
        import torch

        # Before the library, not after. The backend reads the device in its
        # own constructor -- `aclrtGetDevice` at shmem_init_backend.cpp:60 --
        # and that constructor runs when the binding is dlopened, so a device
        # bound afterwards is bound too late: the id stays -1, hybm_init
        # latches -1, and every later AclrtSetDevice(-1) fails with 107001,
        # surfacing as -4 from setup_heap with nothing about devices in it.
        _bind(device)
        _load()
        self.manager = torch.classes.ShmemOps.Manager()
        status = self.manager.attr_init(rank, world_size, HEAP_BYTES, endpoint)
        if status != 0:
            raise RuntimeError(f"aclshmem init returned {status}")
        self.reducer = torch.classes.ShmemOps.AllReduce()

    def sum(self, value):
        """Return the TP sum of `value`, which the caller keeps in FP32."""
        import torch

        out = torch.empty_like(value)
        self.reducer.compute(out, value)
        return out

    def close(self):
        self.manager.finalize()
        self.manager = None
        self.reducer = None


def install(engine, endpoint):
    """Point the engine's TP sum at a one-sided reduction.

    The engine already funnels every TP reduction through one override to
    force FP32, so this replaces that override's body and inherits the
    bfloat16 round trip.
    """
    import torch

    reduce = OneSidedReduce(engine.rank, engine.device, 16, endpoint)

    def one_sided(value):
        if value.dtype == torch.bfloat16:
            return reduce.sum(value.float()).bfloat16()
        return reduce.sum(value)

    engine._tp_sum_group._all_reduce_out_place = one_sided
    engine.tp_reduction = "one_sided_float32"
    return reduce
