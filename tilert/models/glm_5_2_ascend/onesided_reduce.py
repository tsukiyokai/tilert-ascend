import ctypes
import os

LIBRARY_DIR = os.environ.get(
    "TILERT_SHMEM_LIB_DIR",
    "/assets/shmem-lib/lib"
    if os.path.isdir("/assets/shmem-lib/lib")
    else "/workspace/src/shmem-build/shmem/build/lib",
)
HEAP_BYTES = 1024 * 1024 * 1024

DEPENDENCIES = ("libshmem_utils.so", "libshmem.so", "libtp_push_kernel.so",
                "libkv_shuffle_kernel.so", "liballgather_kernel.so")

def _bind(device):
    import torch

    torch.npu.set_device(device)
    index = int(str(device).rsplit(":", 1)[-1])
    runtime = ctypes.CDLL("libascendcl.so")
    bound = runtime.aclrtSetDevice(ctypes.c_int(index))
    if bound != 0:
        raise RuntimeError(f"aclrtSetDevice({index}) returned {bound}")

def _load():
    import torch

    for name in DEPENDENCIES:
        ctypes.CDLL(f"{LIBRARY_DIR}/{name}", mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(f"{LIBRARY_DIR}/aclshmem_torch.so")
    bf16_lib = f"{LIBRARY_DIR}/tilert_shmem_bf16.so"
    if os.path.exists(bf16_lib):
        torch.ops.load_library(bf16_lib)

class OneSidedReduce:

    def __init__(self, rank, device, world_size, endpoint):
        import torch

        _bind(device)
        _load()
        self.manager = torch.classes.ShmemOps.Manager()
        status = self.manager.attr_init(rank, world_size, HEAP_BYTES, endpoint)
        if status != 0:
            raise RuntimeError(f"aclshmem init returned {status}")
        self.has_bf16_op = hasattr(
            getattr(torch.ops, "tilert", None), "onesided_all_reduce"
        )
        self.has_bf16_out_op = hasattr(
            getattr(torch.ops, "tilert", None), "onesided_all_reduce_out"
        )
        self.target_out = None
        if self.has_bf16_op:
            self.reducer = None
            probe = torch.zeros((1, 6144), dtype=torch.bfloat16, device=device)
            _ = torch.ops.tilert.onesided_all_reduce(probe)
            torch.npu.synchronize()
        else:
            self.reducer = torch.classes.ShmemOps.AllReduce()

    def sum(self, value):
        import torch

        out = (
            getattr(self, "target_out", None)
            if hasattr(self, "target_out")
            else None
        )
        if hasattr(self, "target_out"):
            self.target_out = None
        if (
            out is not None
            and self.has_bf16_out_op
            and value.dtype in (torch.bfloat16, torch.float32)
            and out.dtype == value.dtype
            and out.numel() == value.numel()
            and out.is_contiguous()
        ):
            return torch.ops.tilert.onesided_all_reduce_out(value, out)
        if self.has_bf16_op and value.dtype in (torch.bfloat16, torch.float32):
            return torch.ops.tilert.onesided_all_reduce(value)
        out = torch.empty_like(value)
        self.reducer.compute(out, value)
        return out

    def close(self):
        self.manager.finalize()
        self.manager = None
        self.reducer = None

def install(engine, endpoint):
    import torch

    world_size = getattr(engine, "tp_size", 16)
    reduce = OneSidedReduce(engine.rank, engine.device, world_size, endpoint)

    def one_sided(value):
        if reduce.has_bf16_op and value.dtype == torch.bfloat16:
            return reduce.sum(value)
        if value.dtype == torch.bfloat16:
            return reduce.sum(value.float()).bfloat16()
        return reduce.sum(value)

    engine._tp_sum_group._all_reduce_out_place = one_sided
    engine._tp_sum_group._tilert_onesided = bool(reduce.has_bf16_op)
    engine._tp_sum_group._tilert_onesided_reduce = reduce
    engine.tp_reduction = "one_sided_float32"
    return reduce
