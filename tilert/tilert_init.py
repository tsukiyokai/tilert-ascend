"""Runtime warm-up entry points."""

__all__ = ["tilert_init", "tilert_force_init"]


def _has_op(name: str) -> bool:
    import torch

    return hasattr(torch.ops.tilert, name)


def tilert_init() -> None:
    import torch

    from . import _loaded_backend

    if _loaded_backend == "glm5_2_ascend":
        torch.zeros(1, device=f"npu:{torch.npu.current_device()}")
        torch.npu.synchronize()
        return
    if _has_op("tilert_init_op"):
        torch.ops.tilert.tilert_init_op()
        return
    torch.zeros(1, device=f"cuda:{torch.cuda.current_device()}")
    torch.cuda.synchronize()


def tilert_force_init() -> None:
    import torch

    if _has_op("tilert_force_init_op"):
        torch.ops.tilert.tilert_force_init_op()
        return
    tilert_init()
