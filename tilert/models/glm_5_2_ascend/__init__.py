"""GLM-5.2 Ascend decoder and its independent CPU reference path."""

from .config import DecodeConfig

__all__ = ["DecodeConfig", "DecodeEngine", "NativeDecodeEngine", "StepResult"]


def __getattr__(name):
    if name == "NativeDecodeEngine":
        from .native_engine import NativeDecodeEngine

        globals()[name] = NativeDecodeEngine
        return NativeDecodeEngine
    if name in ("DecodeEngine", "StepResult"):
        from .engine import DecodeEngine, StepResult

        globals().update(DecodeEngine=DecodeEngine, StepResult=StepResult)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
