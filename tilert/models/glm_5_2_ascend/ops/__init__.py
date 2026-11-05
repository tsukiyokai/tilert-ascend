"""Ascend operators for the GLM-5.2 decode path."""

from importlib import import_module

__all__ = ["build_kernel", "rms_norm", "rms_norm_reference"]


def __getattr__(name):
    if name in __all__:
        module = import_module(".rms_norm", __name__)
        globals().update({item: getattr(module, item) for item in __all__})
        return globals()[name]
    raise AttributeError(name)
