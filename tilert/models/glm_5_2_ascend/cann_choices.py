"""Capture the indices and expert IDs consumed by one eager CANN forward."""

from functools import wraps
import hashlib
import inspect
from pathlib import Path


_MISSING = object()
_ATTENTION_METHOD = "_execute_sparse_flash_attention_process"
_ROUTING_METHOD = "_select_experts"


class CannChoices:
    """Snapshot actual selections without recomputing either top-k.

    Pass ``CannModel.model`` and run one forward inside the context.
    ``finish()`` returns two tuples of 78 owned, detached tensors on their
    original device: indices are [M,2048], routes are [M,8], and the first
    three dense layers have route None. Integer dtype and order are kept.
    Instance methods are restored on every context exit, including errors.
    """

    def __init__(self, model, width=1):
        widths = (1, 2, 4, 6, 8, 12, 16, 24, 32)
        if type(width) is not int or width not in widths:
            raise ValueError("Unsupported choice capture width")
        backbone = model.model
        if (backbone.start_layer != 0 or backbone.end_layer != 78
                or len(backbone.layers) != 78):
            raise ValueError("Choice capture requires all 78 main layers")
        if any(layer.use_sequence_parallel_moe for layer in backbone.layers):
            raise ValueError("Choice capture requires unsharded token rows")
        self.width = width
        self._attention = tuple(
            layer.self_attn.mla_attn.mla_attn.impl
            for layer in backbone.layers
        )
        self._routers = tuple(
            layer.mlp.experts.routed_experts
            for layer in backbone.layers[3:]
        )
        self._overrides = []
        self._source_paths = {Path(__file__).resolve()}
        self._source_sha256 = None
        self._indices = []
        self._routes = [None] * 3
        self._active = False
        self._finished = False
        self._device = None

    def __enter__(self):
        if self._active:
            raise RuntimeError("A choice capture is already active")
        self._active = True
        self._finished = False
        self._indices = []
        self._routes = [None] * 3
        self._device = None
        self._source_paths = {Path(__file__).resolve()}
        self._source_sha256 = None
        try:
            for layer, target in enumerate(self._attention):
                self._install(target, _ATTENTION_METHOD, layer,
                              self._wrap_attention)
            for layer, target in enumerate(self._routers, start=3):
                self._install(target, _ROUTING_METHOD, layer,
                              self._wrap_router)
            self._source_sha256 = {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(self._source_paths)
            }
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _install(self, target, name, layer, wrap):
        original = getattr(target, name)
        sources = {Path(inspect.getfile(method)).resolve()
                   for method in (original, inspect.unwrap(original))}
        replacement = wrap(layer, original)
        saved = vars(target).get(name, _MISSING)
        setattr(target, name, replacement)
        self._overrides.append((target, name, saved))
        self._source_paths.update(sources)

    def _snapshot(self, layer, value, choices, columns, name):
        import torch

        if not self._active or self._finished:
            raise RuntimeError("Choices arrived outside an active capture")
        if layer != len(choices):
            raise ValueError(
                f"Expected {name} layer {len(choices)}, got {layer}"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Layer {layer} {name} must be a Tensor")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Layer {layer} {name} must be INT32 or INT64")
        if name == "indices" and value.shape == (self.width, 1, columns):
            value = value[:, 0, :]
        if value.shape != (self.width, columns):
            raise ValueError(
                f"Layer {layer} {name} must have shape "
                f"[{self.width},{columns}]"
            )
        if self._device is not None and value.device != self._device:
            raise ValueError(f"Layer {layer} {name} changed device")
        self._device = value.device
        return value.detach().clone()

    def _wrap_attention(self, layer, original):
        signature = inspect.signature(original)
        if "topk_indices" not in signature.parameters:
            raise ValueError("SFA method must expose topk_indices")

        @wraps(original)
        def capture(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            saved = self._snapshot(
                layer, arguments["topk_indices"], self._indices, 2048,
                "indices",
            )
            result = original(*args, **kwargs)
            self._indices.append(saved)
            return result

        return capture

    def _wrap_router(self, layer, original):
        @wraps(original)
        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("Expert selection must return (weights, ids)")
            saved = self._snapshot(layer, result[1], self._routes, 8, "routes")
            self._routes.append(saved)
            return result

        return capture

    def finish(self):
        """Require one SFA call per layer and one router call per MoE layer."""
        if not self._active or self._finished:
            raise RuntimeError("finish requires an unfinished active capture")
        if len(self._indices) != 78 or len(self._routes) != 78:
            raise ValueError(
                "Expected 78 SFA and 75 router calls, got "
                f"{len(self._indices)} and {len(self._routes) - 3}"
            )
        self._finished = True
        return tuple(self._indices), tuple(self._routes)

    def source_metadata(self):
        """Return source hashes saved at context entry, also after exit."""
        if self._source_sha256 is None:
            raise RuntimeError("Enter the capture before reading its sources")
        return dict(self._source_sha256)

    def __exit__(self, exc_type, exc_value, traceback):
        for target, name, saved in reversed(self._overrides):
            if saved is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, saved)
        self._overrides.clear()
        self._indices = []
        self._routes = [None] * 3
        self._active = False
        return False
