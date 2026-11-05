"""Diagnostic hidden-state snapshots for one eager CANN forward."""

from functools import partial


class CannTrace:
    """Capture embedding, 77 merged residuals, and final normalized hidden.

    Pass the causal LM held by ``CannModel.model``. The next layer's input
    norm exposes the preceding layer's actual fused residual. Hooks only
    copy those tensors; they do not reconstruct a residual or project logits.
    """

    def __init__(self, model, width=1):
        if type(width) is not int or width not in (1, 2, 4):
            raise ValueError("Trace width must be 1, 2 or 4")
        backbone = model.model
        if (backbone.start_layer != 0 or backbone.end_layer != 78
                or len(backbone.layers) != 78):
            raise ValueError("Tracing requires all 78 layers on each TP rank")
        if any(layer.use_sequence_parallel_moe for layer in backbone.layers):
            raise ValueError("Tracing requires unsharded token rows")
        self.hidden_size = backbone.hidden_size
        self.width = width
        self.norms = tuple(layer.input_layernorm for layer in backbone.layers)
        self._handles = []
        self._hidden = []
        self._active = False
        self._finished = False

    def __enter__(self):
        if self._active:
            raise RuntimeError("A trace context is already active")
        self._active = True
        self._finished = False
        self._hidden = []
        try:
            self._handles.append(self.norms[0].register_forward_pre_hook(
                self._embedding,
            ))
            for layer, norm in enumerate(self.norms[1:], start=1):
                self._handles.append(norm.register_forward_hook(
                    partial(self._residual, layer),
                ))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _snapshot(self, index, value):
        import torch

        if not self._active or self._finished:
            raise RuntimeError("Hidden state arrived outside an active trace")
        if index != len(self._hidden):
            raise ValueError(
                f"Expected hidden {len(self._hidden)}, got {index}"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Hidden {index} must be a Tensor")
        if value.shape != (self.width, self.hidden_size):
            raise ValueError(
                f"Hidden {index} must have shape [{self.width}, hidden_size]"
            )
        if value.dtype != torch.bfloat16:
            raise ValueError(f"Hidden {index} must be BF16")
        if self._hidden and value.device != self._hidden[0].device:
            raise ValueError(f"Hidden {index} changed device")
        self._hidden.append(value.detach().clone())

    def _embedding(self, module, inputs):
        if len(inputs) != 1:
            raise ValueError(
                "The first input norm must receive only embedding"
            )
        self._snapshot(0, inputs[0])

    def _residual(self, layer, module, inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError(
                f"Layer {layer} input norm must return (x, residual)"
            )
        self._snapshot(layer, output[1])

    def finish(self, step):
        """Return 79 owned snapshots after the single forward completes."""
        if not self._active or self._finished:
            raise RuntimeError("finish requires an unfinished active trace")
        if len(self._hidden) != 78:
            raise ValueError(
                f"Expected 78 hook snapshots, got {len(self._hidden)}"
            )
        self._snapshot(78, step.last_hidden)
        self._finished = True
        return tuple(self._hidden)

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._hidden = []
        self._active = False
        return False
