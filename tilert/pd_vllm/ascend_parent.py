"""Capture one target final-norm row before the MTP parent rotation."""


class ParentHiddenCapture:
    """Own stable storage across requests and arm one target forward hook."""

    def __init__(self, device):
        import torch

        self.tensor = torch.empty((1, 6144), dtype=torch.bfloat16,
                                  device=device)
        self._handle = None
        self._identity = None
        self._captured = False

    def bind(self, model, request):
        if not request.rid or request.num_tokens < 1:
            raise ValueError("Parent hidden requires one complete prompt")
        self.clear()
        self._identity = (request.rid, request.num_tokens)
        self._handle = model.register_forward_hook(
            self._capture, with_kwargs=True,
        )

    def _capture(self, model, arguments, kwargs, output):
        import torch

        if self._captured:
            raise RuntimeError("Target forward ran twice for one parent")
        _, length = self._identity
        for name in ("input_ids", "positions"):
            value = kwargs.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != (length,):
                raise ValueError(f"Parent hidden requires {name}[P]")
        if (
            not isinstance(output, torch.Tensor)
            or output.shape != (length, 6144)
            or output.dtype != torch.bfloat16
            or output.device != self.tensor.device
        ):
            raise ValueError("Target output must be BF16[P,6144] on its die")
        self.tensor.copy_(output[length - 1:length].detach())
        self._captured = True

    def pointer_for(self, request):
        if (
            not self._captured
            or self._identity != (request.rid, request.num_tokens)
        ):
            raise RuntimeError("Parent hidden was not captured for this RID/P")
        return self.tensor.data_ptr()

    def clear(self):
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        self._identity = None
        self._captured = False
