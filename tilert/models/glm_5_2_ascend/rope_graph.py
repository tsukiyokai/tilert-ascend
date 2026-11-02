"""Capture the per-round rope refresh, which runs eager outside the model's graph.

A round issues 1108 eager aclnn calls against 637 captured super kernel calls,
and 530 of the eager ones are `copy_` -- the loop in `refresh_rope` that writes
this round's cos and sin into every distinct destination a speculative builder
keeps. At 11.386 us of eager dispatch against 1.064 under replay that loop is
11.44 ms a round.

The copies are not redundant: `rope_destinations` deduplicates by `data_ptr()`
and each speculative builder holds its own RoPE storage. But every address is
fixed -- the destinations come from hoisted metadata, which is the addresses a
replay captured, and `positions` is a preallocated buffer whose contents change
while its pointer does not. Fixed addresses with changing contents is what a
graph needs.

So the refresh is captured once a width and replayed, which leaves the work
unchanged and removes its dispatch.
"""


class RopeGraph:
    """One captured refresh a width; replay before the model's own graph."""

    def __init__(self, positions, width, destinations, slots=None, slots_i64=None):
        import torch

        from .cann_model import refresh_rope

        self.width = width
        self.replays = 0
        self.handles_slots = slots is not None and slots_i64 is not None
        # Warm the operators before capture: a first call allocates workspace
        # and that allocation cannot happen inside a capture.
        refresh_rope(positions, width, destinations)
        if self.handles_slots:
            slots_i64.copy_(slots, non_blocking=True)
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            refresh_rope(positions, width, destinations)
            if self.handles_slots:
                slots_i64.copy_(slots, non_blocking=True)
        torch.npu.synchronize()

    def refresh(self):
        """Rewrite this round's rope from the current `positions` contents."""
        self.graph.replay()
        self.replays += 1

    def report(self):
        return {"width": self.width, "replays": self.replays}
