class RopeGraph:

    def __init__(self, positions, width, destinations, slots=None, slots_i64=None):
        import torch

        from .cann_model import refresh_rope

        self.width = width
        self.replays = 0
        self.handles_slots = slots is not None and slots_i64 is not None
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
        self.graph.replay()
        self.replays += 1

    def report(self):
        return {"width": self.width, "replays": self.replays}
