"""Persistent typed device inputs backed by one asynchronous host packet.

Each width owns a pinned source ring. The caller must keep the number of
unfinished stages per width within the ring size before reusing a slot.
"""


def _layout(width):
    positions = (4 * width + 7) // 8 * 8
    slots = positions + 8 * width
    length = (slots + 4 * width + 7) // 8 * 8
    return (0, positions, slots, length), length + 8


def _views(packet, width):
    import torch

    offsets, _ = _layout(width)
    return tuple(
        packet[offset : offset + count * size].view(dtype)
        for offset, count, size, dtype in zip(
            offsets,
            (width, width, width, 1),
            (4, 8, 4, 4),
            (torch.int32, torch.int64, torch.int32, torch.int32),
            strict=True,
        )
    )


class HostInputs:
    """Keep packet addresses fixed; stage CPU-only token tuples in one copy."""

    def __init__(self, device, widths, slots=64, pin_memory=True):
        import torch

        self._packets = {}
        self._device_views = {}
        self._host_rows = {}
        self._host_views = {}
        self._next = {}
        for width in widths:
            _, size = _layout(width)
            packet = torch.empty(size, dtype=torch.uint8, device=device)
            host = torch.zeros(
                slots, size, dtype=torch.uint8, device="cpu", pin_memory=pin_memory
            )
            rows = tuple(host.unbind())
            self._packets[width] = packet
            self._device_views[width] = _views(packet, width)
            self._host_rows[width] = rows
            self._host_views[width] = tuple(_views(row, width) for row in rows)
            self._next[width] = 0

    def views(self, width):
        return self._device_views[width]

    def stage(self, tokens, position, depths=None):
        import torch

        width = len(tokens)
        index = self._next[width]
        self._next[width] = (index + 1) % len(self._host_rows[width])
        input_ids, positions, slots, length = self._host_views[width][index]
        input_ids.copy_(torch.tensor(tokens, dtype=torch.int32, device="cpu"))
        torch.arange(
            position, position + width, dtype=torch.int32, device="cpu", out=slots
        )
        if depths is None:
            positions.copy_(slots)
        else:
            positions.copy_(torch.tensor(depths, dtype=torch.int64, device="cpu"))
            positions.add_(position)
        length.fill_(position + width)
        self._packets[width].copy_(self._host_rows[width][index], non_blocking=True)
        return length, positions
