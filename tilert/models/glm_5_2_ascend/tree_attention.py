from .tree_mask import EMPTY, ancestry, prefix_room

def install(layers, picks):
    import torch

    held = []
    for layer in layers:
        held.append((layer, layer.skip_topk, layer.topk_indices_buffer))

    buffer = None
    ancestors = None
    cutoff = None

    def write(parents, prefix_length, device):
        nonlocal buffer, ancestors, cutoff
        lines = ancestry(parents)
        if len(parents) + prefix_length > picks:
            width = len(parents)
            if ancestors is None or ancestors.shape[0] != width:
                ancestors = torch.empty((width, 1, width), dtype=torch.int32,
                                        device=device)
                cutoff = torch.empty((1, 1, 1), dtype=torch.int32,
                                     device=device)
                for layer, _, _ in held:
                    layer._tree_ancestors = ancestors
                    layer._tree_prefix_len = cutoff
            cutoff.fill_(prefix_length)
            ancestors.copy_(torch.tensor(
                [[[prefix_length + node for node in line]
                  + [EMPTY] * (width - len(line))] for line in lines],
                dtype=torch.int32, device=device,
            ))
            return ancestors
        prefix_room(len(parents), prefix_length, picks)
        if buffer is None or buffer.shape != (len(parents), picks):
            buffer = torch.empty((len(parents), picks), dtype=torch.int32,
                                 device=device)
            for layer, _, _ in held:
                layer.skip_topk = True
                layer.topk_indices_buffer = buffer
        deepest = max(len(line) for line in lines)
        buffer[:, :prefix_length] = torch.arange(
            prefix_length, dtype=torch.int32, device=device,
        )
        buffer[:, prefix_length + deepest:] = EMPTY
        buffer[:, prefix_length:prefix_length + deepest] = torch.tensor(
            [[prefix_length + node for node in line]
             + [EMPTY] * (deepest - len(line)) for line in lines],
            dtype=torch.int32, device=device,
        )
        return buffer

    def restore():
        for layer, skip, original in held:
            layer.skip_topk = skip
            layer.topk_indices_buffer = original
            layer._tree_ancestors = None
            layer._tree_prefix_len = None

    return write, restore
