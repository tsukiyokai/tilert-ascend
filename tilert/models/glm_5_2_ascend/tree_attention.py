"""Feed the draft tree's per-row index set to the attention that already takes one.

The sparse attention the engine calls selects, for each query row, the key
positions that row may see: `layout_query="TND"` has no batch axis, so a token
is a row, `sparse_indices` is indexed by that row and `sparse_block_size=1`
makes a block one token. The engine fills that selection from an indexer; with
`skip_topk` it takes the selection from `topk_indices_buffer` instead.

A tree mask is the same selection with a different rule -- the committed prefix
plus a node's own ancestors -- so the wiring is to own that buffer and write
the tree into it each round. `tree_mask.ancestor_rows` builds the rows and
`-1` is the operator's own sentinel for an unused slot.
"""
from .tree_mask import EMPTY, ancestry, prefix_room


def install(layers, picks):
    """Own each layer's index buffer; return how to fill it and how to undo.

    `layers` are the attention objects. `picks` is the operator's per-row
    selection width, which the checkpoint's `index_topk` sets.
    """
    import torch

    # Assign the property, not `_skip_topk` behind it: its setter also
    # recomputes `skip_indexer_pre_process`, and `runtime_has_indexer` reads
    # that, so writing the field leaves the indexer running under a mask that
    # was supposed to replace it.
    held = []
    for layer in layers:
        held.append((layer, layer.skip_topk, layer.topk_indices_buffer))

    buffer = None

    def write(parents, prefix_length, device):
        """Put this round's tree in the buffer every layer reads.

        Every row carries the same prefix and differs only in its own line of
        descent, so the prefix is written on the device and only the lines --
        at most a row's depth each -- are carried over from the host. Building
        a row at a time instead meant a quarter of a million Python integers
        and a megabyte of transfer every round.
        """
        nonlocal buffer
        prefix_room(len(parents), prefix_length, picks)
        if buffer is None or buffer.shape != (len(parents), picks):
            buffer = torch.empty((len(parents), picks), dtype=torch.int32,
                                 device=device)
            for layer, _, _ in held:
                layer.skip_topk = True
                layer.topk_indices_buffer = buffer
        lines = ancestry(parents)
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

    return write, restore
