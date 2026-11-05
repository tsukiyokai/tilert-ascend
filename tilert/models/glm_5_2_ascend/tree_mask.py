"""Turn a draft tree into the per-row index set sparse attention already takes.

The attention operator the engine calls selects, for each query row, the key
positions that row may see -- `layout_query="TND"` has no batch axis, so a
token is a row, `sparse_indices` is indexed by that row, and
`sparse_block_size=1` makes a block one token. The engine uses this to give
each token its own top-k blocks. A tree mask is the same mechanism with a
different index set: a draft token sees the committed prefix and its own
ancestors, and nothing else in the tree.

`-1` is the operator's own sentinel for an unused slot, which `sfa_v1.py:175`
relies on when it sorts the selections.
"""
EMPTY = -1


def prefix_room(rows, prefix_length, picks):
    """How much prefix each row carries, refusing to carry less than all.

    Keeping only the prefix's recent end is not a weaker mask but a different
    model: the indexer picks by relevance, and silently swapping that for
    recency changed the target's own predictions in three arms before this
    refused.
    """
    room = picks - rows
    if room < prefix_length:
        raise ValueError(
            f"A tree of {rows} rows under {picks} picks leaves "
            f"{max(0, room)} for a prefix of {prefix_length}; raise "
            "index_topk or pass the indexer's own selection"
        )
    return room


def ancestry(parents):
    """Each node's own line of descent, root first, as tree row indices.

    Parents precede children, so a node's line is its parent's line and
    itself -- one pass, and no node's chain is walked twice.
    """
    lines = []
    for node, parent in enumerate(parents):
        lines.append(([] if parent < 0 else lines[parent]) + [node])
    return lines


def branch_depths(parents):
    """How far each node stands from the committed prefix.

    A chain's row index is its depth, so nothing needed saying until now; in a
    tree the two part company, and it is the depth that rope must see.
    Parents precede children, so one pass suffices.
    """
    depths = []
    for node, parent in enumerate(parents):
        depths.append(0 if parent < 0 else depths[parent] + 1)
    return depths


def ancestor_rows(parents, prefix_length, picks, prefix_picks=None):
    """One row of key positions a draft token may attend to.

    `parents[i]` is the index within the tree of node `i`'s parent, or -1 for
    a root. The tree's nodes occupy key positions `prefix_length + i`, so a
    node's visible set is the prefix plus the positions of its ancestors plus
    its own.

    `prefix_picks` caps how much of the committed prefix each row carries,
    which is what makes the row fit `picks`: the prefix is shared by every row
    and the engine already selects it sparsely, so the caller passes the
    selection it would have made anyway. With none supplied the whole prefix
    has to fit -- see `prefix_room`.

    This is the rule in one place and in plain Python, which is what the
    checks read; the engine's own writer puts the same rule in the buffer
    without building a row at a time.
    """
    if prefix_picks is None:
        prefix_picks = prefix_room(len(parents), prefix_length, picks)
    shared = list(range(max(0, prefix_length - prefix_picks), prefix_length))
    rows = []
    for line in ancestry(parents):
        row = shared + [prefix_length + node for node in line]
        if len(row) > picks:
            # The ancestors are what correctness needs; the prefix is already
            # a selection, so it is the prefix that gives way.
            row = row[len(row) - picks:]
        rows.append(row + [EMPTY] * (picks - len(row)))
    return rows


def visible(rows, node):
    """The positions a row actually selects, for checking a construction."""
    return [position for position in rows[node] if position != EMPTY]

def ancestor_bitmap(parents, prefix_length, capacity, device=None):
    """The tree's visibility as the mask `shared_kv_fia` already takes.

    `cann_attention.selection_mask` states the convention -- True excludes a
    cache row -- and builds its mask by scattering zeros at the selected
    positions. This does the same with the tree's rule: a node sees the
    committed prefix and its own ancestors, so those positions are zero and
    everything else is one.

    This is the shorter of the two paths into the engine. The other feeds
    `sparse_indices` to the operator the engine calls, which rejected seven
    argument sets from a probe; `shared_kv_fia` passes a mask of shape
    (1, 1, width, capacity) and is known to run.
    """
    import torch

    lines = ancestry(parents)
    deepest = max(len(line) for line in lines)
    bitmap = torch.ones(
        (len(parents), capacity), dtype=torch.uint8, device=device,
    )
    bitmap[:, :prefix_length] = 0
    # One scatter rather than a write for every edge. A short line repeats its
    # own last column to fill the row, which zeroes an already zero slot.
    columns = torch.tensor(
        [[prefix_length + node for node in line]
         + [prefix_length + line[-1]] * (deepest - len(line))
         for line in lines],
        dtype=torch.int64, device=device,
    )
    bitmap.scatter_(1, columns, 0)
    return bitmap.view(torch.bool)
