EMPTY = -1

def prefix_room(rows, prefix_length, picks):
    room = picks - rows
    if room < prefix_length:
        raise ValueError(
            f"A tree of {rows} rows under {picks} picks leaves "
            f"{max(0, room)} for a prefix of {prefix_length}; raise "
            "index_topk or pass the indexer's own selection"
        )
    return room

def ancestry(parents):
    lines = []
    for node, parent in enumerate(parents):
        lines.append(([] if parent < 0 else lines[parent]) + [node])
    return lines

def branch_depths(parents):
    depths = []
    for node, parent in enumerate(parents):
        depths.append(0 if parent < 0 else depths[parent] + 1)
    return depths

def ancestor_rows(parents, prefix_length, picks, prefix_picks=None):
    if prefix_picks is None:
        prefix_picks = prefix_room(len(parents), prefix_length, picks)
    shared = list(range(max(0, prefix_length - prefix_picks), prefix_length))
    rows = []
    for line in ancestry(parents):
        row = shared + [prefix_length + node for node in line]
        if len(row) > picks:
            row = row[len(row) - picks:]
        rows.append(row + [EMPTY] * (picks - len(row)))
    return rows

def visible(rows, node):
    return [position for position in rows[node] if position != EMPTY]

def ancestor_bitmap(parents, prefix_length, capacity, device=None):
    import torch

    lines = ancestry(parents)
    deepest = max(len(line) for line in lines)
    bitmap = torch.ones(
        (len(parents), capacity), dtype=torch.uint8, device=device,
    )
    bitmap[:, :prefix_length] = 0
    columns = torch.tensor(
        [[prefix_length + node for node in line]
         + [prefix_length + line[-1]] * (deepest - len(line))
         for line in lines],
        dtype=torch.int64, device=device,
    )
    bitmap.scatter_(1, columns, 0)
    return bitmap.view(torch.bool)
