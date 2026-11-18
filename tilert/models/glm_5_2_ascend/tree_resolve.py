def accepted_path(tokens, parents, predictions):
    if not tokens:
        return []
    reachable = [False] * len(tokens)
    came_from = [-1] * len(tokens)
    reachable[0] = True
    for node in range(1, len(tokens)):
        parent = parents[node]
        if parent < 0 or not reachable[parent]:
            continue
        if predictions[parent] == tokens[node]:
            reachable[node] = True
            came_from[node] = parent
    deepest = 0
    depth = [0] * len(tokens)
    for node in range(1, len(tokens)):
        if reachable[node]:
            depth[node] = depth[parents[node]] + 1
            if depth[node] > depth[deepest]:
                deepest = node
    path = []
    walk = deepest
    while walk >= 0:
        path.append(walk)
        walk = came_from[walk] if walk else -1
    path.reverse()
    return path


def accepted_tokens(tokens, parents, predictions):
    path = accepted_path(tokens, parents, predictions)
    committed = [tokens[node] for node in path[1:]]
    committed.append(predictions[path[-1]])
    return committed
