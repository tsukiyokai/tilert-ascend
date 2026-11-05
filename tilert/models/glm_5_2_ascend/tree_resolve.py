"""Pick the accepted path through a verified draft tree.

A linear draft is resolved by walking the proposal until a prediction differs.
A tree is resolved by walking the deepest path whose every edge was predicted:
node `i` is reachable when its parent was accepted and the target's prediction
at that parent equals node `i`'s token. The round commits that path's tokens
plus the prediction at its last node, which is a real token whatever the draft
did.

The tree's nodes occupy verification rows in the order the drafter emitted
them, and `parents[i] < i` for every node, so one forward pass over the nodes
settles reachability without recursion.
"""


def accepted_path(tokens, parents, predictions):
    """The node indices of the accepted path, root first.

    `predictions[i]` is what the target produced at node `i`, so an edge from
    `i` to a child `c` holds when `predictions[i] == tokens[c]`. The root's
    own token is the pending token the round was entered with, which the
    target has already committed, so the root is always accepted.
    """
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
    """What the round commits: the path's tokens after the root, then one more.

    The last element is the target's prediction at the path's deepest node,
    which is correct whether or not any draft token was accepted -- the same
    property the linear path relies on.
    """
    path = accepted_path(tokens, parents, predictions)
    committed = [tokens[node] for node in path[1:]]
    committed.append(predictions[path[-1]])
    return committed
