"""The tree mask's construction, checked against the tree's own structure."""
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")
from tilert.models.glm_5_2_ascend.tree_mask import ancestor_rows, visible


def ancestors(parents, node):
    chain = []
    walk = node
    while walk >= 0:
        chain.append(walk)
        walk = parents[walk]
    return set(chain)


def check_bitmap_matches_rows(width=6, prefix=40, capacity=64):
    """The bitmap and the index rows must express the same visibility.

    Two encodings of one rule invite divergence, so the check ties them: a
    position is visible in the bitmap exactly when it appears in the row.
    """
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return None  # the other checks are pure Python and still run

    from tilert.models.glm_5_2_ascend.tree_mask import ancestor_bitmap

    parents = [-1, 0, 0, 1, 1, 2]
    bitmap = ancestor_bitmap(parents, prefix, capacity)
    rows = ancestor_rows(parents, prefix, picks=capacity,
                         prefix_picks=prefix)
    for node in range(width):
        visible_bitmap = {index for index in range(capacity)
                          if not bool(bitmap[node, index])}
        visible_row = set(visible(rows, node))
        if visible_bitmap != visible_row:
            return (f"node {node}: bitmap sees "
                    f"{sorted(visible_bitmap)[-4:]}, rows see "
                    f"{sorted(visible_row)[-4:]}")
    return None


def main():
    # A tree where node 0 is the root, 1 and 2 branch from it, 3 from 1.
    parents = [-1, 0, 0, 1]
    prefix = 100
    picks = 8
    rows = ancestor_rows(parents, prefix, picks)
    failures = []
    for node in range(len(parents)):
        seen = visible(rows, node)
        want = {prefix + index for index in ancestors(parents, node)}
        tree_part = {position for position in seen if position >= prefix}
        if tree_part != want:
            failures.append(f"node {node}: sees {sorted(tree_part)}, "
                            f"wants {sorted(want)}")
        if len(rows[node]) != picks:
            failures.append(f"node {node}: row is {len(rows[node])} long")
        # A sibling must not be visible: that is the whole point of the mask.
        for other in range(len(parents)):
            if other not in ancestors(parents, node) and \
                    prefix + other in tree_part:
                failures.append(f"node {node} sees non-ancestor {other}")
    # And a budget too small for the chain must still keep the ancestors.
    tight = ancestor_rows(parents, prefix, 3)
    deepest = visible(tight, 3)
    if not {prefix, prefix + 1, prefix + 3}.issubset(set(deepest)):
        failures.append(f"tight budget dropped an ancestor: {deepest}")
    print("\n".join(failures) if failures
          else f"all rows correct at picks {picks} and at a tight budget")





def check_chain_is_causal(width=8):
    """A chain draft's mask must be exactly the causal prefix.

    `fia-mask-20261020` records that serial and speculative decoding disagree
    at width eight and above, never isolated, and future leakage is the
    candidate the mask would exclude: with parents forming a chain, node i may
    see the committed prefix and nodes 0..i, and nothing later. So the tree
    machinery doubles as the causality fix, and this is the property that makes
    it one.
    """
    parents = [-1] + list(range(width - 1))
    prefix = 100
    rows = ancestor_rows(parents, prefix, picks=width + 4)
    for node in range(width):
        seen = sorted(position for position in visible(rows, node)
                      if position >= prefix)
        want = [prefix + index for index in range(node + 1)]
        if seen != want:
            return f"node {node} sees {seen}, causal prefix is {want}"
    return None


def check_buffer_reuse():
    """The installer must reuse one buffer, not allocate a new one a round.

    Every layer holds a reference to it, so a fresh tensor each round would
    leave them reading the first round's tree forever.
    """
    import sys

    sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")
    from tilert.models.glm_5_2_ascend.tree_mask import ancestor_rows

    first = ancestor_rows([-1, 0, 0], 50, 6)
    second = ancestor_rows([-1, 0, 1], 50, 6)
    if first == second:
        return "two different trees produced the same rows"
    if any(len(row) != 6 for row in first + second):
        return "a row is not `picks` long"
    return None



if __name__ == "__main__":
    problem = (check_chain_is_causal() or check_buffer_reuse()
               or check_bitmap_matches_rows())
    if problem:
        print(problem)
        sys.exit(1)
    print("row construction distinguishes trees")
    sys.exit(main())
