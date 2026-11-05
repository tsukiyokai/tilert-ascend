"""The tree resolution, against hand-worked cases."""
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")
from tilert.models.glm_5_2_ascend.tree_resolve import (
    accepted_path, accepted_tokens,
)


def check_path_is_a_chain():
    """Reordering the verified rows along the path must give a linear match.

    The round commits through `resolve`, whose match is linear, so the path's
    tokens and the predictions at those nodes have to line up the way a chain
    would: prediction at path[i] equals the token at path[i+1].
    """
    tokens = [100, 101, 102, 103]
    parents = [-1, 0, 0, 1]
    predictions = [101, 103, 999, 777]
    path = accepted_path(tokens, parents, predictions)
    along = [tokens[node] for node in path]
    predicted = [predictions[node] for node in path]
    for index in range(len(path) - 1):
        if predicted[index] != along[index + 1]:
            return (f"at {index}: prediction {predicted[index]} does not "
                    f"lead to {along[index + 1]}")
    return None


def main():
    failures = []

    # A tree: 0 is the root, 1 and 2 are its children, 3 hangs off 1.
    tokens = [100, 101, 102, 103]
    parents = [-1, 0, 0, 1]

    # The target predicts 101 at the root and 103 at node 1, so the path is
    # 0 -> 1 -> 3 and the commit is [101, 103, prediction at 3].
    predictions = [101, 103, 999, 777]
    path = accepted_path(tokens, parents, predictions)
    if path != [0, 1, 3]:
        failures.append(f"deepest path is {path}, wanted [0, 1, 3]")
    committed = accepted_tokens(tokens, parents, predictions)
    if committed != [101, 103, 777]:
        failures.append(f"committed {committed}, wanted [101, 103, 777]")

    # Predicting 102 at the root takes the other branch, which has no children.
    predictions = [102, 103, 555, 777]
    if accepted_path(tokens, parents, predictions) != [0, 2]:
        failures.append("the second branch was not taken")
    if accepted_tokens(tokens, parents, predictions) != [102, 555]:
        failures.append("the second branch committed the wrong tokens")

    # Predicting something no child carries accepts nothing but the free token.
    predictions = [900, 103, 555, 777]
    if accepted_path(tokens, parents, predictions) != [0]:
        failures.append("a miss at the root should accept no draft")
    if accepted_tokens(tokens, parents, predictions) != [900]:
        failures.append("a miss should still commit the prediction")

    # A deeper chain must win over a shallower branch even when both hold.
    tokens = [1, 2, 3, 4]
    parents = [-1, 0, 1, 0]
    predictions = [2, 3, 50, 60]
    if accepted_path(tokens, parents, predictions) != [0, 1, 2]:
        failures.append("the deeper path did not win")

    chain = check_path_is_a_chain()
    if chain:
        failures.append(chain)
    print("\n".join(failures) if failures
          else "tree resolution correct, and the path matches linearly")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
