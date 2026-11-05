"""What a wider tree is worth on the engine's own streams, and what it costs.

The tree is the only drafter still climbing past width 32, but the capture
whitelist and `route_metadata`'s MAX_TOKENS both stop there. This prices the
climb against the round model's measured terms before anything is raised.
"""
import json
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")

from tilert.models.glm_5_2_ascend.tree_draft import TreeDrafter


def accepted_by_tree(tokens, parents, truth):
    depth_of = {}
    best = 0
    for index, (token, parent) in enumerate(zip(tokens, parents)):
        if parent == -1:
            parent_depth = 0
        elif parent in depth_of:
            parent_depth = depth_of[parent]
        else:
            continue
        if parent_depth < len(truth) and truth[parent_depth] == token:
            depth_of[index] = parent_depth + 1
            best = max(best, parent_depth + 1)
    return best


def replay(prompt, truth, budget):
    drafter = TreeDrafter(prompt, longest=32, budget=budget, branches=4)
    position = rounds = nodes = 0
    while position < len(truth):
        rounds += 1
        tokens, parents = drafter.propose_tree(budget)
        nodes += len(tokens)
        took = accepted_by_tree(tokens, parents, truth[position:]) + 1
        drafter.commit(truth[position:position + took])
        position += took
    return len(truth) / rounds, nodes / rounds


FIXED, ROW, ATTENTION = 45.88, 0.814, 0.356


def main():
    base = "/home/shanshan/work/probe"
    prompt = json.load(open(f"{base}/prompt-8k.json"))["prompt"]
    streams = json.load(open(f"{base}/real-streams.json"))
    print(f"{'budget':>6} {'accepted':>9} {'nodes':>7} {'tpot now':>9} "
          f"{'tpot shared':>12} {'tpot 11pass':>12}")
    for budget in (12, 16, 24, 32, 48, 64, 96, 128):
        total = nodes = 0.0
        count = 0
        for group in streams.values():
            for truth in group:
                accepted, used = replay(prompt, truth, budget)
                total += accepted
                nodes += used
                count += 1
        accepted = total / count
        used = nodes / count
        now = (FIXED + ROW * budget) / accepted
        shared = (FIXED + (ROW - ATTENTION) * budget) / accepted
        eleven = (15.97 + (ROW - ATTENTION) * budget) / accepted
        print(f"{budget:>6} {accepted:>9.3f} {used:>7.1f} {now:>9.2f} "
              f"{shared:>12.2f} {eleven:>12.2f}")


main()
