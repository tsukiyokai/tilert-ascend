"""Is the tree's branch cap the only thing between it and the optimum?

`TreeDrafter.propose` already pops by path probability, which is optimal for
expected accepted prefix under a node budget because the objective is additive
over nodes and a prefix's probability dominates its extension's. But
`_continuations` truncates each state to `branches` likeliest tokens, so a
node's fifth child can never enter the frontier however high its path
probability. This sweeps the cap on the engine's own streams.
"""
import json
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")

from tilert.models.glm_5_2_ascend.automaton_draft import AutomatonDrafter
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


def replay_tree(drafter, truth, budget):
    position = rounds = nodes = 0
    while position < len(truth):
        rounds += 1
        tokens, parents = drafter.propose_tree(budget)
        nodes += len(tokens)
        took = accepted_by_tree(tokens, parents, truth[position:]) + 1
        drafter.commit(truth[position:position + took])
        position += took
    return len(truth) / rounds, nodes / rounds


def replay_chain(drafter, truth, width):
    position = rounds = 0
    while position < len(truth):
        rounds += 1
        proposal = drafter.propose(width)
        matched = 0
        while (matched < len(proposal) and position + matched < len(truth)
               and proposal[matched] == truth[position + matched]):
            matched += 1
        took = matched + 1
        drafter.commit(truth[position:position + took])
        position += took
    return len(truth) / rounds


def main():
    base = "/home/shanshan/work/probe"
    prompt = json.load(open(f"{base}/prompt-8k.json"))["prompt"]
    streams = json.load(open(f"{base}/real-streams.json"))
    print(f"{'stream':>8} {'budget':>6} {'automaton':>9} "
          f"{'b=4':>7} {'b=8':>7} {'b=16':>7} {'b=64':>7} {'b=1e9':>7}")
    for name, group in streams.items():
        for budget in (12, 32, 64):
            walked = sum(
                replay_chain(AutomatonDrafter(prompt, longest=32, shortest=2,
                                              adaptive=False), truth, budget)
                for truth in group) / len(group)
            cells = []
            for branches in (4, 8, 16, 64, 10 ** 9):
                total = 0.0
                for truth in group:
                    accepted, _ = replay_tree(
                        TreeDrafter(prompt, longest=32, budget=budget,
                                    branches=branches), truth, budget)
                    total += accepted
                cells.append(total / len(group))
            row = " ".join(f"{value:>7.3f}" for value in cells)
            print(f"{name:>8} {budget:>6} {walked:>9.3f} {row}")


main()
