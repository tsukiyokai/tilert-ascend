"""Does a tree of candidates accept more a round than a single chain?

This is the question the arithmetic left open. Expected accepted tokens are
``(1 - a**(g+1)) / (1 - a)``, which tends to ``1 / (1 - a)``, and the measured
7.1 at width 32 says ``a`` is 0.86 and the asymptote is 7.14 -- so a longer
chain is spent. A tree raises ``a`` instead, and this measures by how much,
on the CPU, before anything touches a device.

The stream is constructed, and says so. There is no GLM-5.2 trace on disk, so
the prompt is a real source file from this repository and the output is that
file again with a share of its tokens perturbed -- which is what an agentic
coding model emits when it edits code it was shown. The edit rate is swept
rather than chosen, because the answer is a curve in the copy rate and a
single number would be a claim about a workload rather than about the
drafters. What the comparison does settle is the part that needs no model:
both drafters see the same stream and the same history.

Run with `-I` and pass the files as arguments.
"""
import argparse
import pathlib
import random
import re
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from tilert.models.glm_5_2_ascend.automaton_draft import AutomatonDrafter
from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter
from tilert.models.glm_5_2_ascend.tree_draft import TreeDrafter

WORD = re.compile(r"\s+|\w+|[^\s\w]")


def tokenise(text, vocabulary):
    """Split into word-like pieces and map them to ids.

    A stand-in for byte-pair encoding. What matters here is that repetition
    in the text becomes repetition in the ids, which any reasonable
    tokeniser preserves.
    """
    tokens = []
    for piece in WORD.findall(text):
        if piece not in vocabulary:
            vocabulary[piece] = len(vocabulary)
        tokens.append(vocabulary[piece])
    return tokens


def perturb(tokens, rate, vocabulary_size, generator):
    """Return `tokens` with `rate` of its positions replaced."""
    edited = list(tokens)
    for index in range(len(edited)):
        if generator.random() < rate:
            edited[index] = generator.randrange(vocabulary_size)
    return edited


def accepted_by_chain(proposal, truth):
    """How many leading tokens of a linear proposal are right."""
    matched = 0
    while (matched < len(proposal) and matched < len(truth)
           and proposal[matched] == truth[matched]):
        matched += 1
    return matched


def accepted_by_tree(tokens, parents, truth):
    """The longest root path through the tree that the target confirms.

    One pass over the nodes in the order they were proposed: a node extends
    the accepted path if its parent is on it and its token is the next true
    one. Parents precede children because the expansion is prefix-closed, so
    no second pass is needed.
    """
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


def run(stream, prompt_length, width, budget, branches, longest):
    """Replay the stream through both drafters and count what each accepts."""
    prompt = stream[:prompt_length]
    # Two chains: the engine's own tail length, and one as long as the tree
    # is allowed to match. Without the second, the comparison would partly
    # measure a longer match rather than a tree.
    chain = LookupDrafter(prompt, longest=8, shortest=2)
    long_chain = LookupDrafter(prompt, longest=longest, shortest=2)
    # The same contract as the chain drafters, so this column is what an
    # engine would see from a module swap alone, with no tree verification.
    automaton = AutomatonDrafter(prompt, longest=longest, shortest=2,
                                 adaptive=False)
    tree = TreeDrafter(prompt, longest=longest, budget=budget,
                       branches=branches)
    chain_accepted = []
    long_accepted = []
    automaton_accepted = []
    adaptive_accepted = []
    adaptive_cost = []
    tree_adaptive_accepted = []
    tree_adaptive_cost = []
    tree_accepted = []
    at = prompt_length
    while at < len(stream):
        truth = stream[at:at + max(width, budget)]
        linear = chain.propose(width)
        matched = accepted_by_chain(linear, truth)
        longer = accepted_by_chain(long_chain.propose(width), truth)
        walked = accepted_by_chain(automaton.propose(width), truth)

        tokens, parents = tree.propose()
        through_tree = accepted_by_tree(tokens, parents, truth)
        # A round commits what the draft got right plus the one token the
        # target produces regardless.
        chain_accepted.append(matched + 1)
        long_accepted.append(longer + 1)
        automaton_accepted.append(walked + 1)

        tree_accepted.append(through_tree + 1)
        step = min(max(matched, longer, walked, through_tree) + 1,
                   len(stream) - at)
        chain.commit(stream[at:at + step])
        long_chain.commit(stream[at:at + step])
        automaton.commit(stream[at:at + step])
        tree.commit(stream[at:at + step])
        at += step
    import os

    factor = float(os.environ.get("FACTOR", "2"))
    floor = int(os.environ.get("FLOOR", "2"))
    # The tree on its own trajectory too, and with its budget sized the way
    # the automaton's width is: a tree whose budget exceeds what the stream
    # will accept pays for the excess in verification rows.
    grove = TreeDrafter(prompt, longest=longest, budget=budget,
                        branches=branches)
    at = prompt_length
    took = 1
    while at < len(stream):
        truth = stream[at:at + budget]
        grove.budget = max(4, min(budget, 2 * took))
        tokens, lineage = grove.propose()
        took = accepted_by_tree(tokens, lineage, truth) + 1
        took = min(took, len(stream) - at)
        grove.commit(stream[at:at + took])
        tree_adaptive_accepted.append(took)
        tree_adaptive_cost.append(len(tokens))
        at += took

    walker = AutomatonDrafter(prompt, longest=longest, shortest=2)
    at = prompt_length
    taken_last = 1
    while at < len(stream):
        truth = stream[at:at + max(width, budget)]
        here = min(width, max(floor, int(round(factor * taken_last))))
        taken = accepted_by_chain(walker.propose(here), truth) + 1
        taken = min(taken, len(stream) - at)
        walker.commit(stream[at:at + taken])
        adaptive_accepted.append(taken)
        adaptive_cost.append(here)
        taken_last = taken
        at += taken

    return (chain_accepted, long_accepted, automaton_accepted,
            adaptive_accepted, adaptive_cost, tree_accepted,
            tree_adaptive_accepted, tree_adaptive_cost)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--longest", type=int, default=32)
    parser.add_argument("--rates", default="0.0,0.02,0.05,0.10,0.20")
    parser.add_argument("--seed", type=int, default=20261020)
    arguments = parser.parse_args()

    vocabulary = {}
    bodies = [tokenise(path.read_text(errors="replace"), vocabulary)
              for path in arguments.sources]
    print(f"vocabulary {len(vocabulary)}  "
          f"files {[len(body) for body in bodies]}")
    print(f"width {arguments.width}  budget {arguments.budget}  "
          f"branches {arguments.branches}  longest {arguments.longest}")
    print(f"{'edit rate':>10}  {'chain8':>7}  {'chainL':>7}  "
          f"{'automaton':>9}  {'adaptive':>8}  {'adp rows':>8}  "
          f"{'tree':>7}  {'tpot@aut':>8}  {'tpot@adp':>8}  "
          f"{'grove':>7}  {'grv rows':>8}  {'tpot@grv':>8}  {'rounds':>7}")
    for rate in [float(piece) for piece in arguments.rates.split(",")]:
        generator = random.Random(arguments.seed)
        chain_total = []
        long_total = []
        automaton_total = []
        adaptive_total = []
        adaptive_width = []
        grove_total = []
        grove_width = []
        tree_total = []
        for body in bodies:
            if len(body) < 64:
                continue
            output = perturb(body, rate, len(vocabulary), generator)
            stream = body + output
            columns = run(stream, len(body), arguments.width,
                          arguments.budget, arguments.branches,
                          arguments.longest)
            (chain_accepted, long_accepted, automaton_accepted,
             adaptive_accepted, adaptive_cost, tree_accepted,
             grove_accepted, grove_cost) = columns
            grove_total.extend(grove_accepted)
            grove_width.extend(grove_cost)
            adaptive_total.extend(adaptive_accepted)
            adaptive_width.extend(adaptive_cost)
            chain_total.extend(chain_accepted)
            long_total.extend(long_accepted)
            automaton_total.extend(automaton_accepted)
            tree_total.extend(tree_accepted)
        if not chain_total:
            continue
        chain_mean = sum(chain_total) / len(chain_total)
        long_mean = sum(long_total) / len(long_total)
        walked_mean = sum(automaton_total) / len(automaton_total)
        tree_mean = sum(tree_total) / len(tree_total)
        adaptive_mean = sum(adaptive_total) / len(adaptive_total)
        width_mean = sum(adaptive_width) / len(adaptive_width)
        grove_mean = sum(grove_total) / len(grove_total)
        grove_rows = sum(grove_width) / len(grove_width)
        grove_tpot = (45.88 + 0.814 * grove_rows) / grove_mean
        fixed_tpot = (45.88 + 0.814 * arguments.width) / walked_mean
        adaptive_tpot = (45.88 + 0.814 * width_mean) / adaptive_mean
        print(f"{rate:>10.2f}  {chain_mean:>7.3f}  {long_mean:>7.3f}  "
              f"{walked_mean:>9.3f}  {adaptive_mean:>8.3f}  "
              f"{width_mean:>8.1f}  {tree_mean:>7.3f}  "
              f"{fixed_tpot:>8.2f}  {adaptive_tpot:>8.2f}  "
              f"{grove_mean:>7.3f}  {grove_rows:>8.1f}  "
              f"{grove_tpot:>8.2f}  {len(chain_total):>7}")


if __name__ == "__main__":
    main()
