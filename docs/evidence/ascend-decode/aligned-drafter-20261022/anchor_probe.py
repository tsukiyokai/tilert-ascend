"""Where does the lookup drafter anchor: in the source, or in its own output?

A copy-heavy edit stream should be drafted from the prompt. `_latest` keeps the
most recent occurrence of a tail over prompt plus generated, so as generation
grows its own output becomes the most recent occurrence of everything -- and
the drafter copies from what it just said instead of from what it is
reproducing. This counts which side each round's anchor falls on.
"""
import pathlib
import random
import re
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")

from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter

WORD = re.compile(r"\w+|\s+|[^\w\s]")


def tokenise(text, vocabulary):
    tokens = []
    for piece in WORD.findall(text):
        vocabulary.setdefault(piece, len(vocabulary))
        tokens.append(vocabulary[piece])
    return tokens


def perturb(tokens, rate, size, generator):
    return [generator.randrange(size) if generator.random() < rate else token
            for token in tokens]


def probe(source, truth, width, rate_label):
    drafter = LookupDrafter(source, longest=32, shortest=2)
    prompt_length = len(source)
    accepted_total = 0
    in_source = in_output = no_anchor = 0
    position = 0
    rounds = 0
    while position < len(truth):
        rounds += 1
        # Repeat propose's search to learn which occurrence it would take.
        context = drafter.context
        anchor = None
        for size in range(drafter.longest, drafter.shortest - 1, -1):
            if len(context) <= size:
                continue
            end = drafter._latest.get(tuple(context[-size:]))
            if end is not None and context[end:end + width]:
                anchor = end
                break
        if anchor is None:
            no_anchor += 1
        elif anchor <= prompt_length:
            in_source += 1
        else:
            in_output += 1
        proposal = drafter.propose(width)
        matched = 0
        while (matched < len(proposal) and position + matched < len(truth)
               and proposal[matched] == truth[position + matched]):
            matched += 1
        took = matched + 1
        drafter.commit(truth[position:position + took])
        position += took
        accepted_total += took
    print(f"{rate_label:>6}  width {width:>3}  accepted/round "
          f"{accepted_total / rounds:6.3f}  rounds {rounds:>5}  "
          f"anchor in source {in_source:>5}  in output {in_output:>5}  "
          f"none {no_anchor:>4}")


def main():
    vocabulary = {}
    paths = [pathlib.Path(name) for name in sys.argv[1:]]
    generator = random.Random(20261022)
    for rate in (0.0, 0.02, 0.05):
        for width in (12, 32):
            for path in paths:
                body = tokenise(path.read_text(errors="replace"), vocabulary)
                if len(body) < 64:
                    continue
                truth = perturb(body, rate, len(vocabulary), generator)
                probe(body, truth, width, f"{rate:.2f}")


main()
