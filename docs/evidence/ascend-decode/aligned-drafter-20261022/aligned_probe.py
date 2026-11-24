"""A cursor into the source against a tail-matching drafter, same stream.

The tail matcher is blind in the round after an edit, because the tail it
would search for was just rewritten: at a 2 percent edit rate and width 32
that is 382 of 1188 rounds proposing nothing, and another 255 anchoring in the
drafter's own output rather than in the source it is reproducing. A cursor
keeps the place instead -- a rejection is itself the edit's position, so the
next round proposes from the source token after it.
"""
import pathlib
import random
import re
import sys

sys.path.insert(0, "/home/shanshan/repo/tile-ai/TileRT-Ascend")

from tilert.models.glm_5_2_ascend.lookup_draft import LookupDrafter

WORD = re.compile(r"\w+|\s+|[^\w\s]")


class AlignedDrafter:
    """Propose the rest of the source from a cursor the accepted tokens move."""

    def __init__(self, source, longest=32, shortest=2):
        self.source = list(source)
        self.longest = longest
        self.shortest = shortest
        self.cursor = 0
        self.tail = []
        self._after = {}
        for size in range(shortest, longest + 1):
            for start in range(len(self.source) - size + 1):
                key = tuple(self.source[start:start + size])
                self._after[key] = start + size

    def propose(self, count):
        return self.source[self.cursor:self.cursor + count]

    def commit(self, tokens):
        for token in tokens:
            self.tail.append(token)
            if (self.cursor < len(self.source)
                    and self.source[self.cursor] == token):
                self.cursor += 1
            else:
                self.cursor = self._reanchor()

    def _reanchor(self):
        for size in range(min(self.longest, len(self.tail)),
                          self.shortest - 1, -1):
            after = self._after.get(tuple(self.tail[-size:]))
            if after is not None:
                return after
        return min(self.cursor + 1, len(self.source))


def tokenise(text, vocabulary):
    tokens = []
    for piece in WORD.findall(text):
        vocabulary.setdefault(piece, len(vocabulary))
        tokens.append(vocabulary[piece])
    return tokens


def perturb(tokens, rate, size, generator):
    return [generator.randrange(size) if generator.random() < rate else token
            for token in tokens]


def replay(drafter, truth, width):
    position = 0
    rounds = 0
    blind = 0
    while position < len(truth):
        rounds += 1
        proposal = drafter.propose(width)
        if not proposal:
            blind += 1
        matched = 0
        while (matched < len(proposal) and position + matched < len(truth)
               and proposal[matched] == truth[position + matched]):
            matched += 1
        took = matched + 1
        drafter.commit(truth[position:position + took])
        position += took
    return len(truth) / rounds, rounds, blind


def main():
    vocabulary = {}
    paths = [pathlib.Path(name) for name in sys.argv[1:]]
    generator = random.Random(20261022)
    print(f"{'rate':>5} {'width':>5}  {'lookup':>8} {'aligned':>8} "
          f"{'gain':>6}  {'lk rounds':>9} {'al rounds':>9} {'al blind':>8}")
    for rate in (0.0, 0.02, 0.05, 0.10):
        for width in (12, 32, 64):
            for path in paths:
                body = tokenise(path.read_text(errors="replace"), vocabulary)
                if len(body) < 64:
                    continue
                truth = perturb(body, rate, len(vocabulary), generator)
                lookup, lk_rounds, _ = replay(
                    LookupDrafter(body, longest=32, shortest=2), truth, width)
                aligned, al_rounds, blind = replay(
                    AlignedDrafter(body, longest=32, shortest=2), truth, width)
                print(f"{rate:>5.2f} {width:>5}  {lookup:>8.3f} "
                      f"{aligned:>8.3f} {aligned / lookup:>5.2f}x  "
                      f"{lk_rounds:>9} {al_rounds:>9} {blind:>8}")


main()
