"""Tuple-label arena arm for the coordinate-arena /tmp prototype.

This retains the production tuple comparisons and reversed words. It removes
per-node Python objects and child dictionaries, using integer node IDs and a
global transition table. Run the sibling prototype with --tuple-arm --check.
"""

from array import array


def transition_key(parent, token):
    if 0 <= token < 1 << 18:
        return (parent << 18) | token
    return parent, token


def common_prefix(label, word, offset):
    length = min(len(label), len(word) - offset)
    part = word[offset:offset + length]
    if label[:length] == part:
        return length
    for index, (left, right) in enumerate(zip(label, part)):
        if left != right:
            return index
    raise AssertionError("Unequal token tails have no differing element")


class TupleArenaLookupDrafter:
    def __init__(self, prompt, longest=32, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Tail lengths must satisfy 1 <= shortest <= longest")
        self.context = []
        self.longest = longest
        self.shortest = shortest
        self.hits = 0
        self.rounds = 0
        self._labels = [None] * 16
        self._labels[0] = ()
        self._latest = array("q", [0]) * 16
        self._latest[0] = -1
        self._size = 1
        self._children = {}
        self.commit(prompt)

    def _reserve(self, minimum):
        capacity = len(self._latest)
        if minimum <= capacity:
            return
        new_capacity = max(minimum, capacity * 2)
        self._latest.extend(array("q", [0]) * (new_capacity - capacity))
        self._labels.extend([None] * (new_capacity - capacity))

    def _new(self, label, end):
        node = self._size
        self._reserve(node + 1)
        self._labels[node] = label
        self._latest[node] = end
        self._size += 1
        return node

    def _insert(self, word, end):
        node, offset = 0, 0
        while offset < len(word):
            key = transition_key(node, word[offset])
            child = self._children.get(key)
            if child is None:
                self._children[key] = self._new(word[offset:], end)
                return
            label = self._labels[child]
            shared = common_prefix(label, word, offset)
            if shared == len(label):
                self._latest[child] = end
                offset += shared
                node = child
                continue
            replacement = self._new(label[:shared], end)
            self._labels[child] = label[shared:]
            self._children[transition_key(
                replacement, self._labels[child][0],
            )] = child
            self._children[key] = replacement
            offset += shared
            if offset < len(word):
                self._children[transition_key(
                    replacement, word[offset],
                )] = self._new(word[offset:], end)
            return

    def commit(self, tokens):
        previous = len(self.context)
        self.context.extend(int(token) for token in tokens)
        if len(self.context) == previous:
            return
        if self._size == 1:
            self._reserve(1 + 2 * max(0, len(self.context) - self.shortest))
        for end in range(max(previous, self.shortest), len(self.context)):
            word = tuple(reversed(
                self.context[max(0, end - self.longest):end],
            ))
            self._insert(word, end)

    def propose(self, count):
        self.rounds += 1
        maximum = min(self.longest, len(self.context) - 1)
        if maximum < self.shortest or count <= 0:
            return []
        word = tuple(reversed(self.context[-maximum:]))
        node, matched, end = 0, 0, -1
        while matched < maximum:
            child = self._children.get(transition_key(node, word[matched]))
            if child is None:
                break
            label = self._labels[child]
            shared = common_prefix(label, word, matched)
            matched += shared
            end = self._latest[child]
            if shared < len(label):
                break
            node = child
        if matched < self.shortest:
            return []
        proposed = self.context[end:end + count]
        self.hits += bool(proposed)
        return proposed

    def report(self):
        return {
            "drafter": "lookup", "longest_tail": self.longest,
            "shortest_tail": self.shortest, "context_tokens": len(self.context),
            "rounds": self.rounds, "tail_hits": self.hits,
        }
