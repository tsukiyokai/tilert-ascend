"""Propose a causal continuation from repeated prompt or output token IDs."""


class _Node:
    __slots__ = ("label", "children", "latest")

    def __init__(self, label, latest):
        self.label = label
        self.children = {}
        self.latest = latest


def _common_prefix(label, word, offset):
    length = min(len(label), len(word) - offset)
    part = word[offset:offset + length]
    if label[:length] == part:
        return length
    for index, (left, right) in enumerate(zip(label, part)):
        if left != right:
            return index
    raise AssertionError("Unequal token tails have no differing element")


class LookupDrafter:
    """Copy what followed the longest, most recently seen context tail.

    A radix trie stores reversed tails, bounded by `longest`. Each compressed
    edge shares one latest continuation position across its token prefixes.
    Appending a token inserts the preceding tail once, and querying walks
    the current reversed tail once. Token comparisons are exact.
    """

    def __init__(self, prompt, longest=32, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Tail lengths must satisfy 1 <= shortest <= "
                             "longest")
        self.context = []
        self.longest = longest
        self.shortest = shortest
        self._root = _Node((), -1)
        self.hits = 0
        self.rounds = 0
        self.commit(prompt)

    def _insert(self, word, end):
        node, offset = self._root, 0
        while offset < len(word):
            child = node.children.get(word[offset])
            if child is None:
                node.children[word[offset]] = _Node(word[offset:], end)
                return
            shared = _common_prefix(child.label, word, offset)
            if shared == len(child.label):
                child.latest = end
                offset += shared
                node = child
                continue
            # A new branch or shorter tail splits the old compressed edge.
            replacement = _Node(child.label[:shared], end)
            child.label = child.label[shared:]
            replacement.children[child.label[0]] = child
            node.children[word[offset]] = replacement
            offset += shared
            if offset < len(word):
                replacement.children[word[offset]] = _Node(word[offset:], end)
            return

    def commit(self, tokens):
        """Index tails only when an accepted token supplies a continuation."""
        previous = len(self.context)
        self.context.extend(int(token) for token in tokens)
        for end in range(max(previous, self.shortest), len(self.context)):
            word = tuple(reversed(
                self.context[max(0, end - self.longest):end],
            ))
            self._insert(word, end)

    def propose(self, count):
        """Return up to `count` tokens after the longest/latest tail match."""
        self.rounds += 1
        maximum = min(self.longest, len(self.context) - 1)
        if maximum < self.shortest or count <= 0:
            return []
        word = tuple(reversed(self.context[-maximum:]))
        node, matched, end = self._root, 0, -1
        while matched < maximum:
            child = node.children.get(word[matched])
            if child is None:
                break
            shared = _common_prefix(child.label, word, matched)
            matched += shared
            end = child.latest
            if shared < len(child.label):
                break
            node = child
        if matched < self.shortest:
            return []
        proposed = self.context[end:end + count]
        self.hits += bool(proposed)
        return proposed

    def report(self):
        return {
            "drafter": "lookup",
            "longest_tail": self.longest,
            "shortest_tail": self.shortest,
            "context_tokens": len(self.context),
            "rounds": self.rounds,
            "tail_hits": self.hits,
        }
