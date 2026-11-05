"""Propose a causal continuation from repeated prompt or output token IDs."""


class _Node:
    __slots__ = ("hi", "length", "children", "latest")

    def __init__(self, hi, length, latest):
        self.hi = hi
        self.length = length
        self.children = {}
        self.latest = latest


class LookupDrafter:
    """Copy what followed the longest, most recently seen context tail.

    A radix trie stores reversed tails, bounded by `longest`. Each compressed
    edge refers to an interval in the append-only accepted context and shares
    one latest continuation position across its token prefixes. Comparisons
    use exact token IDs; insertion and queries do not build reversed words.
    """

    def __init__(self, prompt, longest=32, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Tail lengths must satisfy 1 <= shortest <= "
                             "longest")
        self.context = []
        self.longest = longest
        self.shortest = shortest
        self.hits = 0
        self.rounds = 0
        self._root = _Node(0, 0, -1)
        self.commit(prompt)

    def _shared(self, child, query_hi, remaining):
        length = min(child.length, remaining)
        hi, context = child.hi, self.context
        # Equal forward intervals remain equal when both are reversed.
        if (context[hi - length + 1:hi + 1]
                == context[query_hi - length + 1:query_hi + 1]):
            return length
        for offset in range(length):
            if context[hi - offset] != context[query_hi - offset]:
                return offset
        raise AssertionError("Unequal edge intervals have no differing token")

    def _insert(self, end):
        window_length = min(self.longest, end)
        node, depth, context = self._root, 0, self.context
        while depth < window_length:
            query_hi = end - 1 - depth
            token = context[query_hi]
            child = node.children.get(token)
            if child is None:
                node.children[token] = _Node(
                    query_hi, window_length - depth, end,
                )
                return
            shared = self._shared(child, query_hi, window_length - depth)
            if shared == child.length:
                child.latest = end
                depth += shared
                node = child
                continue
            replacement = _Node(child.hi, shared, end)
            child.hi -= shared
            child.length -= shared
            replacement.children[context[child.hi]] = child
            node.children[token] = replacement
            depth += shared
            if depth < window_length:
                query_hi = end - 1 - depth
                replacement.children[context[query_hi]] = _Node(
                    query_hi, window_length - depth, end,
                )
            return

    def commit(self, tokens):
        """Index tails when an accepted token supplies a continuation."""
        previous = len(self.context)
        self.context.extend(int(token) for token in tokens)
        for end in range(max(previous, self.shortest), len(self.context)):
            self._insert(end)

    def propose(self, count):
        """Return up to `count` tokens after the longest/latest tail match."""
        self.rounds += 1
        maximum = min(self.longest, len(self.context) - 1)
        if maximum < self.shortest or count <= 0:
            return []
        node, matched, end = self._root, 0, -1
        context = self.context
        while matched < maximum:
            query_hi = len(context) - 1 - matched
            child = node.children.get(context[query_hi])
            if child is None:
                break
            shared = self._shared(child, query_hi, maximum - matched)
            matched += shared
            end = child.latest
            if shared < child.length:
                break
            node = child
        if matched < self.shortest:
            return []
        proposed = context[end:end + count]
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
