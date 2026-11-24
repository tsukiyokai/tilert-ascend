"""Propose a causal continuation from repeated prompt or output token IDs."""


class LookupDrafter:
    """Hold the context and propose what followed its tail last time.

    The context is the prompt plus every committed token, which is what the
    model attends to and therefore what it can copy from.
    """

    def __init__(self, prompt, longest=8, shortest=2):
        if shortest < 1 or longest < shortest:
            raise ValueError("Tail lengths must satisfy 1 <= shortest <= "
                             "longest")
        self.context = []
        self.longest = longest
        self.shortest = shortest
        self._latest = {}
        self.hits = 0
        self.rounds = 0
        self.commit(prompt)

    def commit(self, tokens):
        """Extend the context with what the target actually accepted."""
        previous_length = len(self.context)
        self.context.extend(int(token) for token in tokens)
        # Only index suffixes with a known continuation. A newly appended
        # token makes the preceding tail eligible, never the current tail.
        for end in range(max(previous_length, self.shortest),
                         len(self.context)):
            for size in range(self.shortest, min(self.longest, end) + 1):
                key = tuple(self.context[end - size:end])
                self._latest[key] = end

    def propose(self, count):
        """Return up to `count` tokens, longest tail match first.

        Search from `longest` down and take the most recent occurrence at
        the first length with a known continuation.
        """
        self.rounds += 1
        context = self.context
        for size in range(self.longest, self.shortest - 1, -1):
            if len(context) <= size:
                continue
            end = self._latest.get(tuple(context[-size:]))
            if end is not None:
                run = context[end:end + count]
                if run:
                    self.hits += 1
                    return run
        return []

    def report(self):
        return {
            "drafter": "lookup",
            "longest_tail": self.longest,
            "shortest_tail": self.shortest,
            "context_tokens": len(self.context),
            "rounds": self.rounds,
            "tail_hits": self.hits,
        }
