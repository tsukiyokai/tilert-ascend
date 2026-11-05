"""Draft a linear continuation from a suffix automaton over the token stream.

The same contract as `lookup_draft.LookupDrafter` -- `commit` then `propose`,
returning a flat list -- so it is a module swap with no engine change, which
is what makes it measurable end to end before tree verification exists.

Two differences from the hash-tuple drafter, both of which bear on acceptance
rather than on speed.

It matches the longest tail that has any continuation at all, in one walk
down the suffix links, where the other rehashes a tail at every length from
32 down to 2 and stops at the first hit. The walk is exact: a state's suffix
link is the longest proper suffix in a different equivalence class, so
descending it enumerates the context's suffixes longest first.

And at each step it takes the most frequent continuation rather than the one
from the most recent occurrence. For a model that copies its input with
edits, the two differ exactly where it matters: an edit breaks the long tail,
and what follows the latest occurrence of a short tail is a worse predictor
than what follows most of its occurrences.
"""
from .tree_draft import SuffixAutomaton


class AutomatonDrafter:
    """Hold the context and propose what most often followed its tail."""

    def __init__(self, prompt, longest=32, shortest=2, adaptive=True):
        # Kept for report parity with the drafter this replaces; the walk
        # needs no length bounds of its own.
        self.longest = longest
        self.shortest = shortest
        self.automaton = SuffixAutomaton()
        self.context = []
        # A caller that needs a fixed row count -- a graph with static token
        # buckets, or a measurement of the ceiling itself -- turns this off.
        self.adaptive = adaptive
        self.accepted_last = 1
        self.rounds = 0
        self.hits = 0
        self.commit(prompt)

    def commit(self, tokens):
        """Extend the context with what the target accepted."""
        self.accepted_last = max(1, len(tokens))
        for token in tokens:
            value = int(token)
            self.context.append(value)
            self.automaton.extend(value)

    def propose(self, count):
        """Return up to `count` tokens, following the likeliest path.

        `count` is a ceiling, not a quota. A round costs the fixed term plus
        0.814 ms a proposed row, so proposing thirty-two rows to have eight
        accepted pays for twenty-four that were never going to hold. The last
        round's acceptance is the signal: a copy stretch takes the whole
        proposal many rounds running, a rewrite drops to one immediately. Twice
        what the last round took, measured over this corpus, keeps 83 percent
        of the acceptance for 39 percent of the width -- TPOT 6.75 against
        7.18 at an edit rate of 0.02 and 12.02 against 13.24 at 0.05.
        """
        self.rounds += 1
        if self.adaptive:
            count = min(count, max(2, 2 * self.accepted_last))
        counts = self.automaton.occurrences()
        state = self._deepest_with_continuation(counts)
        if state is None:
            return []
        drafted = []
        while len(drafted) < count:
            edge = self._likeliest(state, counts)
            if edge is None:
                break
            token, state = edge
            drafted.append(token)
        if drafted:
            self.hits += 1
        return drafted

    def _deepest_with_continuation(self, counts):
        """The longest suffix of the context that anything ever followed.

        One descent of the suffix links from the state of the whole context.
        A class with no outgoing edge occurs only at the very end of the
        stream, so the descent stops at the first class that has one.
        """
        state = self.automaton.last
        while state > SuffixAutomaton.ROOT:
            if self.automaton.transitions[state]:
                return state
            state = self.automaton.link[state]
        return None

    def _likeliest(self, state, counts):
        """The most frequent continuation of `state`, or None.

        Backing off through the suffix link when a state runs out keeps the
        draft going past the point where the longest match ends, which is
        where a chain drafter stops.
        """
        while state is not None and state > SuffixAutomaton.ROOT:
            edges = self.automaton.transitions[state]
            if edges:
                return max(edges.items(), key=lambda edge: counts[edge[1]])
            state = self.automaton.link[state]
        return None

    def matched_length(self):
        """How long a suffix of the context has any continuation.

        The drafter's own confidence, and the signal that separates a copy
        stretch from a rewrite: a long match means the context agreed, a short
        one means an edit has just broken it.
        """
        counts = self.automaton.occurrences()
        state = self._deepest_with_continuation(counts)
        return 0 if state is None else self.automaton.length[state]

    def report(self):
        return {
            "drafter": "automaton",
            "longest_tail": self.longest,
            "shortest_tail": self.shortest,
            "context_tokens": len(self.context),
            "states": len(self.automaton.length),
            "rounds": self.rounds,
            "tail_hits": self.hits,
            "accepted_last": self.accepted_last,
        }
