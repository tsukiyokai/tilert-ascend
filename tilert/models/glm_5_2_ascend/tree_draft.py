"""Propose a tree of continuations from repeated prompt or output token IDs.

The linear drafter in `lookup_draft.py` is at its ceiling. Expected tokens
accepted a round is ``(1 - a**(g+1)) / (1 - a)`` for single-token acceptance
``a`` and draft length ``g``, which tends to ``1 / (1 - a)``; the measured 7.1
at width 32 puts ``a`` at 0.86, whose asymptote is 7.14. No draft length beats
that. Only a higher ``a`` does, and a tree raises it by letting one round
verify several continuations and keep the best.

Two things make this more than a list of candidates.

The candidates share prefixes, so the object is a trie and the cost of
verifying it is its node count rather than branches times depth.

And under a budget of ``B`` nodes the choice of which to include is an
optimisation: maximise the expected accepted length, which is the sum over
chosen nodes of the probability that the path to the node is all correct,
subject to the chosen set being prefix-closed. Because a child's probability
is its parent's times a factor no greater than one, the weights fall along
every path, and a best-first expansion by that probability is optimal -- the
largest remaining candidate always has its parent already chosen, so the set
stays prefix-closed, and no prefix-closed set of the same size can have a
larger sum. That is a priority queue and ``O(B log B)``.

The probabilities come from the context itself. If the current suffix occurs
``m`` times with a given token following ``c`` of them, ``c / m`` is the
maximum-likelihood estimate, and a suffix automaton over the token stream
answers that in constant time per edge -- with its suffix links giving the
backoff to a shorter, more frequent context for free.
"""


class SuffixAutomaton:
    """An online suffix automaton over a token stream.

    Each state holds the equivalence class of substrings that end at the same
    set of positions. `extend` is the standard construction, amortised
    constant time, so the whole stream costs time and space linear in its
    length -- against the linear drafter's tuple a suffix a length, which is
    seven hashes of up to eight elements for every token committed.
    """

    ROOT = 0

    def __init__(self):
        self.transitions = [{}]
        self.link = [-1]
        self.length = [0]
        # Set on the state of every prefix, which is where occurrences start.
        self.prefix_end = [False]
        self.last = self.ROOT
        self._occurrences = None

    def extend(self, token):
        """Append one token."""
        self._occurrences = None
        current = self._add_state(self.length[self.last] + 1)
        self.prefix_end[current] = True
        walker = self.last
        while walker != -1 and token not in self.transitions[walker]:
            self.transitions[walker][token] = current
            walker = self.link[walker]
        if walker == -1:
            self.link[current] = self.ROOT
            self.last = current
            return
        existing = self.transitions[walker][token]
        if self.length[walker] + 1 == self.length[existing]:
            self.link[current] = existing
            self.last = current
            return
        clone = self._add_state(self.length[walker] + 1)
        self.transitions[clone] = dict(self.transitions[existing])
        self.link[clone] = self.link[existing]
        self.link[existing] = clone
        self.link[current] = clone
        while walker != -1 and self.transitions[walker].get(token) == existing:
            self.transitions[walker][token] = clone
            walker = self.link[walker]
        self.last = current

    def _add_state(self, length):
        self.transitions.append({})
        self.link.append(-1)
        self.length.append(length)
        self.prefix_end.append(False)
        return len(self.length) - 1

    def occurrences(self):
        """How many positions each state's substrings end at.

        A prefix contributes one occurrence to every suffix of itself, which
        is its state and everything up its suffix links. Summing that is one
        sweep in decreasing order of `length`, and the order comes from a
        bucket pass rather than a sort. Cached until the next `extend`, so a
        round that appends the accepted tokens pays for it once.
        """
        if self._occurrences is not None:
            return self._occurrences
        counts = [1 if end else 0 for end in self.prefix_end]
        buckets = [0] * (max(self.length) + 2)
        for state_length in self.length:
            buckets[state_length] += 1
        start = 0
        for index, size in enumerate(buckets):
            buckets[index] = start
            start += size
        ordered = [0] * len(self.length)
        for state, state_length in enumerate(self.length):
            ordered[buckets[state_length]] = state
            buckets[state_length] += 1
        for state in reversed(ordered):
            parent = self.link[state]
            if parent >= 0:
                counts[parent] += counts[state]
        self._occurrences = counts
        return counts


class TreeDrafter:
    """Hold the context and propose a tree of what followed its tail.

    `propose` returns the tree as parallel arrays -- a token and a parent
    index a node, the root's parent being -1 -- which is exactly what a tree
    attention mask and a position-id vector are built from.
    """

    def __init__(self, prompt, longest=32, budget=32, floor=1, branches=4):
        if budget < 1 or branches < 1 or floor < 1:
            raise ValueError("Budget, branches and floor must be positive")
        self.automaton = SuffixAutomaton()
        self.context = []
        self.longest = longest
        self.budget = budget
        self.floor = floor
        self.branches = branches
        self.rounds = 0
        self.hits = 0
        self.nodes = 0
        self.commit(prompt)

    def commit(self, tokens):
        """Extend the context with what the target actually accepted."""
        for token in tokens:
            value = int(token)
            self.context.append(value)
            self.automaton.extend(value)

    def _match_state(self):
        """The longest suffix of the context with a usable continuation.

        The automaton's `last` state is the whole context, and its suffix
        links run through the context's suffixes from longest to shortest,
        each class holding more occurrences than the one before. So the
        longest suffix with any continuation at all is one walk down that
        chain -- no rehashing a tail at every length, which is what the
        linear drafter does.

        The threshold belongs at one occurrence, not two. A suffix seen once
        before is the whole signal prompt lookup lives on; demanding two
        pushes the match onto a shorter suffix, whose continuations are many
        and flat, and a flat distribution spends the node budget on breadth
        instead of depth.

        Nor is there a cap on how long a match to trust. A class's `length`
        is the longest substring it holds, so inside a repeated region it is
        the whole region; refusing classes above some length does not shorten
        the match, it jumps to a far shorter one and throws away the
        strongest evidence there is. Those two mistakes together had the tree
        accepting 3.9 and then 6.3 tokens where the chain accepted 30.
        """
        automaton = self.automaton
        counts = automaton.occurrences()
        state = automaton.last
        while state > SuffixAutomaton.ROOT:
            total = sum(counts[target]
                        for target in automaton.transitions[state].values())
            if total >= self.floor:
                return state
            state = automaton.link[state]
        return None

    def _continuations(self, state):
        """The likeliest next tokens from `state`, with their probabilities.

        A state deeper in the automaton means a longer, rarer context. When
        it falls below the floor its suffix link offers the same question
        asked of a shorter context, which is the backoff an n-gram model
        needs and here costs one hop.
        """
        automaton = self.automaton
        counts = automaton.occurrences()
        while state is not None and state > SuffixAutomaton.ROOT:
            edges = automaton.transitions[state]
            total = sum(counts[target] for target in edges.values())
            if total >= self.floor:
                ranked = sorted(edges.items(),
                                key=lambda edge: -counts[edge[1]])
                return [(token, counts[target] / total, target)
                        for token, target in ranked[:self.branches]]
            state = automaton.link[state]
        return []

    def propose_tree(self, count):
        """`(tokens, parents)` under a budget of `count` nodes.

        The name is what `MtpRoundExecutor` dispatches on: a drafter that
        offers it gets the tree round, one that does not keeps the chain. The
        budget comes from the round rather than the constructor because the
        captured graph width bounds it -- a tree wider than the graph pays
        for rows the graph cannot replay, and a node past the round's width
        has no verification row to be read from at all. Raising the budget
        to a floor of two overflowed `resolve` at width 2.
        """
        if count < 1:
            raise ValueError("A tree round must allow at least one node")
        self.budget = count
        return self.propose()

    def propose(self):
        """Return `(tokens, parents)` for a tree of at most `budget` nodes."""
        import heapq

        self.rounds += 1
        start = self._match_state()
        if start is None:
            return [], []
        tokens = []
        parents = []
        # Each entry is a candidate node: its probability carries the whole
        # path, so popping the largest keeps the chosen set prefix-closed.
        frontier = []
        order = 0
        for token, chance, target in self._continuations(start):
            heapq.heappush(frontier, (-chance, order, -1, token, target))
            order += 1
        while frontier and len(tokens) < self.budget:
            negated, _, parent, token, target = heapq.heappop(frontier)
            index = len(tokens)
            tokens.append(token)
            parents.append(parent)
            chance = -negated
            for next_token, factor, next_target in self._continuations(target):
                heapq.heappush(frontier,
                               (-(chance * factor), order, index, next_token,
                                next_target))
                order += 1
        if tokens:
            self.hits += 1
            self.nodes += len(tokens)
        return tokens, parents

    def report(self):
        return {
            "drafter": "tree",
            "longest_tail": self.longest,
            "budget": self.budget,
            "branches": self.branches,
            "floor": self.floor,
            "context_tokens": len(self.context),
            "states": len(self.automaton.length),
            "rounds": self.rounds,
            "tail_hits": self.hits,
            "nodes_proposed": self.nodes,
        }
