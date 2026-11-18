class SuffixAutomaton:

    ROOT = 0

    def __init__(self):
        self.transitions = [{}]
        self.link = [-1]
        self.length = [0]
        self.prefix_end = [False]
        self.last = self.ROOT
        self._occurrences = None

    def extend(self, token):
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

    def __init__(self, prompt, longest=32, budget=32, floor=1, branches=6):
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
        for token in tokens:
            value = int(token)
            self.context.append(value)
            self.automaton.extend(value)

    def _match_state(self):
        automaton = self.automaton
        counts = automaton.occurrences()
        state = automaton.last
        while state >= SuffixAutomaton.ROOT:
            total = sum(counts[target]
                        for target in automaton.transitions[state].values())
            if total >= self.floor:
                return state
            state = automaton.link[state]
        return None

    def _continuations(self, state):
        automaton = self.automaton
        counts = automaton.occurrences()
        while state is not None and state >= SuffixAutomaton.ROOT:
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
        if count < 1:
            raise ValueError("A tree round must allow at least one node")
        self.budget = count
        return self.propose()

    def propose(self):
        import heapq

        self.rounds += 1
        start = self._match_state()
        if start is None:
            return [], []
        tokens = []
        parents = []
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
