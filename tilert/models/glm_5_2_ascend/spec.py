from dataclasses import dataclass
import heapq
from time import perf_counter_ns

from .config import VOCAB_SIZE, WIDTHS

EMPTY = -1


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _token(value):
    value = _integer(value, "token")
    if value >= VOCAB_SIZE:
        raise ValueError("token is outside the GLM-5.2 vocabulary")
    return value


@dataclass(frozen=True)
class RoundPlan:
    epoch: int
    number: int
    width: int
    pending_token: int
    main_start: int
    draft_before: int | None
    draft_seed_position: int | None
    draft_verify_start: int | None

    @property
    def main_end(self):
        return self.main_start + self.width

    @property
    def draft_end(self):
        if self.draft_verify_start is None:
            return self.draft_before
        return self.draft_verify_start + self.width


@dataclass(frozen=True)
class CacheCommit:
    before: int
    written: int
    retained: int

    @property
    def discard(self):
        return (self.retained, self.written)


@dataclass(frozen=True)
class HiddenSource:
    kind: str
    row: int
    position: int


@dataclass(frozen=True)
class RoundResult:
    number: int
    width: int
    matched_drafts: int
    natural_accepted: int
    accepted_tokens: tuple[int, ...]
    consumed_tokens: tuple[int, ...]
    pending_token: int
    main: CacheCommit
    draft: CacheCommit | None
    main_hidden: HiddenSource
    draft_hidden: HiddenSource | None
    finish_reason: str | None

    @property
    def committed(self):
        return len(self.accepted_tokens)


class MtpState:

    def __init__(self, width, capacity, max_new_tokens, eos_tokens=()):
        _integer(width, "width", 1)
        if width not in WIDTHS:
            raise ValueError(f"verification width must be one of {WIDTHS}")
        self.width = width
        self.capacity = _integer(capacity, "capacity", 1)
        self.max_new_tokens = _integer(max_new_tokens, "max_new_tokens")
        self.eos_tokens = frozenset(_token(token) for token in eos_tokens)
        self.epoch = 0
        self.main_length = 0
        self.draft_length = None
        self.pending_token = None
        self.emitted_count = 0
        self.finish_reason = "uninitialized"
        self.history = []
        self._active = None

    def reset(self, prefix_length, first_token, draft_prefix_length=None):
        prefix_length = _integer(prefix_length, "prefix_length", 1)
        first_token = _token(first_token)
        if prefix_length > self.capacity:
            raise ValueError("prefix exceeds cache capacity")
        if self.width == 1:
            if draft_prefix_length is not None:
                raise ValueError("plain decode has no MTP cache")
        elif draft_prefix_length is not None:
            _integer(draft_prefix_length, "draft_prefix_length")
            if draft_prefix_length not in (prefix_length - 1, prefix_length):
                raise ValueError("MTP prefix must have length P-1 or P")
            if draft_prefix_length > self.capacity:
                raise ValueError("MTP prefix exceeds cache capacity")
        self.epoch += 1
        self.main_length = prefix_length
        self.draft_length = draft_prefix_length
        self.pending_token = first_token if self.max_new_tokens else None
        self.emitted_count = int(self.max_new_tokens > 0)
        self.finish_reason = self._finish_reason()
        self.history = []
        self._active = None

    def _finish_reason(self):
        if self.pending_token in self.eos_tokens:
            return "eos"
        if self.emitted_count >= self.max_new_tokens:
            return "length"
        if self.main_length == self.capacity:
            return "capacity"
        return None

    def begin(self):
        if self.finish_reason is not None:
            raise ValueError(f"request cannot decode: {self.finish_reason}")
        if self._active is not None:
            raise ValueError("a verification round is already active")
        room = self.capacity - self.main_length
        use_draft = self.draft_length is not None
        width = max(
            value for value in WIDTHS if value <= min(self.width, room)
        )
        draft_start = self.main_length if use_draft else None
        needs_seed = use_draft and self.draft_length == self.main_length - 1
        plan = RoundPlan(
            self.epoch,
            len(self.history) + 1,
            width,
            self.pending_token,
            self.main_length,
            self.draft_length,
            self.main_length - 1 if needs_seed else None,
            draft_start,
        )
        self._active = plan
        return plan

    def _require_active(self, plan):
        if plan is not self._active or plan is None:
            raise ValueError("round is stale or is not active")

    def resolve(self, plan, candidates, target_predictions):
        self._require_active(plan)
        candidates = tuple(_token(token) for token in candidates)
        predictions = tuple(_token(token) for token in target_predictions)
        if len(candidates) != plan.width or len(predictions) != plan.width:
            raise ValueError("candidate and prediction counts must match M")
        if candidates[0] != plan.pending_token:
            raise ValueError("candidate zero must be the pending token")
        matched = 0
        for proposed, predicted in zip(candidates[1:], predictions):
            if proposed != predicted:
                break
            matched += 1
        natural = matched + 1
        count = min(natural, self.max_new_tokens - self.emitted_count)
        for index, token in enumerate(predictions[:count]):
            if token in self.eos_tokens:
                count = index + 1
                break
        accepted = predictions[:count]
        main = CacheCommit(
            plan.main_start, plan.main_end, plan.main_start + count
        )
        draft = None
        draft_hidden = None
        if plan.draft_verify_start is not None:
            draft = CacheCommit(
                plan.draft_before, plan.draft_end, main.retained
            )
            draft_hidden = HiddenSource(
                "verified_mtp", count - 1, main.retained - 1
            )
        self.main_length = main.retained
        if draft is not None:
            self.draft_length = draft.retained
        self.pending_token = accepted[-1]
        self.emitted_count += count
        self.finish_reason = self._finish_reason()
        result = RoundResult(
            plan.number,
            plan.width,
            matched,
            natural,
            accepted,
            candidates[:count],
            self.pending_token,
            main,
            draft,
            HiddenSource("target", count - 1, main.retained - 1),
            draft_hidden,
            self.finish_reason,
        )
        self.history.append(result)
        self._active = None
        return result

    def abort(self, plan):
        self._require_active(plan)
        main = CacheCommit(plan.main_start, plan.main_end, plan.main_start)
        draft = None
        if plan.draft_verify_start is not None:
            draft = CacheCommit(
                plan.draft_before, plan.draft_end, plan.draft_before
            )
        self._active = None
        return main, draft

    def statistics(self):
        committed = sum(item.committed for item in self.history)
        rounds = len(self.history)
        return {
            "rounds": rounds,
            "verified_positions": sum(item.width for item in self.history),
            "matched_drafts": sum(item.matched_drafts for item in self.history),
            "natural_accepted": sum(
                item.natural_accepted for item in self.history
            ),
            "committed": committed,
            "emitted": self.emitted_count,
            "mean_committed_per_round": committed / rounds if rounds else 0,
        }


def prefix_room(rows, prefix_length, picks):
    room = picks - rows
    if room < prefix_length:
        raise ValueError(
            f"A tree of {rows} rows under {picks} picks leaves "
            f"{max(0, room)} for a prefix of {prefix_length}; raise "
            "index_topk or pass the indexer's own selection"
        )
    return room


def ancestry(parents):
    lines = []
    for node, parent in enumerate(parents):
        lines.append(([] if parent < 0 else lines[parent]) + [node])
    return lines


def branch_depths(parents):
    depths = []
    for parent in parents:
        depths.append(0 if parent < 0 else depths[parent] + 1)
    return depths


def ancestor_rows(parents, prefix_length, picks, prefix_picks=None):
    if prefix_picks is None:
        prefix_picks = prefix_room(len(parents), prefix_length, picks)
    shared = list(range(max(0, prefix_length - prefix_picks), prefix_length))
    rows = []
    for line in ancestry(parents):
        row = shared + [prefix_length + node for node in line]
        if len(row) > picks:
            row = row[len(row) - picks :]
        rows.append(row + [EMPTY] * (picks - len(row)))
    return rows


def visible(rows, node):
    return [position for position in rows[node] if position != EMPTY]


def ancestor_bitmap(parents, prefix_length, capacity, device=None):
    import torch

    lines = ancestry(parents)
    deepest = max(len(line) for line in lines)
    bitmap = torch.ones((len(parents), capacity), dtype=torch.uint8, device=device)
    bitmap[:, :prefix_length] = 0
    columns = torch.tensor(
        [
            [prefix_length + node for node in line]
            + [prefix_length + line[-1]] * (deepest - len(line))
            for line in lines
        ],
        dtype=torch.int64,
        device=device,
    )
    bitmap.scatter_(1, columns, 0)
    return bitmap.view(torch.bool)


def accepted_path(tokens, parents, predictions):
    if not tokens:
        return []
    reachable = [False] * len(tokens)
    came_from = [-1] * len(tokens)
    reachable[0] = True
    for node in range(1, len(tokens)):
        parent = parents[node]
        if parent < 0 or not reachable[parent]:
            continue
        if predictions[parent] == tokens[node]:
            reachable[node] = True
            came_from[node] = parent
    deepest = 0
    depth = [0] * len(tokens)
    for node in range(1, len(tokens)):
        if reachable[node]:
            depth[node] = depth[parents[node]] + 1
            if depth[node] > depth[deepest]:
                deepest = node
    path = []
    walk = deepest
    while walk >= 0:
        path.append(walk)
        walk = came_from[walk] if walk else -1
    path.reverse()
    return path


def accepted_tokens(tokens, parents, predictions):
    path = accepted_path(tokens, parents, predictions)
    committed = [tokens[node] for node in path[1:]]
    committed.append(predictions[path[-1]])
    return committed


def install_tree_attention(layers, picks):
    import torch

    held = []
    for layer in layers:
        held.append((layer, layer.skip_topk, layer.topk_indices_buffer))

    buffer = None
    ancestors = None
    cutoff = None

    def write(parents, prefix_length, device):
        nonlocal buffer, ancestors, cutoff
        lines = ancestry(parents)
        if len(parents) + prefix_length > picks:
            width = len(parents)
            if ancestors is None or ancestors.shape[0] != width:
                ancestors = torch.empty(
                    (width, 1, width), dtype=torch.int32, device=device
                )
                cutoff = torch.empty((1, 1, 1), dtype=torch.int32, device=device)
                for layer, _, _ in held:
                    layer._tree_ancestors = ancestors
                    layer._tree_prefix_len = cutoff
            cutoff.fill_(prefix_length)
            ancestors.copy_(
                torch.tensor(
                    [
                        [
                            [prefix_length + node for node in line]
                            + [EMPTY] * (width - len(line))
                        ]
                        for line in lines
                    ],
                    dtype=torch.int32,
                    device=device,
                )
            )
            return ancestors
        prefix_room(len(parents), prefix_length, picks)
        if buffer is None or buffer.shape != (len(parents), picks):
            buffer = torch.empty(
                (len(parents), picks), dtype=torch.int32, device=device
            )
            for layer, _, _ in held:
                layer.skip_topk = True
                layer.topk_indices_buffer = buffer
        deepest = max(len(line) for line in lines)
        buffer[:, :prefix_length] = torch.arange(
            prefix_length, dtype=torch.int32, device=device
        )
        buffer[:, prefix_length + deepest :] = EMPTY
        buffer[:, prefix_length : prefix_length + deepest] = torch.tensor(
            [
                [prefix_length + node for node in line]
                + [EMPTY] * (deepest - len(line))
                for line in lines
            ],
            dtype=torch.int32,
            device=device,
        )
        return buffer

    def restore():
        for layer, skip, original in held:
            layer.skip_topk = skip
            layer.topk_indices_buffer = original
            layer._tree_ancestors = None
            layer._tree_prefix_len = None

    return write, restore


install = install_tree_attention


class RoundPhases:

    def __init__(self):
        self.reset()

    def reset(self):
        self.totals = {}
        self.counts = {}
        self.rounds = 0
        self._last = None

    def begin(self):
        self.rounds += 1
        self._last = perf_counter_ns()

    def mark(self, step):
        now = perf_counter_ns()
        if self._last is not None:
            self.totals[step] = self.totals.get(step, 0) + now - self._last
            self.counts[step] = self.counts.get(step, 0) + 1
        self._last = now

    def report(self):
        if not self.rounds:
            return {}
        held = {
            step: nanoseconds / self.rounds / 1e6
            for step, nanoseconds in self.totals.items()
        }
        held["marks_a_round"] = {
            step: count / self.rounds for step, count in self.counts.items()
        }
        held["rounds"] = self.rounds
        return held


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
            total = sum(
                counts[target] for target in automaton.transitions[state].values()
            )
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
                ranked = sorted(
                    edges.items(), key=lambda edge: -counts[edge[1]]
                )
                return [
                    (token, counts[target] / total, target)
                    for token, target in ranked[: self.branches]
                ]
            state = automaton.link[state]
        return []

    def propose_tree(self, count):
        if count < 1:
            raise ValueError("A tree round must allow at least one node")
        self.budget = count
        return self.propose()

    def propose(self):
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
                heapq.heappush(
                    frontier,
                    (-(chance * factor), order, index, next_token, next_target),
                )
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


@dataclass(frozen=True)
class DraftSeed:
    token: object
    recycle_hidden: object
    indices: object


@dataclass(frozen=True)
class ExecutedRound:
    candidates: tuple[int, ...]
    target_predictions: tuple[int, ...]
    commit: object
    parents: tuple[int, ...] | None = None


class MtpRoundExecutor:

    def __init__(
        self,
        target,
        draft,
        width=4,
        max_new_tokens=1024,
        eos_tokens=(),
        drafter=None,
        phases=None,
    ):
        if width not in (2, 4, 6, 8, 12, 16, 24, 32):
            raise ValueError("MTP rounds require width 2, 4, 8 or 16")
        if draft is not None and target.capacity != draft.capacity:
            raise ValueError("Main and MTP cache capacities must match")
        self.target = target
        self.draft = draft
        self.phases = phases
        self.drafter = drafter
        self.state = MtpState(width, target.capacity, max_new_tokens, eos_tokens)
        self.seed = None
        self.seed_parent_hidden = None
        self.failed = False
        self.tree_mask = None

    def start(self, first_token, normalized_parent_hidden):
        self.state.reset(
            self.target.position,
            first_token,
            draft_prefix_length=(
                self.draft.position
                if self.drafter is None and self.draft is not None
                else None
            ),
        )
        self.seed_parent_hidden = normalized_parent_hidden
        self.seed = None
        self.failed = False

    @staticmethod
    def _seed(result, row):
        return DraftSeed(
            result.token_ids[row].clone(),
            result.recycle_hidden[row : row + 1],
            result.indices[row : row + 1],
        )

    def advance(self):
        if self.failed:
            raise RuntimeError("Inject fresh prefixes after a failed round")
        if self.target.position != self.state.main_length:
            raise ValueError("Engine lengths differ from the MTP state")
        if self.drafter is not None:
            if self.phases is not None:
                self.phases.begin()
            return self._advance_drafted(self.state.begin())
        if self.draft.position != self.state.draft_length:
            raise ValueError("Engine lengths differ from the MTP state")
        plan = self.state.begin()
        try:
            if self.seed is None:
                if self.seed_parent_hidden is None:
                    raise ValueError("MTP seed needs the parent hidden")
                self.draft.truncate(plan.main_start - 1)
                seeded = self.draft.forward(
                    (plan.pending_token,),
                    self.seed_parent_hidden,
                    full_logits=False,
                )
                self.seed = self._seed(seeded, 0)
            seed = self.seed
            candidates = [plan.pending_token]
            shared_indices = seed.indices
            while len(candidates) < plan.width:
                candidates.append(seed.token)
                if len(candidates) < plan.width:
                    chained = self.draft.forward(
                        [seed.token],
                        seed.recycle_hidden,
                        full_logits=False,
                        external_indices=shared_indices,
                    )
                    seed = self._seed(chained, 0)
            verified = self.target.verify(candidates, full_logits=False)
            self.draft.truncate(plan.main_start)
            conditioned = self.draft.forward(
                verified.token_ids, verified.last_hidden, full_logits=False
            )
            proposals, predictions = self.target.host_tokens(
                candidates, verified.token_ids
            )
            commit = self.state.resolve(plan, proposals, predictions)
            self.target.truncate(commit.main.retained)
            self.draft.truncate(commit.draft.retained)
            self.seed = self._seed(conditioned, commit.committed - 1)
            self.seed_parent_hidden = None
            return ExecutedRound(proposals, predictions, commit)
        except Exception:
            self.failed = True
            raise

    def _advance_tree(self, plan, propose_tree):
        mark = self.phases.mark if self.phases is not None else None
        tokens, parents = (
            propose_tree(plan.width - 1) if plan.width > 1 else ([], [])
        )
        if mark:
            mark("propose")
        nodes = [plan.pending_token, *tokens]
        lineage = [-1, *(parent + 1 for parent in parents)]
        nodes += [plan.pending_token] * (plan.width - len(nodes))
        lineage += [0] * (plan.width - len(lineage))
        if self.tree_mask is not None:
            self.tree_mask(lineage, plan.main_start, self.target.device)
        verified = self.target.verify(
            nodes, full_logits=False, depths=branch_depths(lineage)
        )
        if mark:
            mark("verify")
        proposals = tuple(nodes)
        (predictions,) = self.target.host_tokens(verified.token_ids)
        if mark:
            mark("host_tokens")
        path = accepted_path(nodes, lineage, predictions)
        along = [nodes[node] for node in path]
        predicted = [predictions[node] for node in path]
        bonus = predictions[path[-1]]
        along += [(bonus + 1) % VOCAB_SIZE] * (plan.width - len(along))
        predicted += [bonus] * (plan.width - len(predicted))
        commit = self.state.resolve(plan, along, predicted)
        self.target.compact(plan.main_start, path)
        self.target.truncate(commit.main.retained)
        self.drafter.commit(commit.accepted_tokens)
        if mark:
            mark("resolve")
        return ExecutedRound(proposals, predictions, commit, tuple(lineage))

    def _advance_drafted(self, plan):
        try:
            tree = getattr(self.drafter, "propose_tree", None)
            if tree is not None:
                return self._advance_tree(plan, tree)
            mark = self.phases.mark if self.phases is not None else None
            proposal = self.drafter.propose(plan.width - 1)
            if mark:
                mark("propose")
            candidates = [plan.pending_token, *proposal[: plan.width - 1]]
            candidates += [plan.pending_token] * (plan.width - len(candidates))
            verified = self.target.verify(candidates, full_logits=False)
            if mark:
                mark("verify")
            proposals = tuple(candidates)
            (predictions,) = self.target.host_tokens(verified.token_ids)
            if mark:
                mark("host_tokens")
            commit = self.state.resolve(plan, proposals, predictions)
            self.target.truncate(commit.main.retained)
            self.drafter.commit(commit.accepted_tokens)
            if mark:
                mark("resolve")
            return ExecutedRound(proposals, predictions, commit)
        except Exception:
            self.failed = True
            raise
