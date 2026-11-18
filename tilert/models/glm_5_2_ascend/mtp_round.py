from dataclasses import dataclass

from .mtp_state import VOCAB_SIZE, MtpState

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
        self, target, draft, width=4, max_new_tokens=1024, eos_tokens=(),
        drafter=None, phases=None,
    ):
        if width not in (2, 4, 6, 8, 12, 16, 24, 32):
            raise ValueError("MTP rounds require width 2, 4, 8 or 16")
        if draft is not None and target.capacity != draft.capacity:
            raise ValueError("Main and MTP cache capacities must match")
        self.target = target
        self.draft = draft
        self.phases = phases
        self.drafter = drafter
        self.state = MtpState(
            width, target.capacity, max_new_tokens, eos_tokens
        )
        self.seed = None
        self.seed_parent_hidden = None
        self.failed = False
        self.tree_mask = None

    def start(self, first_token, normalized_parent_hidden):
        self.state.reset(self.target.position, first_token,
                         draft_prefix_length=(self.draft.position
                                              if self.drafter is None
                                              and self.draft is not None
                                              else None))
        self.seed_parent_hidden = normalized_parent_hidden
        self.seed = None
        self.failed = False

    @staticmethod
    def _seed(result, row):
        return DraftSeed(
            result.token_ids[row].clone(),
            result.recycle_hidden[row:row + 1], result.indices[row:row + 1],
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
                    (plan.pending_token,), self.seed_parent_hidden,
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
                        [seed.token], seed.recycle_hidden,
                        full_logits=False, external_indices=shared_indices,
                    )
                    seed = self._seed(chained, 0)
            verified = self.target.verify(candidates, full_logits=False)
            self.draft.truncate(plan.main_start)
            conditioned = self.draft.forward(
                verified.token_ids, verified.last_hidden, full_logits=False,
            )
            proposals, predictions = self.target.host_tokens(
                candidates, verified.token_ids,
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
        from .tree_mask import branch_depths
        from .tree_resolve import accepted_path

        mark = self.phases.mark if self.phases is not None else None
        tokens, parents = (propose_tree(plan.width - 1) if plan.width > 1
                           else ([], []))
        if mark:
            mark("propose")
        nodes = [plan.pending_token, *tokens]
        lineage = [-1, *(parent + 1 for parent in parents)]
        nodes += [plan.pending_token] * (plan.width - len(nodes))
        lineage += [0] * (plan.width - len(lineage))
        if self.tree_mask is not None:
            self.tree_mask(lineage, plan.main_start, self.target.device)
        verified = self.target.verify(nodes, full_logits=False,
                                      depths=branch_depths(lineage))
        if mark:
            mark("verify")
        proposals = tuple(nodes)
        predictions, = self.target.host_tokens(verified.token_ids)
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
            candidates = [plan.pending_token, *proposal[:plan.width - 1]]
            candidates += [plan.pending_token] * (plan.width - len(candidates))
            verified = self.target.verify(candidates, full_logits=False)
            if mark:
                mark("verify")
            proposals = tuple(candidates)
            predictions, = self.target.host_tokens(verified.token_ids)
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
