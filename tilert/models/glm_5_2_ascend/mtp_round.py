"""Greedy MTP rounds over the main-model and MTP CANN interfaces.

A round issues every device call before the host reads anything. Proposal
tokens travel as whatever the engines produce — device scalars for the real
engines — so the chain, the verification and the conditioned MTP pass need
no synchronisation between them. One call to the target's ``host_tokens``
then delivers the proposals and predictions that committing requires.
"""

from dataclasses import dataclass

from .mtp_state import VOCAB_SIZE, MtpState


@dataclass(frozen=True)
class DraftSeed:
    token: object
    recycle_hidden: object
    indices: object


@dataclass(frozen=True)
class ExecutedRound:
    """Retain verified rows and their tree ancestry for receipt validation.

    Parents include the pending root (-1) and padded rows. Chains use None.
    """

    candidates: tuple[int, ...]
    target_predictions: tuple[int, ...]
    commit: object
    parents: tuple[int, ...] | None = None


class MtpRoundExecutor:
    """Run fixed-width greedy rounds without importing a device library.

    ``target.verify`` and ``draft.forward`` return greedy ``token_ids``.
    Main ``last_hidden`` and MTP ``recycle_hidden`` are normalized states.
    All sixteen ranks call this loop in the same order. A failed device
    round requires fresh prefix injection before starting another request.
    """

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
        # None unless a caller wants the round's steps timed.
        self.phases = phases
        # A host drafter replaces the MTP engine outright: no chain, no
        # conditioned pass, and the MTP cache is never written, so nothing may
        # read it while one is installed.
        self.drafter = drafter
        self.state = MtpState(
            width, target.capacity, max_new_tokens, eos_tokens
        )
        self.seed = None
        self.seed_parent_hidden = None
        self.failed = False
        # Set by `tree_attention.install` when a tree drafter is in use; a
        # linear drafter leaves it None and the attention keeps its indexer.
        self.tree_mask = None

    def start(self, first_token, normalized_parent_hidden):
        """Start at main length P with MTP length P-1 or P.

        The first token was produced by prefill. Its normalized parent
        hidden is from the last prompt position, P-1. Recomputing that MTP
        row supplies the first draft token, recycle state and index set.
        """
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
        # A graphed draft returns its persistent output buffers, so the next
        # replay would rewrite a proposal already placed among the candidates.
        # Snapshot the token on the device; the host still reads nothing.
        return DraftSeed(
            result.token_ids[row].clone(),
            result.recycle_hidden[row:row + 1], result.indices[row:row + 1],
        )

    def advance(self):
        """Propose, verify, recompute MTP parents, and commit one prefix.

        Accepted tokens include a terminating EOS when present; serving
        filters that ID before delivering text. Chained draft KV is never
        committed as target-conditioned KV, even for an accepted token.
        """
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
        """One round whose candidates form a tree rather than a chain.

        The attention the engine calls already selects, for each query row, the
        positions that row may see, so a tree needs no kernel change -- only
        the right index set, which is the committed prefix plus a node's own
        ancestors. `write` owns the buffer every layer reads from.
        """
        from .tree_mask import branch_depths
        from .tree_resolve import accepted_path

        tokens, parents = (propose_tree(plan.width - 1) if plan.width > 1
                           else ([], []))
        nodes = [plan.pending_token, *tokens]
        lineage = [-1, *(parent + 1 for parent in parents)]
        nodes += [plan.pending_token] * (plan.width - len(nodes))
        lineage += [0] * (plan.width - len(lineage))
        if self.tree_mask is not None:
            self.tree_mask(lineage, plan.main_start, self.target.device)
        verified = self.target.verify(nodes, full_logits=False,
                                      depths=branch_depths(lineage))
        proposals = tuple(nodes)
        predictions, = self.target.host_tokens(verified.token_ids)
        # The accepted path through a tree is itself a chain, so reordering the
        # verified rows along it lets `resolve` do the committing: its linear
        # match, the EOS check, the output budget and the cache commit all
        # apply unchanged. A parallel resolver would duplicate four rules to
        # restate one.
        path = accepted_path(nodes, lineage, predictions)
        along = [nodes[node] for node in path]
        predicted = [predictions[node] for node in path]
        # The chain pads with the pending token and lets a pad that happens to
        # match earn a real token. A tree cannot: a padding row is off the
        # accepted path, so its key was never moved into place, and crediting
        # it would commit a token whose cache row belongs to another branch.
        # Padding with a token the last prediction is not stops `resolve` at
        # the path's end by construction rather than by luck.
        bonus = predictions[path[-1]]
        along += [(bonus + 1) % VOCAB_SIZE] * (plan.width - len(along))
        predicted += [bonus] * (plan.width - len(predicted))
        commit = self.state.resolve(plan, along, predicted)
        # `resolve` committed the path as the chain it reordered, so the cache
        # has to agree: the path's rows become the rows it was told about.
        self.target.compact(plan.main_start, path)
        self.target.truncate(commit.main.retained)
        self.drafter.commit(commit.accepted_tokens)
        return ExecutedRound(proposals, predictions, commit, tuple(lineage))

    def _advance_drafted(self, plan):
        """One round whose candidates come from the host, not the MTP engine.

        The round is a verify and a host read. A proposal shorter than the
        width is padded with the pending token, which is safe rather than
        lucky: whatever is proposed, what gets committed is the target's own
        predictions, so a pad that happens to match earns a real token.
        """
        try:
            tree = getattr(self.drafter, "propose_tree", None)
            if tree is not None:
                return self._advance_tree(plan, tree)
            # A round measures 50.7 ms where its 78 super kernels account for
            # 16.4, so the step timings are what the record was missing.
            mark = self.phases.mark if self.phases is not None else None
            proposal = self.drafter.propose(plan.width - 1)
            if mark:
                mark("propose")
            if 0 < len(proposal) < plan.width - 1:
                proposal = (proposal * (plan.width // len(proposal) + 1))[:plan.width - 1]
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
