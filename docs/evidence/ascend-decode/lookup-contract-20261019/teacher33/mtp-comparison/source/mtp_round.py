"""Greedy MTP rounds over the main-model and MTP CANN interfaces.

A round issues every device call before the host reads anything. Proposal
tokens travel as whatever the engines produce — device scalars for the real
engines — so the chain, the verification and the conditioned MTP pass need
no synchronisation between them. One call to the target's ``host_tokens``
then delivers the proposals and predictions that committing requires.
"""

from dataclasses import dataclass

from .mtp_state import MtpState


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


class MtpRoundExecutor:
    """Run fixed-width greedy rounds without importing a device library.

    ``target.verify`` and ``draft.forward`` return greedy ``token_ids``.
    Main ``last_hidden`` and MTP ``recycle_hidden`` are normalized states.
    All sixteen ranks call this loop in the same order. A failed device
    round requires fresh prefix injection before starting another request.
    """

    def __init__(
        self, target, draft, width=4, max_new_tokens=1024, eos_tokens=(),
        drafter=None,
    ):
        if width not in (2, 4, 6, 8, 12, 16, 24, 32):
            raise ValueError("MTP rounds require width 2, 4, 8 or 16")
        if target.capacity != draft.capacity:
            raise ValueError("Main and MTP cache capacities must match")
        self.target = target
        self.draft = draft
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

    def start(self, first_token, normalized_parent_hidden):
        """Start at main length P with MTP length P-1 or P.

        The first token was produced by prefill. Its normalized parent
        hidden is from the last prompt position, P-1. Recomputing that MTP
        row supplies the first draft token, recycle state and index set.
        """
        self.state.reset(self.target.position, first_token,
                         draft_prefix_length=(self.draft.position
                                              if self.drafter is None
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

    def _advance_drafted(self, plan):
        """One round whose candidates come from the host, not the MTP engine.

        The round is a verify and a host read. A proposal shorter than the
        width is padded with the pending token, which is safe rather than
        lucky: whatever is proposed, what gets committed is the target's own
        predictions, so a pad that happens to match earns a real token.
        """
        try:
            proposal = self.drafter.propose(plan.width - 1)
            candidates = [plan.pending_token, *proposal[:plan.width - 1]]
            candidates += [plan.pending_token] * (plan.width - len(candidates))
            verified = self.target.verify(candidates, full_logits=False)
            proposals, predictions = self.target.host_tokens(
                candidates, verified.token_ids,
            )
            commit = self.state.resolve(plan, proposals, predictions)
            self.target.truncate(commit.main.retained)
            self.drafter.commit(commit.accepted_tokens)
            return ExecutedRound(proposals, predictions, commit)
        except Exception:
            self.failed = True
            raise
