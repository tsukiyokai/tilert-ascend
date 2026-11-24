"""Greedy, batch-one MTP control for verification widths 1, 2, 4 and 8.

The main cache stores consumed tokens. Its pending token has already
been emitted but has not been consumed by the main model. MTP row p
uses target hidden[p] and the embedding of token[p+1], while retaining
position p for RoPE and the cache. Equal main/MTP cache lengths thus
allow the MTP cache to include the pending token's embedding.
Chained draft hidden states must not survive as target-conditioned KV.

Plans describe logical writes and truncation. The caller executes those
writes, then resolves the round. No tensors or device APIs are imported.
"""

from dataclasses import dataclass


VOCAB_SIZE = 154880
WIDTHS = (1, 2, 4, 6, 8, 12, 16, 24, 32)


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
    """Logical lengths; discard is a half-open range, not a memory erase."""

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
    """Own one request's greedy control state, independently of KV storage.

    reset() counts first_token as the first output when the budget is
    nonzero. A host drafter leaves draft_prefix_length unset. For MTP it
    must be P-1 or P. P-1 requires
    a seed from target hidden[P-1] and first_token at MTP position P-1;
    P asserts that this target-conditioned seed already exists.

    During verification every planned MTP row must be recomputed from
    target hidden and target predictions, even when a chained draft row
    contains the same token. resolve() does not inspect tensor contents.
    """

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
        """Reserve one verification round without advancing valid lengths."""
        if self.finish_reason is not None:
            raise ValueError(f"request cannot decode: {self.finish_reason}")
        if self._active is not None:
            raise ValueError("a verification round is already active")
        room = self.capacity - self.main_length
        use_draft = self.draft_length is not None
        width = max(value for value in WIDTHS
                    if value <= min(self.width, room))
        draft_start = self.main_length if use_draft else None
        needs_seed = use_draft and self.draft_length == self.main_length - 1
        plan = RoundPlan(
            self.epoch, len(self.history) + 1, width, self.pending_token,
            self.main_length, self.draft_length,
            self.main_length - 1 if needs_seed else None, draft_start,
        )
        self._active = plan
        return plan

    def _require_active(self, plan):
        if plan is not self._active or plan is None:
            raise ValueError("round is stale or is not active")

    def resolve(self, plan, candidates, target_predictions):
        """Commit a matching prefix and its correction or bonus token.

        target_predictions[i] follows candidates[i]. Each MTP verify row
        uses that prediction with target hidden[i], not draft hidden[i].
        EOS and the output budget can shorten the natural accepted span.
        draft_hidden.row also selects the next MTP draft prediction.
        """
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
            plan.number, plan.width, matched, natural, accepted,
            candidates[:count], self.pending_token, main, draft,
            HiddenSource("target", count - 1, main.retained - 1),
            draft_hidden, self.finish_reason,
        )
        self.history.append(result)
        self._active = None
        return result

    def abort(self, plan):
        """Return truncations for a failed round; preserve request state."""
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
        """Return logical work counts, without timing or performance claims."""
        committed = sum(item.committed for item in self.history)
        rounds = len(self.history)
        return {
            "rounds": rounds,
            "verified_positions": sum(item.width for item in self.history),
            "matched_drafts": sum(
                item.matched_drafts for item in self.history
            ),
            "natural_accepted": sum(
                item.natural_accepted for item in self.history
            ),
            "committed": committed,
            "emitted": self.emitted_count,
            "mean_committed_per_round": committed / rounds if rounds else 0,
        }
