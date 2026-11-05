"""Greedy MTP delivery and cache retention for one Ascend PD request."""

from dataclasses import dataclass, field

from tilert.pd_vllm.capabilities import (
    CapabilityUnavailable,
    InvalidParameter,
)

from .session import AscendPDEngine


@dataclass
class _Output:
    budget: int
    stop_ids: frozenset
    callback: object
    cancel_event: object
    tokens: list[int] = field(default_factory=list)
    reason: str | None = None

    def check_cancelled(self):
        if (self.reason is None and self.cancel_event is not None
                and self.cancel_event.is_set()):
            self.reason = "cancelled"
        return self.reason is not None

    def deliver(self, token):
        if self.check_cancelled():
            return
        if token in self.stop_ids:
            self.reason = "stop"
            return
        if self.callback is not None:
            self.callback(token)
        self.tokens.append(token)
        if len(self.tokens) == self.budget:
            self.reason = "length"


class AscendMtpPDEngine(AscendPDEngine):
    """Deliver verified tokens and trim every started MTP request.

    The first token comes from prefill and counts toward ``max_tokens``.
    Each backend round completes on every rank before token delivery.
    ``accepted`` counts tokens accepted by completed rounds, including EOS
    and any tail withheld after cancellation. ``output_tokens`` counts
    returned token IDs or callbacks that completed successfully.
    """

    def __init__(self, backend, width=4, stop_ids=(), rounds_per_command=1):
        if (isinstance(width, bool) or not isinstance(width, int)
                or width not in (2, 4, 6, 8, 12, 16, 24, 32)):
            raise ValueError("MTP serving requires width 2, 4, 8 or 16")
        if (isinstance(rounds_per_command, bool)
                or not isinstance(rounds_per_command, int)
                or rounds_per_command < 1):
            raise ValueError("rounds_per_command must be a positive integer")
        super().__init__(backend, stop_ids)
        self.width = width
        self.rounds_per_command = rounds_per_command

    def inject(self, request):
        if request.seq_len < 1:
            raise ValueError("MTP serving requires a nonempty prefix")
        super().inject(request)

    def decode(
        self, first_token_id, max_tokens, sampling=None, on_token=None,
        cancel_event=None, grammar_session=None, top_logprobs=None,
    ):
        if self._request is None or self._finished:
            raise RuntimeError("Inject a new prefix before decode")
        if self._decoding:
            raise RuntimeError("A decode request is active")
        if (isinstance(max_tokens, bool) or not isinstance(max_tokens, int)
                or max_tokens < 0):
            raise InvalidParameter("max_tokens must be a nonnegative integer")
        if grammar_session is not None or top_logprobs is not None:
            raise CapabilityUnavailable("Greedy decode returns token IDs")
        ignore_eos = self._sampling(sampling)
        initial_position = self.backend.position
        budget = min(max_tokens, self.backend.capacity - initial_position + 1)
        stop_ids = frozenset() if ignore_eos else self.stop_ids
        output = _Output(budget, stop_ids, on_token, cancel_event)
        first_token = int(first_token_id)
        started, calls = False, 0
        failure, finish_error = None, None
        spans = []
        self._decoding = True
        try:
            if budget == 0:
                output.reason = "length"
            else:
                output.deliver(first_token)
            if not output.check_cancelled():
                self.backend.start_mtp(
                    first_token, max_tokens, self.width, tuple(stop_ids)
                )
                started = True
            while started and not output.check_cancelled():
                position = self.backend.position
                try:
                    results = self.backend.advance_mtp_n(
                        self.rounds_per_command
                    )
                except BaseException:
                    calls += 1
                    raise
                calls += len(results)
                if not results:
                    raise RuntimeError("MTP command completed no round")
                for result in results:
                    commit = result.commit
                    if (not commit.accepted_tokens
                            or commit.main.before != position
                            or commit.main.retained
                            != position + commit.committed):
                        raise RuntimeError("MTP accepted span and cache differ")
                    position = commit.main.retained
                    span = {
                        "width": commit.width,
                        "matched_drafts": commit.matched_drafts,
                        "natural_accepted": commit.natural_accepted,
                        "accepted": commit.committed,
                        "output_tokens": 0,
                        "main_start": commit.main.before,
                        "main_retained": commit.main.retained,
                        # Predictions beyond the first mismatch belong to
                        # the rejected branch, not the serial continuation.
                        "candidates": list(result.candidates),
                        "predictions": list(result.target_predictions),
                        **({"parents": list(result.parents)}
                           if result.parents is not None else {}),
                    }
                    spans.append(span)
                    before = len(output.tokens)
                    try:
                        for token in commit.accepted_tokens:
                            output.deliver(token)
                            if output.reason is not None:
                                break
                    finally:
                        span["output_tokens"] = len(output.tokens) - before
                    if (output.reason is None
                            and commit.finish_reason is not None):
                        raise RuntimeError(
                            "MTP stopped before a delivery limit"
                        )
                if self.backend.position != position:
                    raise RuntimeError("MTP accepted span and cache differ")
        except BaseException as error:
            failure = error
            output.reason = "error"
            raise
        finally:
            count = len(output.tokens)
            consumed = count if output.reason == "stop" else max(0, count - 1)
            retained = initial_position + consumed
            try:
                if started:
                    self.backend.finish_mtp(retained)
                    if self.backend.position != retained:
                        raise RuntimeError("MTP finish retained the wrong KV")
            except BaseException as error:
                output.reason = "error"
                finish_error = f"{type(error).__name__}: {error}"
                if failure is None:
                    raise
            finally:
                self._decoding = False
                self._finished = True
                self.last_stats = {
                    "finish_reason": output.reason,
                    "round_calls": calls,
                    "completed_rounds": len(spans),
                    "verified_positions": sum(item["width"] for item in spans),
                    "accepted": sum(item["accepted"] for item in spans),
                    "accepted_spans": spans,
                    "output_tokens": count,
                    "initial_position": initial_position,
                    "committed_position": self.backend.position,
                    "finish_error": finish_error,
                }
        return output.tokens
