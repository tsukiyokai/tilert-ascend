"""Batch-one greedy request lifecycle for the Ascend TP16 backend."""

from dataclasses import dataclass, field
import math

from tilert.pd_vllm.capabilities import (
    CapabilityUnavailable,
    InvalidParameter,
    engine_capabilities,
    validate_generation_request,
)
from tilert.pd_vllm.grammar_spec import GrammarUnsupported


@dataclass(frozen=True)
class PrefixRequest:
    """Identify rank-local prefix storage without carrying device tensors."""

    request_id: str
    seq_len: int
    metadata: dict = field(default_factory=dict)


def greedy_token(result):
    return int(result.logits.argmax(dim=-1).item())


class AscendPDEngine:
    """Adapt a coordinated TP16 backend to the existing PDEngine interface.

    ``first_token_id`` has already been produced by prefill. It is emitted
    before any step and remains outside main KV until a successor is needed.
    Decode retains KV for inspection; the request owner calls ``reset``.
    """

    def __init__(self, backend, stop_ids=(), select_token=greedy_token):
        self.backend = backend
        self.stop_ids = frozenset(stop_ids)
        self.select_token = select_token
        self.last_stats = {}
        self._request = None
        self._decoding = False
        self._finished = False

    def inject(self, request):
        if self._decoding:
            raise RuntimeError("A decode request is active")
        if request.seq_len < 0 or request.seq_len > self.backend.capacity:
            raise ValueError("Prefix exceeds the decode cache capacity")
        self.backend.inject(request)
        if self.backend.position != request.seq_len:
            raise RuntimeError("Injected prefix and backend position differ")
        self._request = request
        self._finished = False
        self.last_stats = {}

    def prepare_grammar(self, grammar_spec, enable_thinking=True):
        if grammar_spec is not None:
            raise GrammarUnsupported("Ascend greedy decode has no grammar")
        return None

    def supports_logprobs(self):
        return False

    def supports_penalties(self):
        return False

    def supports_ignore_eos(self):
        return True

    def _sampling(self, sampling):
        sampling = sampling or {}
        validate_generation_request(sampling, engine_capabilities(self))
        value = sampling.get("temperature")
        temperature = 0.0 if value is None else float(value)
        if not math.isfinite(temperature):
            raise InvalidParameter("temperature must be finite")
        top_k = sampling.get("top_k")
        if temperature >= 1e-5 and (top_k is None or int(top_k) != 1):
            raise CapabilityUnavailable("Ascend decode uses greedy sampling")
        return bool(sampling.get("ignore_eos", False))

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
        self._decoding = True
        tokens, calls = [], 0
        initial_position = self.backend.position
        reason = "length"
        token = int(first_token_id)
        # The final output token has not entered KV, so it needs no cache row.
        budget = min(max_tokens, self.backend.capacity - initial_position + 1)
        try:
            while len(tokens) < budget:
                if cancel_event is not None and cancel_event.is_set():
                    reason = "cancelled"
                    break
                if not ignore_eos and token in self.stop_ids:
                    reason = "stop"
                    break
                tokens.append(token)
                if on_token is not None:
                    on_token(token)
                if len(tokens) == budget:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    reason = "cancelled"
                    break
                result = self.backend.step(token)
                calls += 1
                if self.backend.position != initial_position + calls:
                    raise RuntimeError("Each step must consume one token")
                token = self.select_token(result)
        except Exception:
            reason = "error"
            raise
        finally:
            self._decoding = False
            self._finished = True
            self.last_stats = {
                "finish_reason": reason, "step_calls": calls,
                "output_tokens": len(tokens),
                "initial_position": initial_position,
                "committed_position": self.backend.position,
            }
        return tokens

    def reset(self):
        if self._decoding:
            raise RuntimeError("Cancel and finish the request before reset")
        self.backend.reset()
        self._request = None
        self._finished = False
        self.last_stats = {}
