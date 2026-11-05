"""Combine the shared and routed MoE outputs with a plain add.

vLLM-Ascend overrides upstream's inline combine for one reason: to route it
through `torch._foreach_add`, whose launch count does not grow with the tensor
list. It passes a list of one. At matched shapes in a single capture the
multi-tensor kernel costs 21.282 us against the plain add's 1.820, and its
cost is flat in row count -- 21.282 at one row, 21.442 at eight -- so all of
it is structure rather than work. Over 75 MoE layers that is 1.4 ms a round.

Both compute an elementwise add of two BF16 tensors, so the substitution is
bit-exact. The vendor's own comment retires the override once a superkernel
add exists; `enable_super_kernel` is on and the capture's plain `Add` runs on
AI_VECTOR_CORE.

Evidence: docs/evidence/ascend-decode/critical-path-20261020.
"""


def install():
    """Send one-element combines to a plain add; return how to undo it."""
    import torch

    original = torch._foreach_add

    def combine(tensors, others, **rest):
        if len(tensors) == 1 and len(others) == 1 and not rest:
            return [tensors[0] + others[0]]
        return original(tensors, others, **rest)

    torch._foreach_add = combine

    def restore():
        torch._foreach_add = original

    return restore
