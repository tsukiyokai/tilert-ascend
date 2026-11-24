# Removing a synchronisation removed a snapshot, and only acceptance noticed

`a3-84` idle, TP16, the composed flag set with `HCCL_OP_EXPANSION_MODE=AIV`,
`--max-tokens 1024 --warmup 3 --requests 3`, profiler off. The change under
test: proposal tokens travel as device scalars so a round issues every kernel
before the host reads anything
([commit](../../../../tilert/models/glm_5_2_ascend/mtp_round.py)).

| case | TPOT median (ms) | rounds | A | round (ms) | tokens vs reference |
| --- | --- | --- | --- | --- | --- |
| reference (host ints) | 14.540 | 298 | 3.433 | 49.92 | — |
| device tokens, chunk 1 | 24.504 | 492 | 2.079 | 50.74 | identical |
| device tokens, chunk 8 | 23.972 | — | — | — | identical |
| device tokens, chunk 32 | 23.622 | — | — | — | identical |

Three facts together name the defect. The output tokens are bit-identical to
the reference, so correctness held. The round's wall time is unchanged at
50.74 ms against 49.92, so nothing got slower. But acceptance fell from 3.433
to 2.079 and the request needed 492 rounds instead of 298. The regression is
entirely in proposal quality.

## The cause

`--draft-graph` means `CannGraph.forward` returns the graph's own persistent
output buffers — its docstring says so: "Returned hidden storage is reused on
replay." The previous `_seed` read `token_ids` through `.tolist()`, which
incidentally took a host snapshot. Replacing that with a device view
(`result.token_ids[row]`) left the chain holding aliases:

1. replay one writes the token buffer, `_seed` keeps a view of it
2. the view is appended to `candidates`
3. replay two **overwrites the same buffer**, so the candidate already
   appended now reads the newer token

By verification time the earlier candidates hold later proposals, so the
drafts mismatch and acceptance collapses. The target still verifies from
`candidates[0]`, the pending token, which is a host integer — which is
exactly why the delivered tokens stayed correct and the gate passed.

The fix is one device clone per proposal, `result.token_ids[row].clone()`:
a one-element copy, no host transfer, snapshot restored.

## The host suite could not see it, and now can

Every scalar double returned fresh Python lists of immutable ints, so they
were accidentally immune to the contract the real engines have. The round
suite gained `Cell`, a mutable token slot whose only detach is `clone()`, and
`RecycledDraft`, which refills the same cells on every forward. The new case
runs a plain and a recycled draft to completion and requires equal output
**and equal round counts**.

Polarity checked both ways on the same binary: with the clone removed the
case fails 2 subtests; with it restored, 10/10 pass. A test that only passes
is not evidence it would have caught anything.

## The fixed build, and a retraction

With the clone restored, every gate holds and the change is a win:

| case | TPOT median (ms) | three runs | rounds | A | tokens vs reference |
| --- | --- | --- | --- | --- | --- |
| reference | 14.540 | — | 298 | 3.4329 | — |
| chunk 1 | **14.247** | 14.442 / 14.247 / 14.190 | 298 | 3.4329 | identical |
| chunk 8 | 14.255 | 14.115 / 14.385 / 14.255 | 298 | 3.4329 | identical |
| chunk 32 | 14.624 | 14.602 / 14.645 / 14.624 | 298 | 3.4329 | identical |

Removing three of the round's four host synchronisations is worth
**0.293 ms per token**, with rounds, acceptance and output all unchanged.

Batching rounds per command is dead, and the earlier claim derived from it is
withdrawn. On the fixed build chunk 8 matches chunk 1 and chunk 32 is 0.38 ms
worse. The arithmetic is worse than it looks: `tpot_ms` is
`(last − first) / intervals`, and batching delays the first delivery to the
end of the first batch, so chunk 32's span covers rounds 33–298 (266 rounds)
against chunk 1's 2–298 (297). Equal per-round cost would put its span at
0.896 × 14575 ≈ 13.06 s; it measured 14.96 s. Per round that is 56.24 ms
against 49.07, a **14.6% regression**, and all three cases do identical work
— 298 rounds, 298 `round_calls`, 1024 tokens.

So the earlier "the per-round host command costs roughly 1.9 ms" is retracted:
it came from the buggy build, where acceptance had collapsed to 2.079 and the
request needed 492 rounds, and it does not reproduce once the proposals are
correct. The per-round Gloo broadcast is cheap; what was expensive were the
`tolist` synchronisations, which is why removing those paid and removing the
broadcast does not. `rounds_per_command` stays at its default of 1.

This also agrees with the independent audit, which attributes the round's host
serialisation to `aclrtSynchronizeStream` (24.255 ms over 15 calls) and
`tolist`, not to the broadcast
([audit](../stack-selection-20261018/README.md)).
