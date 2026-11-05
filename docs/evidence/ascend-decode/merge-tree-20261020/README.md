# Merged chain-and-lookup tree, and the longest tail

Offline replay of one frozen 8192+1024 trajectory, acceptance only. No device
run, no TPOT claim: the round-cost extrapolations of that shape are withdrawn
pending a device measurement of the actual expert union.
[Bound result and input hashes](result.json),
[harness](../../../../tests/glm52/merge_acceptance.py).

## Two conventions the replay has to mirror

A round's pending token is the last token already committed: `resolve` sets
`pending_token = accepted[-1]` and `accepted = predictions[:count]`, so the
emitted stream is the target's predictions and the drafter's context ends at
the pending. `propose` therefore matches the tail ending at the pending with
nothing passed in, and the deployed call is already correct. A replay whose
context lags one token behind keys on the wrong tail.

A round must then advance by what it accepted. Scoring a configuration at the
*chain's* round starts instead counts one long repeat once per boundary it
spans; at the chain's 4.507 tokens a round that inflated k=31 lookup
acceptance from 5.879 to 11.304.

The harness reproduces three independent anchors on the same trajectory: the
measured chain's 4.507, and the CPU replay's 5.8793 and 6.9592 at k=31 for
minimum tail 2 and 1. [Replay these figures](../lookup-contract-20261019/README.md)

## The merged tree is not established

A merged tree offers the chain's 7 and the lookup's k continuations from one
point and keeps whichever runs further. The chain's match length is only on
record at the boundaries the measured run visited, so a merged walk has to
deal those lengths independently of position — the assumption `plan.md`
already records as unsupported. Dealing the same 227 measured lengths in four
orders gives, at minimum tail 1 and k=31:

| Deal order | Merged acceptance | Against lookup alone (6.959) |
| --- | --- | --- |
| ascending | 8.317 | 1.195 |
| recorded | 10.129 | 1.455 |
| reversed | 11.494 | 1.652 |
| descending | 13.286 | 1.909 |

The merged round keeps the six draft forwards and the conditioned pass that a
host drafter removes, 12.35 of the measured 53.05 ms round. Its round is
therefore at least `53.05/40.70 = 1.303` times the lookup-only round, and the
seven extra verify rows move that ratio by 0.001 — the break-even rests on the
chain's measured 23.3% share, not on any row extrapolation.

So the merge must raise acceptance by 30.3% to break even, and the deal order
alone moves the gain between 19.5% and 90.9%. The lower end loses. The pairing
measured at the chain's own boundaries is `r = +0.358`: the two branches are
positively correlated, which lowers the expected maximum and pushes the true
value toward that losing end. The 67 rounds where the lookup finds nothing
carry a mean chain match of only 2.418.

Defer the tree and its attention mask. Reopen it with a mechanism that
supplies the chain's match at positions the chain never visited, not with
another deal of the same 227 numbers.

## The longest tail was already optimal

`longest` has never been swept. It is not a free parameter to raise: the
search takes the longest match first, so a longer maximum prefers a stale long
repeat over a fresher short one. k=31, lookup alone, acceptance:

| longest | minimum tail 1 | minimum tail 2 |
| --- | --- | --- |
| 4 | 6.912 | 5.846 |
| 8 | 6.959 | 5.879 |
| 12 | 6.912 | 5.846 |
| 16 | 6.912 | 5.846 |
| 24 | 6.820 | 5.780 |
| 32 | 6.730 | 5.715 |
| 64 | 6.600 | 5.621 |

Flat from 4 to 16 and monotonically worse beyond. Keep 8 and stop sweeping it.
One trajectory, so this settles the direction, not a held-out optimum.
