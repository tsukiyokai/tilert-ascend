# Corrected: the hook measured dispatch, and the model forward is 43 ms

## The arithmetic settles what the hook could not

The verify's 2.196 ms is **below its own weight-read floor**. A round reads
5.985 GB a rank, which is 3.886 ms at the die's measured marginal rate, and
the verify reads nearly all of it — the six chain steps are one MTP layer
each. A forward cannot complete in less time than it takes to read the weights
it uses, so 2.196 is dispatch and nothing else, by a factor of 1.77.

Which makes the conditioned pass's 42.877 ms the device work it blocks on, not
its own cost. The round then decomposes as 42.9 model + 5.7 chain + about 4
commit = 52.5 against a p10 of 55.33.

**So the model forward runs at 11.0× its weight floor**, which is the same
10.4× the budget arithmetic gives for non-weight work, reached independently.

This reverses the conclusion below, written an hour earlier, that the verify
is not the problem and that plans aimed at the model forward were aimed at 4%
of the round. They were aimed at 78% of it. The megakernel argument stands.

The A/B that would have shown this — capturing the conditioned pass so it
cannot be eager — is running, and on this reading it will move the round
barely at all. Its first attempt measured nothing: the graph was captured at
the verify width and still never replayed, because `_graph` carried a second
hardcoded 4.

What follows was written before this correction.

---

# The full-model verify dispatches in 2.2 ms; one draft forward takes 42.9

Measured with `tests/glm52/phasehook`, a post-import hook, on the
configuration holding the best deterministic TPOT: width 8, the three gmm1
patches, cache fix on. No profiler — it adds host work to a device the host
feeds. 1363 rounds over six requests, reported as distributions because the
first version's means mixed warmup rounds with measured ones and had the
phases summing to 124.505 ms against a 57 ms round.

| phase | p10 | median | p90 | calls a round |
| --- | --- | --- | --- | --- |
| `chain` | 0.942 | 1.682 | 1.876 | 6.00 |
| `verify` | 2.196 | 2.366 | 2.510 | 1.00 |
| `conditioned` | **42.877** | 43.311 | 43.763 | 1.00 |
| `host_tokens` | 0.489 | 1.248 | 2.198 | 1.00 |
| `seed` | 5.326 | 5.480 | 2299.054 | 0.01 |

The floors times the calls sum to 51.239 ms against a round's p10 of 55.33, so
93% is accounted and about 4 ms is the commit, the truncates and the state
resolve.

**The target-conditioned draft pass is 84% of that.** One forward of one MTP
layer costs twenty times the full-model verify.

## Why it is that pass and not another

The six chain steps replay a shared index set — `cann_mtp.py:324` sets
`skip_topk = external_indices is not None` — and are captured in a width-1
graph. The conditioned pass selects its own indices and, until today, was
captured at a literal 4: `enable_graph` built `CannMtpGraph(self, 4)`
regardless of the verify width. A captured graph is replayed only when its
width matches, so at width 8 this pass has been running eager while every
other forward in the round replays.
[the fix](https://github.com/tsukiyokai/TileRT-Ascend/commit/6ee072e)

An MTP layer eager is on the order of a hundred operator dispatches, and the
census priced an operator's prologue at 8.3 to 16.4 µs, so tens of
milliseconds is the right order for that explanation.

## What this measurement cannot separate

The hook's premise was that each phase returns a value the host reads, making
each synchronous. That holds for a chain step, which reads `.token`, and for
`host_tokens`. It does **not** hold for `verify`: it returns device tensors,
so 2.196 ms may be dispatch alone and the conditioned pass that follows may be
absorbing the verify's device time while it blocks.

Both readings put the same action next, so the control was run rather than the
attribution refined: capture the conditioned pass at the verify width and
measure the round. If the pass is eager dispatch, the round should fall by
most of 42.9 ms. If instead it is absorbing the verify's device work, the
round will barely move.

Separating them properly needs a device synchronisation between the phases,
which perturbs the overlap it measures — worth doing only if the control comes
back ambiguous.

## What it already settles

The verify is not the problem. A full-model forward at 8 rows, 75 MoE layers
and the dense stack, against a whole-round weight floor of 3.886 ms, is not
where a 10.4× on non-weight work lives. Every plan on this board that treated
the model forward as the target — the layer-scale megakernel among them — was
aimed at 2.2 ms of a 55 ms round.
