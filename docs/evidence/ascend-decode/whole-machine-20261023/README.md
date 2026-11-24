# The first gated whole-machine arms, and the 9.3794 ms baseline has no record

2026-10-23, a3-180, sixteen dies, 78 layers, W4A8C8, the 8K prefix capture
`prefix8k-v3/case-00`, `LookupDrafter` at width 12 -- the configuration the
project's documents name as its best.

| output tokens | accepted a round | round | TPOT |
| --- | --- | --- | --- |
| 256 | 1.364 | 57.90 ms | **42.46 ms** |
| 1024 | 2.59 | 57.33 | **22.13** |

Both arms ran on a machine whose every die was free, with the checkpoint on
the 686 MB/s export rather than the asset NFS, so weights load in about 80
seconds instead of an hour.

## The drafter is not at fault, and neither is the engine

Replaying this capture's own prompt and this arm's own output through
`LookupDrafter` on the CPU gives **1.364** accepted a round -- the engine's
figure to three decimals. The output is not degenerate either: 111 distinct
tokens of 256, and 100 of its 254 two-token contexts occur in the prompt. So
the engine executes the drafter faithfully and this fixture simply does not
copy enough for a chain drafter to live on.

## What this says about the recorded best

`9.3794 ms` at 5.441 accepted is cited by `handoff.md`, `plan.md` and
`lesson.md`, and **no rank record in this evidence tree holds it**. A scan of
every `*.json` under `docs/evidence` for a request with TPOT under 15 ms finds
only `rope-contiguous-20261018` at 14.52 to 15.57 ms -- width 4, 1024 tokens,
no host drafter. The width table in `width-and-denominator-20261022` is a model
anchored to "the engine's 5.441", not a measurement of it.

Until an arm reproduces it, the project's measured best on this capture is
**22.13 ms**, and the gap to 2.x is larger than the documents say. The
difference is acceptance, not the round: 57.33 ms a round here against the
50.73 ms those documents record is 13 percent, while 2.59 accepted against
5.441 is a factor of 2.1.

## Three wiring faults this arm found, all in code that had no caller

- `bench_cann_mtp.py` passed `lookup_automaton=` to an engine whose booleans
  had become a drafter registry. The repository was already correct; the
  host's tree was a stale copy, so the arm died on a name the source no longer
  uses.
- The registry handed `shortest=` to every drafter, and `TreeDrafter` has no
  such bound -- its expansion is governed by budget and branching. The registry
  now records how each drafter names its shortest matched tail, and None where
  it has none.
- `index_picks` took the `--index-topk` argument, which defaults to None while
  the config keeps the checkpoint's own 2048. The tree mask divides by it, so
  the first tree round raised `NoneType - int`. It now reads the effective
  value from the config.

The pass gate also refused both arms for a bookkeeping reason rather than a
measurement one: it expected `final_positions[1]` to be None under `--lookup`,
but a draft engine is absent only when `--layers` truncates the backbone -- a
host drafter leaves it in place and unused. The TPOT figures above are
unaffected; the gate now follows `--layers`.
