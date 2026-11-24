# 9.3794 ms was measured, it was the automaton's, and its output copies 74 percent

2026-10-23. An earlier entry said this figure had no primary record. It has
one; it is not in this repository. It lives on a3-84 at
`/data/shanshan/trt-asc/workspace/experiments/lever84-20261020/auto12`, and
searching only `docs/evidence` was the mistake.

| measured request | TPOT | accepted a round | distinct | in prompt |
| --- | --- | --- | --- | --- |
| 1 | 9.4007 ms | 5.441 | 55 | 0.744 |
| 2 | 9.3879 | 5.441 | 55 | 0.744 |
| 3 | 9.4434 | 5.441 | 55 | 0.744 |

## It passes the oracle this session calibrated, by a wide margin

The share of an output's two-token contexts that occur in the prompt separates
sound arms from corrupted ones: 0.010 and 0.027 for the two arms known broken,
0.281 to 0.370 for every sound one. This record reads **0.744** -- it copies
the prompt twice as heavily as anything measured this session, which is what an
agentic coding model is supposed to do and why 5.441 accepted is earned rather
than an artefact.

## The drafter was the suffix automaton, not the chain

The record's command carries `--lookup --automaton`, and the tree that ran it,
`experiments/lookup-20261019/tree`, writes `report["drafter"] = "lookup"` as a
literal while passing `lookup_automaton=args.automaton` through to
`ascend_cache`, where `if self.lookup_automaton:` selects `AutomatonDrafter`.
So `11c06ea` -- "the best result is the lookup drafter's, and the suffix
automaton has never run in the engine" -- read a hard-coded field and is
wrong on both counts.

## What is identical to today's arms, and what is not

Ruled out by comparison, not by argument:

- the checkpoint's three JSON digests, and the prompt digest `fe53282d`
- the capture: the record's path differs but its manifest `created_at` is
  `2026-10-17T06:12:02.900444`, the same export
- the weights: the record's shards are symlinks into a `.partial` staging
  path, and the first shard's leading 100 MB hashes identically on both hosts
- the drafter: on this session's own stream the automaton gives 2.570 accepted
  against the chain's 2.590 at width 12, so it is not what separates 5.441
  from 2.590
- width, output tokens, warmups, capacity, `sfa`, and every graph and kernel
  switch

Left standing: **the code**. Of the 37 sources the record tracked, 9 differ
from today's tree, including `cann_model.py`, `mtp_round.py` and
`cann_graph.py`, all on the decode path. Its engine was a pip build, vllm
0.30.1rc1.dev447+gc4df37dfc with vllm-ascend 0.19.1rc2.dev2828+gf558b58c1, in
a venv inside a container whose writable layer is gone; the `src` trees on
a3-84 are older than that build and their `GlmMoeDsaForCausalLM` has no `rot`,
which the pinned `ascend-python-build` does have.

So the reproduction runs the record's own tree against the pinned vendor trees.

## Correction: the in-prompt share is fooled by a loop, and both fast arms loop

The oracle above is wrong, and it is wrong in the direction that flatters a
fast reading. The share of an output's bigrams that occur in the prompt rises
when the output repeats a handful of bigrams that happen to be in the prompt --
which is exactly what a looping decode does.

The prompt is not repetitive: 8192 tokens, 1322 distinct, 4059 distinct bigrams
of 8191. Against that, bigram variety -- distinct bigrams over total -- sorts
the arms the other way:

| arm | bigram variety | distinct tokens | TPOT |
| --- | --- | --- | --- |
| `shared_fia` chain, width 32 | 0.371 | 188 | 29.148 ms |
| `sfa` chain, width 12 | 0.249 | 139 | 22.135 |
| the record, auto12 | **0.110** | 55 | 9.4007 |
| `shared_fia` chain, width 8 | **0.080** | 32 | 14.743 |

The record's output repeats token 9897 eighty times and 13865 seventy-two
times in 1024, and holds 113 distinct bigrams of 1023. So its 5.441 accepted a
round is a drafter reading a loop, not a drafter exploiting an agentic coding
model's copying -- and **this session's own best figure, 14.743 ms, loops
harder still**.

Both are therefore withdrawn as measurements of speed. What survives is the
ordering: TPOT falls as the output loops, because acceptance rises with
repetition, so any arm's TPOT is only comparable at equal output variety.

The gate this session added -- refuse an output with no more distinct tokens
than a round verifies -- does not catch this either: 32 distinct at width 8
passes, and 55 at width 12 passes. Bigram variety is the quantity that
separates them, and it belongs in the gate.
