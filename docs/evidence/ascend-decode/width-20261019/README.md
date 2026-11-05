# Eight draft tokens per round, and one literal that was holding it back

TPOT falls from 14.485 ms at width 4 to **13.338 ms at width 8** (repeat
12.894), which is 7.9–11.0%; against the project's best recorded 14.247 it is
6.4–9.5%. Width 16 is worse at 14.037, so 8 is the optimum. Measured at 16
ranks inside the ACL graph on one frozen prefix, 1024 output tokens, three
warmups and three measured requests, median of the measured requests, with
five runs emitting bit-identical token streams.

## The enabling change is one literal

`cann_model.py:80` read `max_num_seqs=4 if super_kernel else 1`, and its own
comment says why it matters: AscendCompiler filters static token buckets by
that sequence limit. So the compiled static kernels only ever covered four
rows, and `--compile-target --super-kernel` — worth 8.9 ms per round at width
4 — was worth nothing at width 8.

Changing it to `max(4, graph_width)` is an identity transform at width 4, so
the baseline does not move. With it, width 8's round falls from 68.443 to
59.014 ms (repeat 57.366), acceptance and the output token stream unchanged
bit for bit, and TPOT from 15.408 to 13.338. Width 6 likewise goes 15.384 to
13.673.

`max_num_batched_tokens=16` is the real ceiling above that: it sizes the
indexer's RoPE buffers (`indexer.py:559-560,908-912`, the only capacity on the
draft path that raises), its slot and mask buffers, `_reserved_mc2_mask`, and
both `topk_indices_buffer` allocations. With `max(16, graph_width)` and the
matching assertion in `cann_mtp.py:219-221` reading the scheduler config,
width 32 runs.

## Acceptance against width

A(4)=3.3322 over 307 rounds, A(6)=4.1250 over 248, A(8)=4.5066 over 227, all
on one token stream with bit-identical output across repeats. The margin
decays fast: 4→6 buys 0.396 tokens per extra position, 6→8 only 0.191.

Extrapolating from width 4's per-position rates (0.9161, 0.8864, 0.8719) with
a flat tail predicts A(8)≈5.47 against the measured 4.51, because a wider
round consumes a whole run of easy predictions at once and the per-round
statistic is then dominated by hard starts.

**Width changes the trajectory.** Widths ≤6 and ≥8 diverge at token 4, so
cross-width acceptance comparisons hold only within a stream group. Changing
trajectory at fixed width matters far less: width 8 under `gather_fia` gives
4.1417 and 4.1585 on two streams, a 0.4% spread against the 35% that 4→8 buys.

**A caution on the prefix.** This frozen prefix's greedy continuation is
partly degenerate: the width-16 trajectory has period-13 self-similarity 0.914
and only 131 distinct values in 1024 tokens, with 24% of rounds piled at 12
matched drafts. Its A=6.6429 is largely a memorised 13-token loop and must not
be extrapolated. Other trajectories are much less repetitive (period 17 at
0.826 down to period 35 at 0.057), but acceptance tracking repetitiveness is a
risk for any A calibrated on a single prefix.

## Where a wider round spends

Single die, device 7, eager, 20 calls per batch and 20 batches, median:

Attention barely moves with rows. `npu_sparse_flash_attention` at 4 heads,
top-k 2048, 8K paged MLA: 89.51, 89.95, 90.96, 91.95, 95.76 µs at width
1/2/4/8/16. Sixteen times the query rows and sixteen times the selected rows
cost 7.0% more — 0.42 µs per unit of M — while the read grows from 2.36 to
37.75 MB, 26 to 394 GB/s. The same rows read by `index_select` alone take
28.82 to 86.03 µs, flat to 4 rows then linear, so SFA's inner loop hides that
read all the way to 16. The task's expectation that attention scales with rows
is refuted.

Dense projections are flat: `npu_quant_matmul` at `[rows,2048]×[2048,6144]`
in FRACTAL_NZ with 16 rotating banks gives 24.04 to 25.12 µs across rows
1 through 64.

The expert GMM ignores rows and sees only the number of activated groups.
`gate_up` at 27 groups costs 77.16, 84.49, 81.20, 81.30 µs for 1/2/4/8 rows
per group — flat — while at one row per group, 27 to 64 groups costs 77.16 to
107.50, fitting 0.820 µs per extra group on a 55.0 µs intercept.

So M arrives through the expert union, captured from real TP16 rounds over 75
routed layers: E(4)=25.512, E(8)=43.434, E(16)=71.030, with a self-consistent
control that takes only the first four rows and recovers 25.2–25.5 in all
three runs.

## The round's cost model, and the item it exposes

round(4)=47.093 ms, round(6)=55.651, round(8)=59.014 and 57.366 — about
2.774 ms per unit of M, decomposed:

| term | ms per unit M |
| --- | --- |
| attention, 0.42 µs × 78 layers | 0.033 |
| dense projections | ~0 |
| expert GMM via the union, 0.820 µs × 4.48 groups × 2 × 75 | 0.55 |
| **one more chained M1 draft** | **2.19** |

The residual has structural corroboration: the draft graph's replay count is
exactly (M−2) × rounds × requests — 8892 = 6×247×6 at M=8, and
15372 = 14×183×6 at M=16.

**So width is nearly free on the hardware, and what costs is the choice to
make the draft a serial chain.** Turning M proposals into one parallel draft
would remove 2.19 ms per unit of M outright. That is a larger item than
anything width itself buys, and it was not on the board before.

## The weight floor moves the wrong way

floor(E) = 1.469 + 0.0773·E ms per round per card gives 3.441, 4.826 and
6.960 ms at E(4), E(8), E(16). Folded through each width's acceptance that is
1.033, 1.071 and 1.299 ms per delivered token, or 52%, 54% and 65% of the 2 ms
budget. Width buys amortisation of fixed overhead and pays with a larger share
of the incompressible part, which cuts against the main line of pressing a
round down toward its weight-stream time.

## A correctness defect that had to be fixed first

Correction after source audit: the serial/speculative failure below remains
observed, but its cause is unresolved. The production lightning indexer uses
`sparse_mode=3` with per-query causal lengths; no source boundary exists at
width six. The saved failure counts do not prove that any early query selected
a future token. The width gate remains while that cause is isolated.

`--attention-kernel gather_fia` stops being equivalent to serial decoding past
width 6. Three-phase serial/speculative/reset comparison with
`run_cann_mtp_rounds.py`, 32 tokens, `--ignore-eos`, no graph, same prefix and
binary: width 4 passes with zero failing checks, width 6 passes, **width 8
fails** — output diverges from serial at token 4 and 941 bit-wise KV checks
fail. Dropping `--fp32-head` does not save it, only moves the divergence to
token 20. The same widths under `sfa` pass completely, and width 16 differs
only in one INT8 index key row by a single quantisation step.

`cann_attention.py`'s `gathered_fia` gathers
`width × 2048` rows and calls `npu_fused_infer_attention_score_v2` with
`sparse_mode=0`, `actual_seq_qlen=[1]*width` and
`actual_seq_kvlen=[2048]*width`, without an additional mask. Correctness
depends on valid causal selected IDs and matching logical/physical cache
rows. The production indexer's causal mode supports that contract; the saved
failure counts alone do not locate a violation or distinguish it from
numerical drift. The first failing query's selected IDs and operands are
needed to identify the cause.

`install_gather_fia` now takes the width and refuses past
`GATHER_FIA_MAX_WIDTH = 6`, and the docstring records what was measured
instead of what was assumed. The recorded +0.101 acceptance from `gather_fia`
at width 4 still stands, since width 4 is bit-equivalent.

## Two shared-state hazards

The shared AOT compile cache at `/root/.cache/vllm/torch_compile_cache` can
hold a truncated entry that makes a later run fail with `execution_fn` of
None, and entries from concurrent runs were found in it. Width experiments
should use a private `VLLM_CACHE_ROOT`.

And `bench_cann_mtp.py` rank 0 raises `ValueError: Delivery clocks and session
acceptance differ` on every run including the project's own reference, so
`summary.json`'s `status: fail` is a pre-existing clock assertion rather than a
failed measurement. Worth fixing so that a real failure is visible.
