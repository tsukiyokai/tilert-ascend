# What a MIX kernel's structure costs, from the model's own operators

Three attempts at a rendezvous benchmark hung on the flag protocol, so this
reads the price off the capture instead. Both kinds are already there:
operators that run on one engine, and operators that carry vector subblocks
alongside a cube block.

## The floor of each kind

The cheapest call of a kind bounds the fixed cost that kind pays, because no
call can be cheaper than the structure it carries.

| kind | calls | min | p10 | median |
| --- | --- | --- | --- | --- |
| `AI_VECTOR_CORE` | 6151 | **0.72 µs** | 0.94 | 3.98 |
| `AI_CORE` | 1001 | 2.34 | 8.28 | 12.62 |
| `MIX_AIV` | 718 | **13.56** | 18.20 | 22.93 |
| `MIX_AIC` | 1661 | 14.20 | 16.28 | 41.94 |

A single-engine kernel can cost 0.72 µs; a MIX kernel cannot go below 13.56.
**The difference, 12.84 µs, is a lower bound on what the cross-core structure
costs**, and over 595 MIX kernels a verify it is **7.64 ms — 21% of the
36.02 ms fixed cost**.

A check on the method: `AI_VECTOR_CORE` at one block has a median of 1.20 µs,
which is the in-graph dispatch floor this project measured independently. Two
routes, one number.

## Reproduced by `tests/glm52/kernel_accounting.py`, with a small difference

The script reads the same capture and gives slightly lower floors:
`AI_VECTOR_CORE` 0.70 against 0.72, `AI_CORE` 2.30 against 2.34, `MIX_AIV`
**12.04 against 13.56**, `MIX_AIC` 13.58 against 14.20. The first query
grouped by `Block Num` and `Mix Block Num` as well as the core, so it took the
cheapest call of each shape of each kind rather than the cheapest of the kind.

That moves the structure's lower bound from 12.84 µs to **11.34**, and the
MIX total from 7.64 ms to **6.75** — 19% of the fixed cost rather than 21%.
The script's figure is the correct one for the claim as stated, since the
bound is over the kind and not over a shape within it.

## The fixed cost splits three ways

| | a verify | share |
| --- | --- | --- |
| the 595 MIX kernels' structure | 7.64 ms | 21% |
| the 1788 single-engine kernels at their floors | 1.69 | 5% |
| not yet attributed | 26.69 | 74% |

So this morning's "the fixed cost is the kernel count" holds for about a
quarter of it. The rest is what these kernels spend above their own floor —
`MIX_AIC` has a minimum of 14.20 and a median of 41.94, `AI_VECTOR_CORE` 0.72
against 3.98.

## A comparison that does not work, and why

`QuantBatchMatmulV3` appears as `MIX_AIC 24/24` at a median of 18.06 µs and as
plain `AI_CORE 4/0` at 11.64 — the same operator in both structures, which
looked like a controlled pair. It is not: the single-engine form runs
`8,6144 × 8,384,16,32 → 256` and the MIX form `8,1024 × 192,64,16,32 → 6144`.
Different projections, so the 55% gap confounds structure with shape.

## And a correction to the per-kernel figure

19.30 µs a kernel was 36.02 divided by 1867, so "1867 × 19.30 accounts for all
of it" is an identity. I used it to argue the rendezvous must be near zero;
the floors above say it is 21% of the fixed cost. What was measured
independently is a bare launch at 20–30 µs regardless of the body, and an
in-graph floor of 1.2 — and 19.30 falling between them makes the kernel-count
story consistent, not exclusive.
