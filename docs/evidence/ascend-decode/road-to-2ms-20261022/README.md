> Retracted 2026-10-22, the same day. The narrowing row is void, not
> smaller: the vendor kernel skips an empty expert, so the engine is
> already at the narrowed point. The remaining rows are unaffected but
> the path no longer has a measured first step. The fourth step, making
> the latent and rope contiguous, is void too: the gather runs at
> 1094 GB/s and is flat in run length, see
> [`gather-runs-20261022`](../gather-runs-20261022/README.md).
> See [`bank-skip-20261022`](../bank-skip-20261022/README.md).

# The road recomputed at the real distinct-expert count

The 2026-10-21 table ended at 2.60 ms. Two of its inputs were wrong, and
correcting them moves the end to 4.838 ms. Neither correction comes from a new
measurement; both come from reading the existing ones against the right object.

## What the bank narrowing is worth

The lever was priced at 223 microseconds a layer, from `258.17 - 35`. The
258.17 was profiled at `inner` 1536 while the checkpoint's widths are 1024 and
2048, and the 35 assumed a round touches two of a rank's sixteen experts.

The count follows from the routing rather than from an assumption. A row picks
eight distinct experts of 256, so after M rows a given expert is untouched with
probability `(248/256)**M`, and a rank holds `256/16 = 16` experts:

| rows | distinct of 256 | predicted | a rank of 16 |
| ---: | ---: | ---: | ---: |
| 12 | 81 measured | 81.1 | 5.07 |
| 32 | 158 measured | 163.3 | 10.21 |

The measured column is the routed/shared timing fixture's own report of how
many distinct experts its inputs select. Uniform routing predicts it to within
one expert at twelve rows, so the fixture's distribution is the uniform one and
the analytic count can be used.

A bandwidth model reproduces both profiled points: cost is
`max(17 us, bytes / 1255 GB/s)`, giving 80.24 against 80.21 measured for the
gate/up call at 100.7 MB and 160.40 against 160.53 for the down call at
201.3 MB. At 5.07 of sixteen experts a layer's two calls fall from 240.64 to
76.24 microseconds.

So the narrowing is worth 164.39 microseconds a layer and 12.82 milliseconds a
round, not 223 and 17.41. It remains the largest single lever and the only
measured one in the table.

## Width 32 is now null

The old table's last step bought 6.872 accepted tokens for 13 microseconds a
layer and carried the path from 3.09 to 2.60 ms. But more rows touch more
experts: at 32 rows a rank sees 10.21 of its sixteen, so the narrowing falls
from 164.39 to 87.12 microseconds a layer. The acceptance gain and the
narrowing loss cancel almost exactly -- 4.858 ms at width 32 against 4.838 at
width 12. The step is not a regression; it is simply not a step.

This also removes what the old table called its weakest link. The 6.872 was a
single-stream number against a twelve-stream saturation value of 4.289, and
that doubt no longer has to be resolved.

## Where the path ends

| step | saving us a layer | TPOT ms | basis |
| --- | ---: | ---: | --- |
| the bank narrowed to what a round touches | 164.40 | 6.967 | measured |
| the routed metadata as one kernel | 25.00 | 6.608 | estimate |
| the expert combine in the matmul epilogue | 20.00 | 6.322 | estimate |
| the latent and rope made contiguous | 23.00 | 5.992 | estimate |
| the four projections fused | 35.00 | 5.490 | estimate |
| one reduction a layer, not two | 34.00 | 5.003 | measured |
| both reductions inside the launch | 11.50 | 4.838 | measured |

The last row is new: folding both tensor-parallel reductions into the layer
launch saves 23.05 microseconds a two-layer pair, so 11.5 a layer. Its fixture
states that the figure mixes launch fusion with the SHMEM-versus-HCCL choice
and does not isolate either.

At 4.838 ms a round is 26.32 milliseconds, or 337.5 microseconds a layer,
against a narrowed traffic floor near 47. The remaining factor of seven is not
itemized, and 2.x ms needs a mechanism that is not on this table. The round's
unattributed 5.86 milliseconds is 75 microseconds of that 337.5, so closing the
account is now worth more than any remaining estimated step.

[`recompute.py`](recompute.py) derives the table and [`road.json`](road.json)
records it at both widths.
