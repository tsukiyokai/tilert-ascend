# Same process, same shape: the hand-written kernel is 8.3 percent behind

2026-10-21, a3-84, one die, one process. Twelve tokens, 96 sorted rows,
sixteen experts -- the shape a width-twelve round presents.

    vendor grouped matmul    67.336 us   5.25 ms a round
    hand-written kernel      72.933 us   5.69 ms a round

## Three wrong comparisons this replaces

- "1.98x faster": the cube half skipped every expert, so nothing ran.
- "2.33x slower": true, but at 256 declared experts, not the decode shape.
- "25.7 percent faster": 49.004 us in a clean process against 65.950 in a
  mixed one. Cross-process, which the 2.5x state difference makes meaningless.
- "16.6x slower": 1133.927 us, with seven tensors allocated inside the timed
  call. A probe error of the same family as the rest of today's.

Only the figure measured in one process with arguments built once survives.

## Why 8.3 percent behind is the useful number

The kernel still carries a five percent numerical error, reads ND rather than
fractal-NZ, and dequantises on the vector side instead of through the fixpipe.
It is 8.3 percent behind the vendor while giving all three away.

And the operator's own cost is a per-execution constant: 2.8 percent over a
sixteenfold row range, flat across a 4x weight bank range, linear in call
count, unchanged by graph capture. About 60 us that the data does not move.
A fused vendor operator pays it once per layer by construction; a kernel that
covers several layers in one execution pays it once for all of them. That is
the only mechanism measured today that can reach the round's fixed term.

## What is left to check

The engine's own state. Every figure here is from a clean process, and the
engine's recorded 157.2 us a layer says it runs 2.5x slower than this. The
comparison that decides the design is the same pair measured inside the
engine.
