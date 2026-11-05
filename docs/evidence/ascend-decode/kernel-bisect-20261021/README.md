# Bisecting the kernel: two defects, and the fast commit is the empty one

2026-10-21, a3-84, one die. Each commit's kernel built from `git archive` and
judged against the vendor chain at sixteen experts, constant weights.

    a04e6ce  vendor 0.45117  kernel 0.47461  unequal 98304/98304
    ba67cf3  vendor 0.45117  kernel 0.00000  unequal 98304/98304

`ba67cf3` is "barriers once a launch, 214.5 to 79.206 us at 256 experts". The
speedup is real and what it accelerated produces nothing.

## Two defects, not one

They were being chased as one, which is why ten hypotheses in a row failed to
hold: every explanation for the zero had to also explain a magnitude error,
and no single mechanism does.

- `ba67cf3` returns zero. Structural, introduced by the stage-major
  restructuring, reproducible at sixteen experts.
- `a04e6ce` returns the right magnitude and the wrong value -- 0.47461 against
  0.45117, about five percent, with every element unequal. Numerical, and
  already present before the restructuring.

## What this costs

The 79.206 µs at 256 experts, 1.98x the vendor's 157.176, measured a kernel
whose output is absent. The timing stands as a timing. The 5.8 ms a round it
implied does not stand as a saving, because there is no saving in replacing a
correct operator with an empty one.

The constant-weight agreement recorded earlier -- 0.45117 on both sides, one
bfloat16 ULP -- belonged to an earlier binary. The vendor produced 0.45117 in
both runs here, so the vendor did not change; the kernel did.

## Method

Two archived builds and one judge settled in one command what ten hypotheses
could not. Bisection was available from the first measurement and was not used
because each new symptom looked like a reason to reason instead.
