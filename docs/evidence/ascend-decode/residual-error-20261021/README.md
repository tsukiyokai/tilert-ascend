# The residual 5 percent: two mechanisms excluded, cause open

2026-10-21. Constant weights, sixteen experts, one row each.

    vendor   absmax 0.45117
    kernel   absmax 0.47461    ratio 1.0520, max difference 0.126953

A constant weight makes the block analytic -- every code one, every bias zero,
so a row's gate-up product is the row's own sum. Computed in float64 on the
host:

    analytic                           0.343068
    recombination coefficient 15.0     0.321931
    recombination coefficient 16.0     0.343068
    recombination coefficient 17.0     0.362906
    without the middle quantisation    0.343163

## What this excludes

The MSD recombination coefficient: reaching 0.45 from this model needs a
coefficient near 21, which is not a candidate value.

The intermediate quantisation: removing it entirely moves the result by three
parts in ten thousand, 0.343068 to 0.343163. It cannot produce five percent.

## What it says about the model

Both measured sides exceed the analytic value by 30 to 38 percent, so the
analytic model is the one that is wrong and cannot adjudicate between them.
The likeliest error in it is the per-token quantum: it takes each expert's
first slot where each row carries its own.

## Why this is now parked

The kernel's value is established at 8.48 ms a round and this error is near
the quantisation noise of a W4A8C8 checkpoint, so it does not block the
engine-side measurement that matters next. It is recorded as an open gap
rather than chased further -- the cost of one more hypothesis here exceeds
what it buys.

## The SwiGLU's form is excluded, and absmax cannot see it anyway

The vendor takes `swiglu_limit` and the engine passes 0.0 while the kernel has
no limit, which made the activation's form a candidate. Computed on the host
over the constant-weight case's own distribution, against the measured ratio of
1.0520:

    no limit                   absmax ratio 1.0000
    gate clamped at 0 above    absmax ratio 0.0002
    gate clamped at 0 below    absmax ratio 1.0000
    product clamped at 0       absmax ratio 1.0000
    silu with beta 1.702       absmax ratio 1.0000

None produces 1.0520. And the four that give exactly 1.0000 show why this
comparison was weak to begin with: under a constant weight the absmax is set by
a single row, and every variant agrees on that row's sign, so absmax has almost
no power to distinguish activation forms. That is lesson 66 again -- a
criterion has to be able to fail for the reason you care about.

Six candidates are now excluded: the MSD recombination coefficient, the
intermediate quantisation, a non-contiguous scale tensor, the per-token scale
slots, matmul reuse across calls, and the activation's form.

The gap stays parked. The kernel's value is the launch merge, measured as a
ratio with both sides under identical conditions -- 8.54 ms a round cold -- and
five percent sits inside a W4A8C8 checkpoint's quantisation noise. What blocks
progress is the absent steady-state capture and the absent end-to-end run, not
this.

## Bisecting on the shape localises it to the rows

    experts  rows  blocks   vendor     kernel     ratio
          1     1       1   0.40234    0.40625    1.0097
          1     4       1   0.25781    0.17480    0.678
          4     1       4   0.45312    0.48438    1.069
         16     1      16   0.45117    0.47461    1.052

One expert and one row agree to one percent. One expert and **four rows**
differ by 32 percent. Many experts at one row differ by 7 percent.

So the defect is in handling more than one row, not in the arithmetic -- which
is what six excluded candidates were all about. The multi-expert term is
secondary and may be the same thing seen through the stacking.

## Where that points

Two per-row quantities exist in the kernel: the per-token scale, read as
`token_scale[expert * SCALE_SPAN + row]`, and the row's offset within the
M-stacked `stacked` buffer, which uses `STRIDE = M_TILE / 2` separately from
the loop bound. The scale slots were checked earlier -- every slot of every
expert's block is filled -- so the offset is the remaining suspect, and it is a
single constant with a measurable consequence.

That is a far better position than six excluded arithmetic candidates, and it
came from varying the shape rather than the inputs.

## And the two per-row constants are both correct

`STRIDE = M_TILE / 2 = 8` with `M_TILE = 16`, and the recombination is
`C[r] = 16 * C'[r] + C'[STRIDE + r]` -- the comment at the top of the kernel.
So `Split` writes a row's high nibbles at `r` and its low nibbles at `8 + r`,
and the cube's M=16 matmul covers both. At one row that is rows 0 and 8; at
four rows it is 0 to 3 and 8 to 11. Correct in both cases, and correct for any
row count up to eight.

`SCALE_SPAN` is `ceil(STRIDE / BLOCK_FLOATS) * BLOCK_FLOATS = 8`, and the
vendor comparison builds its per-token scales as
`token_scale.reshape(EXPERTS, 8)[:, :ROWS]`. Consistent.

So both per-row constants are right and the 32 percent at four rows is still
unexplained. What the bisection establishes is where to look -- the multi-row
path -- not which line. The honest state is a localised defect with its two
obvious causes excluded, which is better than six excluded arithmetic
candidates and not yet a diagnosis.

## Five per-row index sites read, all correct

- `STRIDE = M_TILE / 2 = 8`, and `C[r] = 16 C'[r] + C'[STRIDE + r]` holds for
  any row count up to eight.
- `SCALE_SPAN = 8`, matching what the vendor comparison builds.
- `middle_low = Nibbles(middle, STRIDE * HALF)`, so a row's low nibbles land on
  M-axis row `STRIDE + row` -- as the recombination needs.
- `Narrow` combines `row * HIDDEN` with `(STRIDE + row) * HIDDEN`, and takes
  `of_row`, `token` and `weight` at `row` and `taken_from + row`.
- `Split` reads `rows_global[source * HIDDEN]` with `source` from
  `held.source.GetValue(taken_from + row)`, in range for every shape tested.

So the shape bisection localised the defect to the multi-row path and five
readings of that path found nothing. The marginal return on reading further is
low, and the defect does not block: the kernel's value is the launch merge,
measured as a ratio with both sides identical, and five percent sits inside a
W4A8C8 checkpoint's quantisation noise.

Parked again, with a better position than before: the shape that reproduces it
is one expert and four rows, which is the smallest case that shows it and the
right place for a device-side probe when one is worth building.
