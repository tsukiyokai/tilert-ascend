# Folding the shared expert the cheap way, and why it cost double

2026-10-20, 187, one die, with another tenant on dies 0 and 4.

    routed alone              37.193 us
    routed + shared folded    71.625 us
    the two kernels apart     37.193 + 13.863 = 51.056 us

Both outputs correct in shape and finite, every value nonzero, one launch
instead of two. And slower than two launches, so the fold is a regression as
written and is reverted.

## The shape, not the structure

The fold ran the shared expert as sixteen experts of one row sharing one
weight bank, because that needed no change to the loop: the weight offset
simply does not advance. Sixteen matmuls of M=1 read the same weight matrix
sixteen times. I had argued this was affordable from `aic_mac_ratio` 0.109 --
the cube is nearly idle -- which is the wrong ratio. The binding constraint is
`aic_mte2_ratio` 0.733: the kernel is moving weights, and sixteen reads of one
bank is sixteen times the movement. My own profile had that number in it.

## What doing it properly requires, and what that settles

One matmul of M=16 reads the bank once, and needs the sixteen rows contiguous.
They are not: `stacked` lays each expert's slice out on an `M_TILE` stride of
sixteen, because `Matmul` reads and writes a whole sixteen-row tile whatever M
says, so the real rows sit sixteen apart.

Worse, one matmul is one piece of work for sixteen blocks. Spreading it means
splitting the shared expert along N -- each block taking its slice of the 256
gate-up columns and the 6144 down columns -- and then the row maximum spans
every slice and the down projection reads every slice. Which needs grid-wide
barriers, not the per-pair `CrossCoreSetFlag` this kernel uses.

That is exactly mixbrick's design, and exactly why its commit message says
"two grid barriers, because the down projection reads every slice and the row
maximum spans every slice". So the conclusion is not about the shared expert:

**the megakernel's foundation has to be mixbrick's grid-barrier structure
carrying routed_int4's MSD weight path, and the N-split is what makes the
grid barrier necessary rather than stylistic.**

I had reached for that structure earlier and then talked myself out of it,
thinking the per-pair flags would stretch. They do not stretch across an
N-split.

## A packaging trap, in passing

A schema string that disagrees with the C++ signature throws
`c10::Error: Inferred operator schema for a C++ kernel function doesn't match
the expected function schema` at `dlopen` and dumps core, rather than naming
the argument that differs.
