# The routed kernel cannot walk more than five experts a block

2026-10-20, a3-84, one die, with another tenant on the machine.

Sixteen experts, varying only how many a block walks:

    blocks 16 (rounds 1): wall  47.783 us
    blocks  8 (rounds 2): wall  66.070 us
    blocks  4 (rounds 4): wall 114.925 us
    blocks  3 (rounds 6): aicore exception
    blocks  2 (rounds 8): aicore exception

And eighty experts, which isolates the rounds from the count:

    experts 80 blocks 16 (rounds 5): wall 147.514 us
    experts 80 blocks 14 (rounds 6): aicore exception
    experts 80 blocks 13 (rounds 7): aicore exception
    experts 81 blocks 17 (rounds 5): wall 149.426 us

The limit is five rounds a block. It does not depend on the expert count, the
block count by itself, the indexing, the scratch size, or the machine.

## Not introduced by the absorption

Every earlier measurement of this kernel -- the numerics against the vendor,
the real checkpoint weights, the variable group list, the absorbed gather and
scatter -- ran at one expert a block. The limit was always there and nothing
had exercised it. The real shape is what exposed it: a rank holds 256 experts,
and sixteen blocks walking them is sixteen rounds.

## Why raising the block count does not rescue it

Five rounds over 256 experts needs 52 blocks. A3 has 24 AI cores a die, and
`numBlocks` above the physical core count makes the framework insert the wrong
synchronisation -- a trap recorded earlier in this project. So the reachable
ceiling is 24 blocks, eleven rounds, still past the limit.

## What the mechanism is not

The device reports "the GM address accessed by scalar exceeds 48 bits". Ten
explanations were tried and each was disproven by experiment: a slot index
past the valid assignments, an over-read of the group list, an abnormal
scratch address, a machine difference, a self-inconsistent probe table, the
indexing path, the witness buffer's flush allocation, the scratch size, the
block count, and the manually allocatable event ids -- the flag constants are
fixed at 4, 5, 6 and 7 and do not grow with rounds.

What survives is the shape of the fact: the cross-core flags are counted, each
round sets two on each half, and nothing drains them between rounds. A counter
that saturates after five rounds fits every observation, and nothing else
tried does.

## Settled: the TPipe, not the flags

Replacing the paired flags with `SyncAll<false>()` grid barriers did not move
the limit -- six rounds still faulted -- so the counted flags were not it
either. What lifts it is constructing the two `Matmul` objects once a launch
instead of once an expert:

    experts  16 blocks 16 (rounds  1):  47.584 us
    experts  16 blocks  4 (rounds  4): 113.460 us
    experts  16 blocks  3 (rounds  6): 163.381 us
    experts  16 blocks  2 (rounds  8): 212.101 us
    experts  16 blocks  1 (rounds 16): 408.542 us
    experts 256 blocks 16 (rounds 16): 583.396 us

Every count that faulted now passes, including the engine's own shape: 256
experts over sixteen blocks, output fully populated.

`Project` constructed a `Matmul` and called `Init(&cube, &pipe)` on every
call. `TPipe` hands out buffers cumulatively -- this file already says so
about the vector workspace, in a comment written after that exact mistake --
and the cube's `Init` takes its scratch from the same pipe. Two a round, so
the sixth round is the eleventh and twelfth instantiation, and it runs out.

The lesson that was already in the file applied to the half I had not thought
to apply it to.

## What it settles

The kernel now serves 256 experts. The grid barriers went in anyway and stay:
they put the synchronisation outside both halves' branches, which makes the
empty-expert case structural rather than a convention to remember, and the
shared-expert fold needs them for its N-split. But they were not what the
round limit was about, and saying so is the point -- the barrier rewrite was
the right change for the wrong reason.
