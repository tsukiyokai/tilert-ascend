# The front stage built in Ascend C: 33.88 microseconds a layer

The layer's first dependency level now runs as one matmul invocation instead of
four. The kernel is [`tests/glm52/megakernel`](../../../../tests/glm52/megakernel/README.md);
`MEGAKERNEL_FRONT_SPLIT=1` builds the control from the same source, the same
tiling and the same concatenated output, differing only in how many invocations
each core issues.

## The comparison

One idle die of a3-194, CANN 9.2, four widths, two alternating passes a width,
eight changing inputs and five timing repeats a pass.

| rows | fused us | split us | saving us | digests equal |
| ---: | --- | --- | ---: | --- |
| 1 | 48.00, 47.97 | 81.30, 81.21 | 33.30 | 8 of 8 |
| 2 | 48.39, 48.07 | 81.14, 81.26 | 32.87 | 8 of 8 |
| 12 | 48.37, 48.52 | 82.40, 82.34 | 33.88 | 8 of 8 |
| 32 | 49.24, 49.59 | 85.58, 85.03 | 35.99 | 8 of 8 |

The two passes of a width agree to within 0.4 microseconds and the arms' sample
ranges do not overlap. At the engine's twelve rows the saving is 33.88
microseconds a layer, or 2.64 milliseconds a round.

## It computes the same thing

Every width runs eight changing inputs through both arms, and the product's
digest is identical between them at every input. That is the check that matters:
concatenating the output axis changes which invocation computes a column, not
the reduction that produces it, so anything other than equality would mean the
slicing is wrong. Both arms also match a float reference to 5.5e-4 against
magnitudes near 320, which is bf16 accumulation.

## It is the reduction depth, not the rows

The fused arm costs 48.00 microseconds at one row and 49.59 at thirty-two, a
rise of 1.6 over a 32-fold change in rows. The split arm rises 4.3 over the same
range, because it pays the depth four times. This is the same shape the
shape probes found and the reason the concatenation works.

## Scope

bf16 operands and a float product, which isolates the invocation count from the
quantization contract. The deployed projections are W4A8 with per-channel
scales, and the concatenated weight layout is a load-time step; neither is built
here. One die, no tensor-parallel reduction, no model.

[`run.sh`](run.sh) is the alternating runner and the JSON files are its output,
one a width a pass an arm.

## The core count, and where the kernel still loses

Sweeping the cores at twelve rows, both arms numerically correct to 6.4e-4:

| blocks | fused us | split us | saving us |
| ---: | ---: | ---: | ---: |
| 4 | 152.69 | 214.65 | 61.96 |
| 8 | 97.07 | 134.23 | 37.16 |
| 12 | 63.75 | 132.69 | 68.94 |
| 16 | 65.68 | 118.27 | 52.59 |
| 20 | 59.36 | 103.83 | 44.47 |
| 24 | 48.64 | 82.50 | 33.86 |

Twenty-four cores is best for both, so the fused arm's 48 microseconds is not a
core-count problem. It is a tiling one: `torch.matmul` computes the same
`[12, 6144] x [6144, 3040]` in 39.12 microseconds, and 37.36 MB at the die's
1600 GB/s would be 23. So the stage as built sits at 768 GB/s against the
vendor's 955 and a bandwidth ceiling of 1600.

## The row stride has to be 512-byte aligned

Widening the weight's row stride while computing the same 3040 columns is
faster, not slower:

| stride | bytes a row | us |
| ---: | ---: | ---: |
| 3040 | 6080 | 48.12 |
| 3072 | 6144 | 38.43 |
| 4096 | 8192 | 37.79 |
| 6080 | 12160 | 38.61 |
| 8192 | 16384 | 38.71 |

Every stride from 3072 up costs the same, so it is the alignment: 6080 bytes is
not a multiple of 512 and 6144 is, so each core's column window straddles a
boundary on every row. Padding the concatenation to 3072 columns -- 32 columns
of zeros -- takes a fifth off, and 38.43 matches `torch.matmul`'s 39.12 on the
same shape, so the stage reaches vendor parity once aligned. The alignment is
now part of the contract rather than a caller's choice.

With it, both arms at the aligned stride:

| rows | fused us | split us | saving us |
| ---: | --- | --- | ---: |
| 1 | 38.18, 38.13 | 80.84, 80.94 | 42.76 |
| 12 | 38.43, 38.49 | 82.62, 82.85 | 44.36 |
| 32 | 38.27, 38.63 | 85.31, 84.80 | 46.68 |

The alignment helps only the fused arm; the split arm's windows are 96, 32, 32
and 32 columns and move by less than 0.5 microseconds. The fused arm is now flat
in rows -- 38.18, 38.43, 38.27 -- which is the reduction-depth model stated
plainly. Digests remain identical between the arms in every case.

The cube tile is not a further lever: at twelve rows the tiling API's own choice
costs 47.99 microseconds against 48.10 for an explicit 16/128/128 and 50.9 to
51.4 for any 64-wide output tile, with 512-deep reduction tiles rejected
outright.

## An open hypothesis, and why it is not recorded as a result

Each core reads 128 columns out of a 3040-column row, so 256 contiguous bytes
every 6080. A weight arranged as one contiguous matrix a core would remove that
stride, and three attempts to build it -- correcting `SetOrgShape`'s weight
stride, giving the product its own per-core block, and matching the host tiling's
org width to the kernel's -- all produced errors near 500 against magnitudes
near 330. The variant is removed rather than kept: its timings were within noise
of the correct arm at every width while it returned NaN, so they establish
nothing. The hypothesis stands, and the gap it would address is the 48 against
23 microseconds above.
