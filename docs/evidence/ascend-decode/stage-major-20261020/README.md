# Barriers once a launch, not once an expert

2026-10-20, a3-84, one die, another tenant on the machine.

At the engine's own shape -- 256 experts, twelve tokens, ninety-six
assignments:

    expert-major, four barriers a round   214.5 us
    cube skips empty experts             214.8 us
    stage-major, four barriers a launch    79.206 us

And sixteen experts, where the barrier count barely differs, stays at 41.9 µs.

## What the sparse case was telling me

Before the restructuring, the real sparse routing (about seventy experts
holding rows of 256) was *slower* than every expert holding one row -- 214.8
against 192.3. An expert with no rows skipped its arithmetic and still
arrived at all four barriers, so the barriers, not the matmuls, were the
dominant term. Sixteen rounds times four is sixty-four of them.

All the scratch is sized per expert, so every expert's intermediate can be
live at once: the stages can run expert-major inside and stage-major outside,
which is four barriers a launch whatever the expert count. 2.7x.

## Against the vendor, at the shape that counts

The vendor's two grouped matmuls cost 157.176 µs a layer at this shape. This
kernel now costs 79.206, so 1.98x, and 5.8 ms over seventy-five MoE layers.

That figure has been wrong twice. 9.4 ms came from extrapolating sixteen
experts, which understated the expert count by sixteen times; then "the saving
is negative" came from the 214.5 µs measurement before the barriers were
fixed. 5.8 ms is the first version measured at the engine's expert count with
the barriers where they belong.

## A trap worth recording

The first attempt at this restructuring did not apply: the slice ran from the
expert-major `for` to `PipeBarrier<PIPE_ALL>()`, and `str.index` found that
barrier's *first* occurrence, hundreds of lines earlier in another function.
The slice inverted, the assertion caught it, and the binary that then ran was
the previous one -- measuring 214.452 µs against the previous 214.827, which
reads exactly like "the restructuring did nothing". Confirm the edit is in the
file before measuring; an assertion is a weaker guard than looking.
