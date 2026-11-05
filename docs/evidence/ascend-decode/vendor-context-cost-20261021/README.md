# The vendor operator costs 2.5x more inside a mixed process

2026-10-21, a3-84, one die. The same operator, the same shape -- 96 rows,
sixteen experts -- in two process states.

    clean process                        61.617 us   (event clock 61.036)
    after a domain build and a custom
    kernel has run                      153.325 us

Both clocks agree within one percent, so this is not a measurement artefact.

## Why it matters more than the number

The engine is a mixed process: it builds a communication domain before it
decodes. So the figure that belongs in the round's accounting is the high one,
and the recorded 157.2 us a layer stands -- it was measured in the right
state by accident.

But every vendor baseline taken in a clean probe understates what the engine
pays, and every one taken in a mixed probe overstates what a clean kernel
would. Two of today's comparisons sit on opposite sides of that: the
hand-written kernel's 2.33x disadvantage was measured against a vendor in the
same mixed state, so it is the honest comparison; the row sweep that showed
the operator is fixed cost was also mixed, so its conclusion holds.

## The rule

A baseline is a property of the process, not of the operator. Record which
state it was taken in, and compare only within one state.

## Neither candidate explains it

2026-10-21, each turned on separately against the same measurement:

    baseline                           62.287 us
    after loading the kernel library   60.309 us
    after running the kernel once      61.161 us
    baseline                           68.891 us
    after building a domain            66.478 us

So loading the library, running the kernel and building a communication domain
are all innocent. The 153.325 us state remains unexplained, and the remaining
difference between the two probes is that the slow one had already built a
second fractal-NZ weight bank and run the full two-stage vendor chain.

## The opening this leaves

The engine records 157.2 us a layer, which is the slow state. A clean process
runs the same operator at 61.6. If the condition that puts the engine in the
slow state can be found and avoided, that is about 90 us a layer -- near 7 ms
a round -- without writing a kernel. Larger than any lever measured today and
reachable from a single die.
