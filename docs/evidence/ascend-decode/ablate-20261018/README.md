# Which flag earns what, and by which mechanism

Leave-one-out from the composed set on `a3-84`, machine idle, source
`04a6628`, same frozen 8K prefix and checkpoint, `--max-tokens 1024
--capacity 9344 --warmup 3 --requests 3`, profiler off. Each case drops one
flag and keeps `--target-graph` and the rest. Baseline is
[composed](../compose-20261018/README.md) at 16.640 ms, A = 3.433,
298 rounds.

| dropped | TPOT (ms) | Δ TPOT | A | Δ A | round (ms) | Δ round |
| --- | --- | --- | --- | --- | --- | --- |
| `--super-kernel` | 18.447 | +1.807 | 3.433 | 0 | 63.33 | +6.21 |
| `--draft-graph` | 18.321 | +1.681 | 3.433 | 0 | 62.90 | +5.78 |
| `--fp32-head` | 17.957 | +1.317 | 3.148 | −0.285 | 56.52 | −0.60 |
| `--attention-kernel gather_fia` | 16.856 | +0.216 | 3.332 | −0.101 | 56.17 | −0.95 |
| `--draft-projection-tp` | 16.740 | +0.100 | 3.399 | −0.034 | 56.89 | −0.23 |

The round column is derived as TPOT × A, not independently measured.

The five flags split cleanly into two kinds, and the sign of Δ round is what
separates them.

Two are time optimisations. `--super-kernel` and `--draft-graph` leave
acceptance and round count untouched — both cases run 298 rounds at
A = 3.433, exactly the baseline — and lengthen the round when removed. They
are worth 1.807 and 1.681 ms, together 3.488 ms.

Three are numerical changes whose entire return is acceptance. Dropping
`--fp32-head`, `gather_fia` or `--draft-projection-tp` makes the round
shorter, by 0.60, 0.95 and 0.23 ms, and yet TPOT worse, because acceptance
falls by 0.285, 0.101 and 0.034. They cost time and buy guesses. Together
they are worth 1.633 ms, of which `--fp32-head` alone is 1.317 ms.

So about a third of the 4.570 ms between the default and the composed set
comes from guessing better rather than computing faster. That matters for
what remains: acceptance at M = 4 cannot exceed 4, and 3.433 is already 86%
of it, so this route has at most 16% left. It is also exactly the route
whose three flags owe the frozen-teacher and eight-input acceptance of the
plan's section 4.

`gather_fia` additionally contradicts the screen that selected it. The
[reuse screen](../sfa-reuse-20261018/README.md) measured SFA 89.05 µs
against direct gathers plus FIA 69.15 µs on the real layer-0 M4 map, which
extrapolates to 1.56 ms saved over 78 layers; in the full model it makes the
round 0.95 ms longer. An independent synthetic measurement on a base-format
cache also found it slower, 137 µs against 88 µs
([shape scaling](../sfa-shape-20261018/README.md)). Two measurements now
agree against the screen.

Three of these cases failed before this run with
`TypeError: 'NoneType' object is not callable` at
`vllm/compilation/caching.py:224`, and exactly the three that alter the
compiled graph. Adding `VLLM_DISABLE_COMPILE_CACHE=1` alongside
`TORCHINDUCTOR_AUTOGRAD_CACHE=0` fixed all three, which supports the
candidate mechanism: disabling the AOTAutograd cache stops the artifact
being written while vLLM's own cache still records a hit and returns
nothing. Both variables are required for any compiled measurement.
