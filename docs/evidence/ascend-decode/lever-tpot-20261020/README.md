# Each lever, measured end to end

One arm, one variable, read the record. Host 7.150.10.194, the same warm
directory for every arm so the 245 static kernels are compiled once, and the
same flags otherwise -- `--target-graph --compile-target --super-kernel
--attention-kernel sfa --fp32-head --lookup`, which is what `launch_arm.sh`
supplies. Six requests an arm; the first is a warm-up and the mean is over the
other five.

| arm | one variable | requests, ms | mean |
| --- | --- | --- | --- |
| tail8 | the drafter's tail at 8, as shipped | 10.9518 10.9262 10.9466 10.9217 10.9224 | **10.9337** |
| tail32 | the drafter's tail at 32 | 11.1580 11.1157 11.0746 11.2049 11.1642 | 11.1435 |

## The drafter's tail is worth −1.9%, not nothing

Lengthening it costs 0.21 ms, and the CPU harness could not have seen why.
That harness counts accepted tokens; it does not count what the drafter spends
to propose them. `LookupDrafter.commit` builds one tuple key a suffix length
for every committed token, so a tail of 32 does twenty-four hashes of up to
thirty-two elements where a tail of 8 does six of up to eight -- four times
the host work, per token, against a round that commits about seven. The
acceptance the longer tail buys at this workload's copy rate is small, by that
same harness's own curve, and the host cost is not.

So the first of the four levers to get an end-to-end number gets a negative
one, and the useful part is the flaw it exposes: an acceptance harness that
ignores the proposer's cost can only ever report an upper bound.

## And the tiling lever does not cross the operand width

`SetFixSplit(16, 128, 512)` is worth 16% on the INT8 routed kernel and costs
the INT4 one a factor of two. Swept at sixteen experts on sixteen blocks:

| baseN | baseK | INT4 wall, µs |
| --- | --- | --- |
| the library's own choice | | 39.480 |
| 128 | 512 | 80.243 |
| 128 | 1024 | 79.756 |
| 256 | 512 | 39.276 |

The reasoning that produced the INT8 corner -- narrow N, deep K, fill L0B,
fewest K iterations -- was derived from a profile of the INT8 kernel and does
not transfer. For INT4 the library already picks near the corner and the fixed
split is removed rather than retuned.


## TPOT alone is unreadable, and one arm proves it

TPOT is round time over tokens accepted a round, so a change can lower it by
shortening the round or by making the draft easier to guess -- and the second
is worthless. Read from the engine's own round records rather than from
delivery gaps:

| arm | rounds | accepted a round | round, ms | TPOT, ms | first tokens |
| --- | --- | --- | --- | --- | --- |
| tail8 | 194 | 5.273 | 56.9 | 10.93 | `6500, 382, 262, 576, 6500, 374, 264, …` |
| tail32 | 195 | 5.246 | 57.0 | 11.14 | as above |
| sbs16 | 40 | **25.575** | **116.0** | **4.59** | `6500, 0, 0, 0, 0, 0, 0, …` |

`sparse_block_size = 16` reads as a 2.38x win on TPOT and is nothing of the
kind. Its round got *twice as slow*, 56.9 to 116.0 ms. What moved TPOT is that
the model collapsed into emitting token zero forever, which a prompt-lookup
drafter predicts perfectly -- 25.6 of 32 drafts accepted. Block-pooled
selection at that granularity does not degrade the attention, it destroys it.
The lever is rejected, and anyone reading TPOT alone would have reported it as
the best result of the day.

So the reporting rule is three columns, not one: round time, accepted a round,
and the first tokens of the output. Any one of them alone can be made to look
good by breaking the other two.

## Two of my own figures were wrong, and the records say so

Acceptance here is **5.273 a round at width 32**, not the 7.1 this project has
been carrying -- which lands almost exactly on the published figure for a
draft model on coding tasks, 5.24. Solving `(1 - a^33)/(1 - a) = 5.273` puts
`a` at about 0.81 rather than 0.86, so the asymptote argument was built on a
number from another configuration and the ceiling it implies is `1/(1-0.81)`,
about 5.3 -- which is where this already sits.

And the round is **56.9 ms**, not the 46 to 53 ms the budget has assumed. Both
factors of the TPOT product were wrong, in the same direction: less headroom
on acceptance than claimed, more round time than claimed.

The correction to the tail32 result is smaller but it was also mine.
Lengthening the tail cost acceptance -- 5.273 to 5.246 -- at an unchanged round
time of 57.0 ms. The host-cost explanation offered above is wrong: the
drafter's extra hashing does not show in the round at all. What the longer tail
did was propose slightly worse continuations.
