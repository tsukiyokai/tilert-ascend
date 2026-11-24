# Four projections off the hidden states, as one matmul: 2.46 ms a round

2026-10-21, a3-84, one die, TP16 shard, twelve tokens.

    separate     68.724 us   5.36 ms a round
    one matmul   37.238 us   2.90 ms a round
    saving       31.486 us   2.46 ms a round

Relative difference 4.058e-04 against output magnitudes of 306 to 458, where
bfloat16's ULP is 1 to 2. Reblocking the same arithmetic, not an approximation.

## Why they are worth fusing

At the TP16 shard each projection costs 12 to 19 us whatever its size:

    q_a_proj, replicated       6144 x 2048   24.0 MB   16.915 us   1386 GB/s
    q_b_proj, column split     2048 x 1024    4.0 MB   13.027 us    300 GB/s
    kv_a_proj, replicated      6144 x  576    6.8 MB   18.620 us    354 GB/s
    o_proj, row split          1024 x 6144   12.0 MB   13.176 us    889 GB/s
    router, replicated         6144 x  256    3.0 MB   18.510 us    158 GB/s
    shared gate+up, column     6144 x 1536   18.0 MB   17.102 us   1028 GB/s
    shared down, row            768 x 6144    9.0 MB   12.098 us    726 GB/s

The router's 3 MB runs at 158 GB/s where q_a's 24 MB reaches 1386, so the
small ones are paid for by the boundary rather than the bandwidth. Four read
`hidden_states` directly -- q_a, kv_a, the router and the shared gate-up -- so
concatenating their weights along the output axis makes them one matmul with a
sliced output.

The earlier unsharded measurement put these seven at 364 us a layer, three
times too high: o_proj alone measured 178 us on a 200 MB weight the engine
splits sixteen ways. At the real shard they are 109.447 us, 8.54 ms a round.

## The levers now measured

    MoE four-into-one        8.48 ms
    one-sided reduce         6.50
    projections four-into-one 2.46
    operator dispatch         1.40
    total                   18.84 ms   round 55.4 -> 36.6

With the adaptive tree's 9.288 accepted over 13.1 nodes that is a TPOT near
5.1 ms. The remaining factor of two needs another 20 ms off the round or
double the acceptance, and both are now specific rather than open.
