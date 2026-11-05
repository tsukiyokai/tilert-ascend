# Attention is flat in the query rows too

Through `cann_attention.shared_kv_fia`, this project's own call, which runs
where six attempts at the engine's sparse operator were rejected. Capacity
fixed at the engine's 9344, four heads a rank.

    width   us a call   marginal us a row   a round of 78
        1     123.229                             9.61 ms
        4     122.226              -0.334         9.53
        8     120.513              -0.388         9.40
       12     125.282              +0.187         9.77
       16     124.853              +0.108         9.74
       24     123.203              -0.001         9.61

Flat across a 24-fold range, with the marginal term between -0.39 and +0.19 us
a row -- about ±0.015 ms a row over 78 layers against the round model's 0.66.

## So the slope's source is not on one die

Everything measurable here is flat in the rows:

    the layer's projections and norms, warm      +0.10 ms a row a round
    the same, cold                               -0.02
    the grouped matmul, 16 to 256 rows           +2.8 percent total
    attention, 1 to 24 query rows                ±0.015
    graph capture width                          +0.05
    the round model's fit                        +0.66

Five sixths of the slope is unattributed and the single-die candidates are
exhausted. What is left is specific to sixteen ranks: the TP reduction at TP16
rather than at the two ranks measured today, or the host's coordination of
sixteen ranks a round.

## And a mechanism found while looking

`shared_kv_fia` passes `atten_mask` of shape `(1, 1, width, capacity)` -- a
per-row mask, already in use and already known to run. Tree verification needs
exactly that, and this project built `tree_mask` and `tree_attention` today to
feed `sparse_indices` instead. The mask path looks shorter -- a boolean of width by capacity, which
`tree_mask.ancestor_bitmap` fills directly, against an index set the sparse
operator would not accept from a probe at all.

**But it is gated at width six.** `install_gather_fia` raises above
`GATHER_FIA_MAX_WIDTH`, with the reason in its own docstring: "widths above six
failed serial-equivalence checks". That is the unisolated failure
`fia-mask-20261020` records, and a tree wants width 16 to 24.

So the shorter path is shorter only if that gate opens, and opening it means
isolating a correctness failure rather than adding a mask. Which also explains
why the engine runs SFA instead: not that this path was missed, but that it is
unvalidated at the widths that matter.

The gate itself is now the item: it bounds the tree, it bounds `gather_fia`'s
own 21.3 us a layer, and it has been recorded as unisolated since 10-20.
