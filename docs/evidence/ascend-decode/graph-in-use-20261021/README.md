# The engine does decode inside its graph, so the 2.7x is not a capture failure

The engine fits 588 us a layer where the same operators replay in 214 on one
die. One explanation would have been that capture silently fails and decode
runs eager -- 317 us a layer plus sixteen-rank collectives would land near 588.

The records settle it. Every arm carries, at rank 0:

    target_graph: true
    graph: { width: 12, captures: 1, replays: 1128,
             entries: [ { has_aclgraph: true,
                          descriptor: num_tokens=12, num_reqs=1, uniform=true } ] }

One capture and a thousand-odd replays, with `has_aclgraph` true on the entry
whose descriptor matches the arm's width. At widths 8, 12 and 16 the replay
counts are 1350, 1128 and 1098 -- which tracks rounds inversely with width, as
a wider draft accepting more tokens needs fewer rounds for the same output.

So the graph is in use and 588 us a layer is an **in-graph** cost. The 2.7x
against the synthetic layer is therefore not capture, and the remaining
candidate is what the synthetic layer omits: sixteen ranks.

## A figure in these records that cannot be used

`compiled_graph_warmup.rounds[0].elapsed_ns` reads 84 to 91 **seconds** for one
round. That is the warmup round, carrying compilation and capture, and it is
recorded per arm where a reader looking for a round time would find it first.
Noted here so it is not mistaken for a steady-state figure later.
