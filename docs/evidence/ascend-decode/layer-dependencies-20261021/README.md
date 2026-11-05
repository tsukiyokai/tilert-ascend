# What a layer actually permits to overlap

Read from the reference implementation rather than inferred:
`/tmp/glm_dsa_fetch/tf_modeling_glm_moe_dsa.py`, `GlmMoeDsaDecoderLayer.forward`
and `GlmMoeDsaMoE.forward`.

    residual = hidden
    hidden = input_layernorm(hidden)
    hidden, _, topk = self_attn(hidden, ...)
    hidden = residual + hidden
    residual = hidden
    hidden = post_attention_layernorm(hidden)
    hidden = mlp(hidden)
    hidden = residual + hidden

and inside the MoE:

    _, topk_weights, topk_indices = self.gate(hidden)
    hidden = self.experts(hidden, topk_indices, topk_weights)
    hidden = hidden + self.shared_experts(residuals)

## One real fork, and it is the MoE's

`shared_experts(residuals)` and `experts(hidden, topk, weights)` read the same
input and meet only at the add. So the shared expert's whole path -- its gate,
its up and its down projection -- is independent of the routed experts' grouped
matmul and can be issued concurrently with it.

`gate` must precede `experts` because it produces the indices, so the gating
cannot overlap the routed experts. It can overlap the shared expert.

## And the attention side permits nothing

    norm -> q_a/kv_a projection -> q_b projection -> attention -> o projection

Each step consumes the previous one's output. Four of the layer's six small
operators sit on this chain, so none of them can overlap another. The 43.75 us
measured with independent vector work is not available here.

## What that leaves, quantified

Overlappable: the shared down projection at 21.244 us, plus whatever part of
the 44.783 us fused input projection belongs to the shared expert's gate and
up. Call it 21 to 40 us a layer against the 54.5 the target needs.

So the overlap lever is real, bounded by the dependency structure, and smaller
than the gap. It is worth 1.6 to 3.1 ms a round if the concurrency measured on
independent work carries over to these two branches -- which is the next
measurement, with the shared expert and the grouped matmul specifically.

## The structural consequence worth stating

A batch-one decode layer is a chain by construction, and the only parallelism
in it is the MoE's two branches. Any further overlap has to come from somewhere
other than within a layer -- across layers, which cross-layer fusion already
closed at 6.15x, or across the speculative width, which is the one axis this
project has not examined: the draft rows are independent of each other, and at
width 24 there are twenty-four of them.

## Correction: the MoE fork is cube against cube, and concurrency loses

The 21 to 40 us a layer above priced the fork using a vector-under-cube
measurement. But the shared expert is matmuls, so the fork is cube under cube.
Measured on one die at twelve rows:

    routed side alone        44.35 us
    shared side alone        55.00
    sum                      99.35
    larger of the two        55.00
    serial, one stream       71.00
    concurrent, two streams  89.49
    concurrency gains       -18.49 us   -34% of the shared side

Concurrency is **slower** than serial. Both sides need the cube, so neither
hides inside the other, and the second stream only adds its event and
scheduling cost. Serial on one stream is already 71.00 against a sum of 99.35,
so the pipelining that exists happens without being asked for.

So the 21 to 40 us figure is withdrawn. It applied a vector-under-cube result
to a cube-under-cube fork.

## Which closes within-layer overlap

    attention side   a strict chain, four of the six small operators on it
    MoE fork         cube against cube, concurrency costs 18.49 us

What remains of the mechanism is vector-under-cube specifically, and the only
vector operators in the layer are the norm and the dynamic quant -- each
immediately feeding its consumer. The 43.75 us measured earlier required the
vector work to be independent of the cube work, and in this layer it is not.
