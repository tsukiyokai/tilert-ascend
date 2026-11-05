# The megakernel's front stage

A decode matmul's cost follows its reduction length, paid once an invocation,
while its output axis is spread across cores for free. See
[the measurements](../../../docs/evidence/ascend-decode/megakernel-design-20261022/README.md):
at K=16384 and twelve rows, widening the output axis 24-fold costs 1.5 times the
wall clock, and splitting one invocation into four at fixed bytes costs 4.4
times as much.

So the layer's first dependency level belongs in one invocation. Four
projections read the normalized hidden at K=6144 -- the query latent at 2048
columns, the key-value latent with its rope at 576, the indexer's key at 160 and
the router's gate at 256 -- and this entry computes all 3040 columns in a single
mixed AIC/AIV launch, giving each core one contiguous slice aligned to 32.

`MEGAKERNEL_FRONT_SPLIT=1` builds the control arm. It is the same kernel, the
same tiling and the same concatenated output; only the invocation count differs,
because each core issues one call a projection instead of one over its slice.
`orgKc` holds the product's row stride at the full width whatever a core
computes, so both arms write the same layout.

## Measured

Three levels, one die, two alternating passes a row count, eight changing
inputs. At twelve rows the saving from concatenating is 31.06 microseconds for
`prolog`, the only grouping the engine can adopt, and −2.16 for `query_norm`.
Both remove one invocation, so the reduction's depth decides: 6144 pays and 2048
does not. Every arm's product digests match its counterpart's. See
[the evidence](../../../docs/evidence/ascend-decode/megakernel-levels-20261022/README.md).

Its weight row stride must be 512-byte aligned, which the contract enforces: the
four-part `hidden` level costs 48.12 microseconds at 3040 columns and 38.43 at
the padded 3072.

## Scope

bf16 operands and a float product, which isolates the invocation count from the
quantization contract. The deployed projections are W4A8 with per-channel
scales; that and the load-time weight concatenation are separate steps. Rows 1
to 32, blocks up to 24, one die.

## Build and check

    bash tests/glm52/megakernel/build.sh                      # fused
    MEGAKERNEL_FRONT_SPLIT=1 bash tests/glm52/megakernel/build.sh
    python3 -I tests/glm52/megakernel/check.py \
        tests/glm52/megakernel/build-fused/megakernel_front.so --rows 12

`check.py` runs eight changing inputs through the arm, compares each product
against a float reference, and reports a digest of the product so the two arms
can be compared exactly. They must agree digest for digest: the concatenation
changes which invocation computes a column, not the reduction that produces it.
