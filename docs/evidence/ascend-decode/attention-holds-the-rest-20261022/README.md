# Attention holds the layer's remaining time, and its trace cannot be built here

The FFN side is now bounded: its body spends 136.40 microseconds against an
85.59 cube critical path, 23.40 of vector work and 27.4 of locked imbalance, so
about 23 is available and the ported skew takes 2.94. Attention is four times
larger and almost entirely unbounded.

## The estimate

Attention's five profiled phase windows at twelve rows, with the cube work in
each taken from measurements elsewhere in this project:

| phase | window us | cube us | basis for the cube figure |
| --- | ---: | ---: | --- |
| prolog | 42.38 | — | one fused vendor operator, mixed halves |
| indexer front | 70.37 | ~17 | two walks, K=2048 at N=4096 and K=6144 at N=160 |
| indexer (selector, top-k) | 54.84 | 0 | no matmul in the stage |
| SFA | 76.97 | ~26 | the measured gather of 28.31 MB at 1094 GB/s |
| projection tail | 56.55 | 13.02 | the tail's own per-core trace |

So roughly 98 microseconds of the 301 is cube work and 199 is vector. Perfect
co-scheduling would leave `max(98, 199)` or 199, which is 98 available -- four
times the FFN's 23, and the largest identified item in the layer.

Attention's own reads are small: 11.9 MB of weights and 28.31 of gathered KV is
40.2 MB, or 25 microseconds at the 1607 GB/s measured on this die. So attention
runs at about twelve times its traffic while the FFN runs at 1.9 times its own.
The difference is structural: the FFN's cube work **is** its reads, because the
expert weights are large, while attention's weights are small and its time goes
to the selector, the RoPE, the Hadamard and the norms.

That also sets a limit on overlap there. Vector work can only hide under cube
work, and attention has 98 microseconds of cube against 199 of vector, so even a
perfect schedule leaves 199. Attention's vector work has to become cheaper, not
just better scheduled, and its largest single item is the selector's top-k at
54.84 microseconds over 8192 positions a row.

## Why this stays an estimate

The per-core trace that settled the FFN does not exist for attention. Building
the instrumented library needs the recorded ops-transformer snapshot: its
`optiling::MlaPrologBaseParams` carries `mm1NGroup`, `mm1KGroup`, `mm2NGroup`
and `mm2KGroup`, and `attention_native.h` reads them. CANN 9.2.0's installed
tree does not have those members, and `grep -rl mm1NGroup` finds nothing in
either CANN tree on a3-194, so the snapshot is not on this machine.

Two steps of the build do work and are worth keeping: the vendor root is used
only as `-I`, and the source expects one level above `ascendc` -- it includes
`attention/sparse_flash_attention/...` where CANN installs
`ascendc/sparse_flash_attention/...` -- so a shadow tree of symbolic links with
`attention` pointing at `ascendc` satisfies every include path. The build then
reaches the tiling-structure mismatch, which is the real blocker.

    src=/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc
    mkdir -p <shadow>; ln -sfn $src <shadow>/attention
    for d in $src/*/; do ln -sfn $d <shadow>/$(basename $d); done
    OPS_TRANSFORMER_ROOT=<shadow> ATTENTION_INDEXED=1 ATTENTION_INDEXED_PROFILE=1 \
        bash tests/glm52/whole_layer/attention_build.sh <out>
