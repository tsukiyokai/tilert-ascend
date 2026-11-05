# Shared-attention terminal completion

This variant retains the default callable `FullForward` completion contract.
Only its standalone global entry selects `FullForward<true>`, which replaces
the last mixed-core join and reset with local `PIPE_ALL`, DDR completion and
ordinary `TPipe` destruction. All earlier joins, arithmetic, buffers, tiling
and host arguments remain unchanged. It adds no GM or UB allocation.

A later kernel on the same stream observes completion of the entire mixed
task. A device callable returning on one core provides no such edge; using
terminal completion before an in-kernel consumer or resource reuse is invalid.
The [source audit](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal-build-v1/source-audit/README.md)
records the installed CANN 9.2 destructor and synchronization paths and the
frozen parent sources.

The candidate library SHA256 is
`2a2deacf850ce108825002559a1a44e08d6176fdc1c9a092955acc6840eb1ea4`;
its parent is `64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c`.
The [single-die A3 gate](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal212-wide-v1-a3-194/summary.json)
passes 384 raw-byte comparisons for M=1/2/12/32 and 16/24 core groups, with eight
changing inputs per geometry. This includes 128 immediate output-consumer
comparisons inside the captured graph. Outputs and consumer destinations are
poisoned before replay; the latent, KV and RoPE cache boundaries are also checked.

At M12/24, sequence length 8192 and 16 weight banks, the
[same-address paired comparison](../../../../docs/evidence/ascend-decode/megakernel-device-20261021/integration/shared-attention-terminal212-timing-v1-a3-194/summary.json)
saves median 1.233/0.733 microseconds in the two owner arenas, with 26 of 30
pairs faster in each. Candidate/parent medians are 148.290/149.564 and
148.657/149.402 microseconds. All 1,152 numerical and 480 event-tail comparisons
pass. This is a small local shared-attention gain; actual engine integration
and model TPOT are unmeasured.
