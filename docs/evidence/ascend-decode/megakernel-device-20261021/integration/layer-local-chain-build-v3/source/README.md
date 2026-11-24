# One-launch local layer chain

This entry composes input AddRMSNorm and dynamic A8 quantization, indexed or shared-indexer attention, and the complete NZ FFN in one mixed AIC/AIV launch. Attention source is frozen from `8deff750`; FFN source is frozen from `381febf9`. The input requires a separate BF16 residual and M1–32 with 16 or 24 core groups.

Both TP reductions are identity cuts. The actual rank-local attention partial feeds the post-attention norm and FFN; no external post-AR1 value is substituted. Output owners are FFN `bridge[10]` and retained residual `bridge[1]`. This local dataflow is a composition test, not the TP16 decoder's mathematical result. The entry has not passed a device comparison.

The caller owns validated ND activations, metadata and caches, original NZ expert weights, prepared tilings, all scratch tensors and the current submitted stream. Input norm outputs have separate storage; normalized BF16 feeds the indexer while its A8 codes/scales feed Prolog. The attention callable's final publication barrier precedes FFN reads. FFN retains its original global-task terminal completion. A consumer inside the same kernel would need a publication boundary before reusing its output or UB.

`prepare.py ATTENTION_BUILD FFN_BUILD NEW_OUTPUT` verifies both libraries and every frozen source before copying them into an isolated tree. The generated source includes the attention MLA-tail definition before the FFN copy: their only difference is the standalone tail's combined-scale instantiation; the shared FFN explicitly requests the same combined-scale helper. Parent arithmetic, tilings and internal joins remain intact. Kernel compilation, original-chain numerical comparison, changing-input graph replay and matched timing are separate gates.
