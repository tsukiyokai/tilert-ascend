# Full-decoder indexer diagnostic

The frozen wrapper observes layer 6's full indexer inside the original asynchronous DecoderLayer 6/7 comparison. Layer 7 and both original FFNs execute unchanged. The accepted storage alias and native DSOs are retained. Two device clones preserve the Q/K inputs to in-place RoPE; other taps hold existing references. CPU reads occur only after the original comparison ends.

The reproduction gate requires exact hidden/residual inputs, both external post-attention inputs, all metadata and every output from both original arms to match the [saved M12 failure](../decoder-index-m12-audit-v1/audit.json). The diagnostic cannot promote a perturbed execution to an explanation without this gate. It retains WK, layer norm, RoPE, Hadamard, dynamic-quantization codes/scales, query/head weights and native front accumulators. FP64 WK and FP32 RoPE calculations are explicitly auxiliary CPU references.

[Four CPU tests](cpu-tests.log) check transparent shared execution, exact reproduction including signed zero and TP inputs, residual-branch rejection and cache-coordinate mapping. [The manifest](manifest.json) pins the unchanged parent, failure tensor identity and all 787 deployed files. These checks do not establish device numerical correctness.
