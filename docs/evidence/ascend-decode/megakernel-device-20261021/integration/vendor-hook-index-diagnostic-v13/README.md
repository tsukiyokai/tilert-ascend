# Actual indexer boundary diagnostic

The frozen harness preserves the original outer forward, adapter and native
libraries. It records WK projection, K normalization, Q projection, both RoPE
paths and the C8 Hadamard/quantization boundaries. Two device copies retain the
Q/K inputs before the official in-place RoPE; the report must reproduce the
original failure's inputs, metadata and both complete result sets byte for byte.

Run from the host task directory after the single-die admission gate:

```bash
bash vendor-hook-runtime/tests/glm52/whole_layer/vendor_hook_run.sh \
  4 vendor-hook-index-diagnostic-bf16-v13 \
  --mode bf16 --forward-widths 32 --forward-replays 6 \
  --index-diagnostic \
  --index-diagnostic-parent /evidence/vendor-hook-forward-o-combined-v11-bf16-wide/outer-forward-bf16-m32-r5.pt \
  --indexed-library /workspace/logs/attention-o-combined-v1/indexed/whole_layer_indexed_attention.so \
  --shared-library /workspace/logs/attention-o-combined-v1/shared/whole_layer_full_attention.so
```

An outer mismatch still fails the process. Read
`outer-index-diagnostic.json` only after checking
`original_failure_reproduction.pass`; the FP64 WK and FP32 RoPE CPU
reconstructions supplement the actual production boundaries. The report makes
no timing claim. Source identities and CPU-only checks are in
[manifest.json](manifest.json).
