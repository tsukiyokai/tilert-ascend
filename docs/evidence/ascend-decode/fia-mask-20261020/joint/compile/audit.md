# Compile boundary

The server CPU contract built one full Dynamo graph containing 78 opaque `vllm.mla_forward` calls. The actual installed outer forward, custom-op kernel body and frozen SharedKVAttention binder were used; the FIA primitive only observed masks on CPU. A/B/reset matched eager mask observations, preserved layer order and reused one compiled graph. B changed output and reset restored A.

Installed `vllm_ascend/ops/mla.py:269–280` calls the opaque operation; `:285–298` executes the implementation and binder inside it; `:310–315` registers the output mutation and fake implementation. The new Python mask assignment is therefore outside the ordinary backbone Dynamo trace. `vllm/compilation/breakable_cudagraph.py:108–111` bypasses segmentation for FULL runtime mode, which CannGraph uses.

`cann_model.py:248,368–385` binds the helper before the first forward. `:451–508` refreshes persistent slot metadata. `bench_cann_mtp.py:400–470` first warms the compiled width and then enables graph capture. `run_cann_mtp_rounds.py:393–400` omits serial M1 in compile mode to avoid locking the first trace to M1. This path cannot establish serial equivalence on its own; the report must retain null for that claim.

The graph tail-poison conflict was real: a full-capacity dense PV cannot consume NaN masked rows. The narrow fix is finite poison only for shared_fia main-model `latent` and `rope`; main indexer fields and all draft fields keep their existing NaN/integer poison. Parent reports this fix in d49ed4f; the live source at `run_cann_mtp_rounds.py:406–419` now has that boundary.

No further CPU structural blocker was found. This does not validate NPU backend compilation, static/SuperKernel lowering, real 78-layer graph resources or model numerical equivalence. The next model operation is a width8 compile/graph/reset pilot using the existing entry below, followed by the registered serial/teacher and full request gates. No NPU was used by this audit.

```bash
torchrun --nproc-per-node 16 --master-port "$MASTER_PORT" \
  "$TREE/tests/glm52/run_cann_mtp_rounds.py" \
  --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
  --capture-case /workspace/offline/prefix8k-allocation-20261017/capture/case-00 \
  --output "$OUT" --source-commit "$SOURCE_COMMIT" \
  --width 8 --max-tokens 32 --capacity 9344 --ignore-eos \
  --target-graph --compile-target --super-kernel \
  --attention-kernel shared_fia --fp32-head \
  --gpu-memory-utilization 0.78
```

The pilot requires an otherwise approved whole-machine allocation, activated production environment, TREE on PYTHONPATH, isolated VLLM_CACHE_ROOT and HCCL port range, and TORCHINDUCTOR_AUTOGRAD_CACHE=0. The command is recorded only; it was not executed.
