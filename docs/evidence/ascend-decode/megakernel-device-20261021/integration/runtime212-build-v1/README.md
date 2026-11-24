# TorchNPU 2.12 real attention runtime

The three libraries below were built against Python 3.11.12,
Torch 2.12.0+cu130 and TorchNPU 2.12.0. They load together in the actual
a3-194 runtime image without an exposed or initialized NPU. Actual imports
of `DeepseekV2DecoderLayer`, `AscendSFAImpl`, `ACLGraphWrapper` and
`GlmMoeDsaConfig` pass. This is a CPU ABI/import gate; device execution is
reported separately. [CPU report](cpu-v2/report.json)

| Library | SHA-256 |
| --- | --- |
| Indexed attention | `e957650c7def45dff8b918b74c987a8ce44eb0b2cca816e7246cbbff8a52378e` |
| Shared-index attention | `64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c` |
| Official SFA adapter, independent build | `bd8aff02f36dca1948b4eea1aa03805d6977b5ae8bf39573b51d73b68a61be3e` |

The [attention manifest](attention/manifest.json) records the original
CANN 9.2 device and tiling object identities and the newly compiled host
bindings. Indexed attention retains the accepted M32 WK K-order correction
and combined-scale O epilogue. Shared-index attention retains its accepted
combined-scale O epilogue. The [SFA manifest](sfa/manifest.json) records the
unmodified official C++ adapter, extracted registration and every included
header identity. It is not a build of the complete vLLM-Ascend extension.

The CPU build used image
`7448df539bcb40397063c4b7e83a6a1590ba20bcc16639c0aa83f1c60f1a0449`
on a3-78. The CPU runtime gate used image
`1910b1acedb37acc42beda9cb695c2cf84c57c81d8dd8b5191a98a97bc6d5fe6`
on a3-194. Both report the package versions above; they are distinct images.
The first CPU runtime command omitted `TRITON_CACHE_DIR` and failed writing
the read-only `/root/.triton`. The second command uses `/tmp/triton`, as the
device launcher already does. Neither CPU command initialized an NPU.
[First report](cpu-v1/report.json), [second report](cpu-v2/report.json)

## Official source and selected layer

vLLM is `a98247ab4db686ee03c66d5feb3c761e52a2f8ab`; vLLM-Ascend is
`8237ee6d690d1399bc6eb48ef5977ac2bd470f3f`. The independent vLLM source copy
changes only the multiline f-string quoting in `sharded_state_loader.py`.
The complete original and changed file ASTs are equal under Python 3.12;
all 7701 source-tree file identities are checked against that retained
compatibility copy. No official arithmetic or class implementation is
changed. [AST evidence](../whole-layer-composition-source-audit/ast-equivalence.json)

[The fixture configuration](vendor_hook_glm52_config.py) retains the published
GLM-5.2 dimensions, RoPE parameters and layer pattern in the
[local configuration snapshot](../../../../../reference/mla-decode-fusion.md).
It expands the two abbreviated layer lists from that snapshot. With
`first_k_dense_replace=3`, `index_topk_freq=4`,
`index_skip_topk_offset=3` and `index_topk_pattern=None`, layer 6 is a MoE
layer with a full indexer; layer 7 is its shared-index successor. The actual
Transformers configuration produces 21 full-index and 57 shared-index layers.

The first device fixture constructs the real layer-6 attention class,
requires its actual `impl.skip_topk` to be false and uses synthetic nonzero
TP-shard weights. Sixteen real Gloo participants establish TP16 dimensions;
only rank 0 may initialize the single exposed NPU. Attention sets
`reduce_results=False`. Its result is the local projection before the TP
sum. Layer 7 is recorded as the successor, but is not exercised by this
fixture. Neither FFN nor either whole-decoder TP sum is covered.

The [device diff](device.patch) is relative to the retained real-outer
fixture. Its eager and official ACLGraphWrapper comparison modules are
unchanged. Graph replay retains strong output owners, claims the workspace
reserve on the actual capture stream, changes input/length/page/slot
buffers, compares every cache and ordered top-k, and rejects Python body
execution during replay.

## Frozen device entry

All files are under the independent host directory
`/home/shanshan/tilert-megakernel-20261021/vendor-hook-runtime212`.
[The staging manifest](staging-manifest.json) and
[launch-time hash list](frozen-inputs.sha256) retain source and library
identities. The original TorchNPU 2.10 runtime remains separate.

```bash
bash /home/shanshan/tilert-megakernel-20261021/vendor-hook-runtime212/tests/glm52/whole_layer/vendor_hook_runtime212_run.sh \
  4 vendor-hook-runtime212-layer6-m1-v1 \
  --mode int8 --forward-widths 1 --forward-replays 2
```

After that gate, use a new report name with `--graph --forward-widths 1 2
12 32 --forward-replays 8`; use `--mode bf16` for the unquantized indexer
cache. Every invocation still requires the device runner's fresh idle gate
and exclusive namespace admission. The launcher itself does not select or
reserve an idle device.

The full DecoderLayer has a separate extension boundary: the actual
residual RMSNorm imports `vllm_ascend.vllm_ascend_C` and calls
`_C_ascend::npu_add_rms_norm_bias`. The current runtime intentionally
contains only the independently built SFA adapter; the CPU report records
that the full extension and norm schema are absent. An attention gate
cannot establish that residual path, the FFN, or the retained residual
returned after the second TP sum.
