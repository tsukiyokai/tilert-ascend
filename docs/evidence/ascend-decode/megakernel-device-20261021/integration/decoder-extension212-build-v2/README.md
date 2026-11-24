# Official decoder host extension

The [build manifest](build/manifest.json) records a successful CPU build and
real import under Python 3.11, Torch/TorchNPU 2.12. The library SHA is
`acac5e7220b51bdd24e41352ffc52f88aac5f58dd9d262b0a43aad2ca9efd0fc`.
No NPU was exposed or initialized. This independently built subset retains
the complete original `camem_allocator.cpp` Python entry and the original
host functions, schemas and Meta functions for AddRmsNormBias, MoeGatingTopK,
GroupedMatmulSwigluQuantWeightNzV2 and DequantSwigluQuantV2. It is not a build
of the entire vLLM-Ascend extension.

The added shared-expert activation function is extracted unchanged from
vLLM-Ascend `8237ee6d690d1399bc6eb48ef5977ac2bd470f3f`,
`torch_binding.cpp:1948–2010`. The manifest records the complete source,
registration segment, generated unit, dependency and resolved API hashes.
Norm and TopK resolve to the private official oracle package
`b24a676805eac8a528fd6534ad890c038fd68e01e28935f2cb9d22d9f6f53cda`.
The new activation's execution and workspace functions both resolve to
CANN 9.2 `libopapi.so`
`347437ef3aaf109b4088af90763a2330fb66bb5278e63245366afa8fd7a2233a`.
The original Meta activation returns INT8 `[12,128]` and FP32 `[12]` for an
INT32 `[12,256]` input. Meta checks establish shapes and registration only.

The [actual CPU worker and shared-MLP constructor probe](cpu-packed-mapping/report.json)
uses the same quantization description before and after the original
`configure_quant_config` call. Without the model loader's packed mapping,
Gate-Up is BF16. With `GlmMoeDsaForCausalLM.packed_modules_mapping`, both
shared projections are INT8 and use `AscendW8A8DynamicLinearMethod`.
This probe uses real TP1 Gloo groups and no NPU. The device fixture separately
requires TP16 shapes and the actual `A8_INT_FUSED` path.

The previous three-operator library and constructor evidence remain in
[v1](../decoder-extension212-build-v1/manifest.json). Binary objects are
retained outside the repository; [their manifest](binary-artifacts.json)
records local paths, byte sizes and hashes.
