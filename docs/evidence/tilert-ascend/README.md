# TileRT wheel ABI evidence

Artifact: `tilert-0.1.6.post3-cp312-cp312-manylinux_2_28_x86_64.whl`, 8,331,800 bytes, SHA256 `d6fbf0a55be1fbde8869cb31412f33bfd4a82a2a73786651b61ab8dfff69de25`. Static extraction ran on the CPU side of `a3-gs`; no library was loaded and no accelerator workload ran. All 180 hashed wheel RECORD entries matched their contents.

## Operator inventory

| Library | Registered schema definitions recovered | SHA256 |
| --- | ---: | --- |
| `libtilert_dsv32.so` | 96 | `c9cb455555fe4ace2c0d12c0f492dcdc64f44c2738e7fdd68bbcf6bee5ed4b7b` |
| `libtilert_glm5.so` | 37 | `c09053e372a379e7575515e3be2ce84414b52fd082f3698b4e260464d4d5d632` |
| `libtilert_glm52_rocm.so` | 48 | `4f9b74ef16e479042616cb43c0389078bd5bd7a3069f4ef63b7d186d8e1f98de` |

There are 181 variant/schema entries and 175 distinct operator names. The CUDA libraries share `supported_models`, `supported_mla_models`, `set_use_raw_fuse`, `tilert_init_op`, and `tilert_force_init_op`. The ROCm library shares only `supported_models` with them. A process loads one backend, enforced by [the Python loader](../../../tilert/__init__.py#L58); the union is not one simultaneously loadable API.

- [schemas.tsv](schemas.tsv): every name, full schema, ELF file offset, virtual address, and recovery method.
- [inventory.json](inventory.json): the same definitions with code references, declared mutation arguments, wheel Python references, metadata, and selected native constraint strings.
- [glm52-contracts.md](glm52-contracts.md): all 48 GLM5.2 entries and the Ascend adaptation boundary.
- [glm52-native-limits.txt](glm52-native-limits.txt): preparation registration, TP8/MTP checks, and selected failure strings with native addresses.
- [schema-completeness.json](schema-completeness.json): arrow-fragment audit and direct registration/parser call sites for all three libraries.
- `libtilert_*-elf.txt`: ELF headers, dependencies, build IDs where present, and dynamic symbols.
- `libtilert_*-registration.txt`: schema references in native registration code; the GLM5 file also contains the four assembled-schema instruction sequences.

All three libraries are stripped x86-64 ELF shared objects. `nm -D --defined-only` returns no definitions for each library. Their undefined C++ symbols import `torch::Library::_def`, `_impl`, and schema parsers; these imports are not TileRT operator names. Internal kernel names and RTTI strings are also not exported Torch operators. Torch registration recovers the interface even when ELF exports are hidden.

The CUDA variants require Torch 2.11 CUDA and link CUDA 13 libraries; GLM5.2 requires Torch 2.12 ROCm and links `libamdhip64.so.7`, `libtorch_hip.so`, and `libc10_hip.so`. The wheel tag describes the Python/host platform, not interchangeable device backends. The package-level `torch>=2.11,<2.13` requirement is narrowed by [the backend loader](../../../tilert/__init__.py#L35).

## Extraction and scope

The extractor reads the host `.rodata`, recovers complete schema literals, and records native code references to each literal. GLM5 has four additional schemas constructed from 16-byte constants and immediate stores. They are reconstructed only for the exact GLM5 SHA256 above: `projq_wkvb__convert_weights`, `projq_wkvb__shard_weights`, `projx_wis__convert_weights`, and `projx_wis__shard_weights`. Their `parseSchema` calls are at `0x9cfda`, `0x9d612`, `0xa49c5`, and `0xa5002`; their `_def` calls are at `0x9d00b`, `0x9d640`, `0xa49f6`, and `0xa5030`. Raw `strings` alone undercounts this library. Every recovered schema has a native code reference.

An independent scan covered every ` -> ` occurrence in host `.rodata`, including constant fragments and strings shorter than the extractor's five-byte threshold. DSV3.2 has 96 arrows in 96 complete schemas. GLM5 has 33 complete schemas and a contiguous constant block containing three arrows; native construction recovers four schemas from that block, with the fourth reusing an existing 16-byte tail. GLM5.2 has 48 complete schemas plus a standalone four-byte ` -> ` string at `0x1e6a7`, referenced by state/peer-access failure messages ending in `torchCheckFail`. No schema-like tuple, optional, or alias return signature was omitted by the return-type pattern, and no unspaced schema-arrow candidate was found.

| Library | Recovered schemas | Direct `_def` call sites | Direct `parseSchema` call sites | Direct `parseSchemaOrName` call sites |
| --- | ---: | ---: | ---: | ---: |
| DSV3.2 | 96 | 14 | 1 | 2 |
| GLM5 | 37 | 25 | 8 | 1 |
| GLM5.2 ROCm | 48 | 21 | 1 | 1 |

Registration helpers are reused, so direct call-site counts do not prove one-to-one registration coverage. The inventory exhausts the recovered host schema literals and reconstructed constants, with no unexplained schema-like arrow fragment; constructor execution and runtime dispatcher enumeration remain unobserved. Exact dtype, dimensions, strides, pointer ownership, stream ordering, or global state effects do not follow from a `Tensor` schema. The contract note separates wrapper-established facts from fields whose complete native contract remains unknown. No numerical validation was performed.

The wheel contains 174 Python files. Against repository commit `f554fa57d0a69e7a44ac4f3134d0134b4964e848`, 159 existing files have identical ASTs, 14 differ semantically, and `tilert/pd_vllm/prompt_cache.py` is absent. All GLM5.2 model/operator files have identical ASTs; repository line links in the contract note refer to their formatted copies. The PD differences are recorded in [wrapper-source-map.json](wrapper-source-map.json). Wheel references in `inventory.json` retain wheel line numbers. Standalone GLM5 Python wrappers contain calls absent from the GLM5 library even though the DSV3.2 library registers those names; wrapper presence alone therefore does not establish per-backend coverage.

## Reproduce

Run the extraction on the server CPU with the exact wheel. The script uses only the Python standard library and binutils and never imports Torch or TileRT.

```bash
python extract_wheel_abi.py \
  tilert-0.1.6.post3-cp312-cp312-manylinux_2_28_x86_64.whl \
  abi-output
```

Server artifact directory: `/home/shanshan/trt-asc/research/tilert-abi-0.1.6.post3`. Original wheel: `/home/shanshan/note/proj/ai-infra/llm/infer/ultra-low-latency/codes/tile-ai/TileRT/bin/tilert-0.1.6.post3-cp312-cp312-manylinux_2_28_x86_64.whl`. The GLM5 hash also matches the earlier `tilert-glm5-20261014-COk05W` investigation; its CUDA graph findings apply to that GLM5 library, not automatically to GLM5.2 ROCm.
