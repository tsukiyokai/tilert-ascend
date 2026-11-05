# Ascend custom-operator packaging — what a vendor package is, and what builds one

Source material for getting a hand-written AscendC kernel into the running inference graph.
Unlike the rest of this directory, the material here is not published documentation: it is read
off the machines we run on, 2026-10-20, read-only. Every claim carries the absolute path it came
from so a later reader can re-check it without repeating the search.

Three trees were read:

| tree | what it is |
| --- | --- |
| `/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer` on 7.150.10.194, inside the `trt-asc` container | 54 MB vendor package, 44 operators, currently prepended to `ASCEND_CUSTOM_OPP_PATH` by benchmark arms. It is **this project's own build output**, produced by `tests/glm52/build_gmm1_barrier.sh` |
| `/opt/ascend/cann-9.2.0` in the same container | CANN 9.2.0: `bin/opc`, `bin/msopgen`, `python/site-packages/asc_opc_tool/`, `tools/op_project_templates/`, and the built-in opp |
| `/workspace/experiments/gmm-fastpath-20261018/prod-kernel/source/csrc` | the full vLLM-Ascend C++ tree the package is built from — `op_host/`, `op_kernel/`, `torch_binding.cpp`, `build.sh`, `scripts/opgen/` |

Verified means read in one of those three trees. Inference is labelled inference.

## Relevance

The routed-expert kernel (`tests/glm52/routed/`) measures 41.990 µs a layer against the vendor
pair's 165.2 and is not integrated, so none of it has reached TPOT. This file is the material for
the integration decision. Its short form: the packaging channel is `opc`, not `msopgen`; it
overrides **one operator at a time and cannot cross an operator boundary**; and it is not the
cheapest path for this kernel, because this kernel replaces two operators. See §10 for the
assumptions that changes.

---

## 1. Three layers define one operator, and the leftmost custom one wins

An operator type resolves against three independent registries, in this order:

1. every directory named in `ASCEND_CUSTOM_OPP_PATH`, left to right;
2. `${ASCEND_OPP_PATH}/vendors/<name>` for each name in `${ASCEND_OPP_PATH}/vendors/config.ini`'s
   `load_priority=` list, left to right;
3. `${ASCEND_OPP_PATH}/built-in/op_impl/...`.

`GroupedMatmulSwigluQuantV2` exists at layer 1 (twice over) and at layer 3
(`/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910_93/ops_transformer/grouped_matmul_swiglu_quant_v2.json`),
so the custom package genuinely shadows a built-in.

Layer 2 does not exist on this host: `/opt/ascend/cann-9.2.0/opp/vendors/` is an empty directory
and `find /opt/ascend -name config.ini` returns nothing. Everything custom arrives through the
environment variable.

Two things set it. The session activation, `/workspace/environment/activate.sh:18`:

```bash
export ASCEND_CUSTOM_OPP_PATH=/workspace/offline/cann-workspace-owner-20261017/kernel-overlay/vendors/custom_transformer:/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
```

Two complete 44-operator packages, both named `custom_transformer`, distinguished only by
position. And vllm-ascend itself, lazily in-process,
`/workspace/src/vllm-ascend-unified/vllm_ascend/utils.py:446-463`:

```python
def _prepend_env_path(env_name: str, path: str) -> None:
    current_value = os.environ.get(env_name, "")
    path_entries = [entry for entry in current_value.split(":") if entry]
    if path not in path_entries:
        path_entries.insert(0, path)
        os.environ[env_name] = ":".join(path_entries)
```

Prepend, idempotent, called from `platform.py:133`, `platform.py:305` and `utils.py:567/582`
(`enable_custom_op()`), not at import.

### The resolver, and the log lines that prove it

`libnnopbase.so` is the library that resolves an aclnn call to a kernel binary. Its strings, from
`op_kernel_lib.cpp` and `aicpu_json_load_manager.cpp`:

```
MM_ENV_ASCEND_CUSTOM_OPP_PATH / ASCEND_CUSTOM_OPP_PATH is null.
/op_impl/ai_core/tbe/      Custom impl path: %s, real path: %s.
/vendors/config.ini        Config.ini file path [%s] is invalid or does not exist.
priority                   Custom file dir [%s] is null, skip vendor [%s].
Checking vendor %s cust json path: %s
Can not find json file: %s, continue next priority
Use ASCEND_CUSTOM_OPP_PATH for custom operator loading.
Fallback to vendors path derived from ASCEND_OPP_PATH.
[%s] of operator [%s] is duplicated; discarding in favor of existing entry.
%s binInfo already exists, no need to add.
```

`continue next priority` is per json **file**, so fall-through is per operator — a package may
carry one operator and inherit the rest. `discarding in favor of existing entry` is the collision
rule: first wins.

Confirmed on a real four-vendor run rather than reconstructed. The plogs under
`/workspace/experiments/gmm2-compact-20261020/key-logs/debug/plog/` show
`aicpu_json_load_manager.cpp:224` reading `Get custom opp path size = 4.`, the paths logged in
exact left-to-right environment order by `op_kernel_lib.cpp:170 GetCustomImplPath`, and then

```
[SparseAttnSharedkvMetadata] of operator [/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer] is duplicated; discarding in favor of existing entry.
```

— the later colon-list entry discarded, the earlier retained.

`load_priority`'s orientation is fixed by the installer,
`/opt/ascend/cann-9.2.0/tools/op_project_templates/ascendc/customize/scripts/install.sh:314-332`
(byte-identical to `/workspace/src/vllm-ascend-unified/csrc/cmake/scripts/custom/install.sh`):

```bash
        echo "load_priority=$vendor_name" > ${config_file}
        ...
        sed -i "/load_priority=$found_vendors/s@…@load_priority=$vendor_name,$vendor@g" "$config_file"
```

New vendor at the head: leftmost is highest, most-recently-installed wins. That branch is the
`else` of `if [ -n "${INSTALL_PATH}" ]`, which is why a package installed with `--install-path`
— like both of ours — has no `config.ini` at all.

**Inference, three items.** That `ASCEND_CUSTOM_OPP_PATH` outranks `config.ini` follows only from
collection order in one initialization (environment at `op_kernel_lib.cpp:170` before `config.ini`
at `:233`) and could not be observed here, since no `config.ini` exists. Separately,
`op_kernel_lib.cpp:372` parses each vendor's `aic-<soc>-ops-info.json` in **reverse** environment
order while `fusion_binary_info.cc:69` in `libaicore_utils.so` reads `binary_info_config.json`
forward; either the reversal is deliberate so that last-write-wins reproduces first-entry-wins, or
it is incidental, in which case op-info registration and kernel-binary registration could resolve a
two-vendor conflict in favour of different vendors in one process. Finally GE and FE each carry a
reader hard-wired to the base path (`plugin_manager.cc:205 GetOppPluginVendors`,
`configuration.cc:814 GetCustomOppPathFromOppConfigFile`), both of which simply fail to open the
non-existent base `config.ini` here.

---

## 2. The vendor package, file by file

```
custom_transformer/
├── version.info                    # custom_opp_compiler_version=9.2.0   <- the only self-identification
├── bin/set_env.bash
├── scripts/uninstall.sh            # carries vendor_name=custom_transformer
├── op_api/                                       1.4M
│   ├── include/aclnnop/*.h                       57 headers
│   └── lib/libcust_opapi.so                      1142864 B, aarch64, stripped
├── op_proto/                                     1.3M
│   ├── inc/*_proto.h                             44 headers
│   └── lib/linux/aarch64/libcust_opsproto_rt2.0.so   1116496 B, stripped
└── op_impl/                                      51M
    ├── ai_core/tbe/
    │   ├── config/ascend910_93/aic-ascend910_93-ops-info.json      44 ops
    │   ├── custom_transformer_impl/              15M
    │   │   ├── ascendc/<op>/*.cpp,*.h            47 dirs — the AscendC SOURCE ships inside the package
    │   │   └── dynamic/<op>.py                   52 opc wrapper modules
    │   ├── kernel/                               27M
    │   │   ├── ascend910_93/<op>/<Name>_<hash>.{o,json}            44 dirs
    │   │   └── config/ascend910_93/{<op>.json, binary_info_config.json,
    │   │                            relocatable_kernel_info_config.json}
    │   └── op_tiling/
    │       ├── liboptiling.so
    │       └── lib/linux/aarch64/libcust_opmaster_rt2.0.so
    └── cpu/
        ├── aicpu_kernel/impl/libtransformer_aicpu_kernels.so       5408584 B
        └── config/cust_aicpu_kernel.json                           3 ops, engine DNN_VM_AICPU
```

What each part is for:

- `version.info` and the directory name are the whole of the package's identity. There is no
  `config.ini`, no `*.cmake`, no `install.sh` and no `.run` inside it.
- `op_proto/` — the GE prototype: arity and attr names. See §7.
- `op_api/` — the aclnn entry points. See §7.
- `op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json` — what GE/FE match a
  node against: the supported dtype/format cross-product, as parallel comma lists.
- `op_impl/ai_core/tbe/custom_transformer_impl/ascendc/<op>/` — **the AscendC source, installed**.
  `opc` compiles from this copy, not from a git tree. This is the single most useful fact in the
  package: it means overriding an operator is editing files that are already in the package.
- `op_impl/ai_core/tbe/custom_transformer_impl/dynamic/<op>.py` — the TBE wrapper module `opc`
  takes as its first positional argument.
- `op_impl/ai_core/tbe/kernel/config/ascend910_93/<op>.json` — the binary lookup table. §3.
- `op_impl/ai_core/tbe/kernel/ascend910_93/<op>/<Name>_<hash>.{o,json}` — the compiled kernels. §3.4.
- `op_impl/ai_core/tbe/op_tiling/liboptiling.so` — the host tiling function. Not rebuilt by an
  `.o` swap, which means **a substituted kernel inherits the vendor's TilingData layout**.

### The `set_env.bash` trap

`bin/set_env.bash`, verbatim:

```bash
export ASCEND_CUSTOM_OPP_PATH=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/workspace/src/vllm-ascend-unified/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
```

It names its **install-time absolute path**, not itself. Sourcing a copied overlay's
`set_env.bash` activates the original package, silently. Set the variable by hand.

### 44 operators — a wholesale fork, not a surgical override

The count is 44, consistently, across four registries: 44 dirs under
`kernel/ascend910_93/`, 46 jsons under `kernel/config/ascend910_93/` (44 + `binary_info_config.json`
+ `relocatable_kernel_info_config.json`), 44 keys in `aic-ascend910_93-ops-info.json`, 44 headers in
`op_proto/inc/`. Both `grouped_matmul_swiglu_quant` and `grouped_matmul_swiglu_quant_v2` are
present; the project's builds have only ever touched **V2**.

A one-operator package would also work — fall-through is per json file (§1) — but neither live
package is one.

---

## 3. The binary config json: `binList`, `simplifiedKey`, `staticKey`

`op_impl/ai_core/tbe/kernel/config/ascend910_93/grouped_matmul_swiglu_quant_v2.json`, 32336 B,
7 entries. (`..._quant.json`, the V1 op, is 16953 B with 5.) The top level is one key:

```
{"binList": [ {...}, ... ]}
```

Per entry:

```
implMode            'high_performance'
int64Mode           False
simplifiedKeyMode   0
simplifiedKey       [ 4 strings ]
dynamicParamMode    'folded_with_desc'
staticKey           '<64 hex>,<64 hex>'
inputs              [ tensor-desc | [tensor-desc] | ... ]
outputs             [ tensor-desc, ... ]
attrs               [ {name,dtype,value}, ... ]
opMode              'dynamic'
optionalInputMode   'gen_placeholder'
optionalOutputMode  'gen_placeholder'
deterministic       'ignore'
binInfo             {'jsonFilePath': 'ascend910_93/grouped_matmul_swiglu_quant_v2/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.json'}
```

There is no `binFileName`, `binFileSuffix` or `supportInfo` at this level — those live in the
per-kernel json that `binInfo.jsonFilePath` names. The config json is purely a lookup table, and
`binInfo.jsonFilePath` is its only link to a binary: a path relative to the `kernel/` root, with
the `.o` beside the `.json` under the same basename.

**There are three files in the lookup, not two, and `simplifiedKey` must agree across all of them.**
`cmake/util/ascendc_ops_config.py` writes both `kernel/config/<soc>/<op_snake>.json` (the `binList`
above, carrying `staticKey`) and `kernel/config/<soc>/binary_info_config.json` (`params` plus
`binaryList`, keyed on `simplifiedKey`). `/opt/ascend/cann-9.2.0/lib64/libnnopbase.so` — the only
library containing the string `op_impl/ai_core/tbe/kernel/config/` — reads both, and the per-kernel
descriptor's `supportInfo` carries a third copy. Ship one without the others and lookup fails with
nnopbase's "simplifiedKey not found". For an override of an existing operator this costs nothing,
because `binary_info_config.json` describes parameter *shape* rather than the kernel and is left
untouched (§4.4); for a new operator all three must be generated together.

### 3.1 The operator's real signature

`binary_info_config.json`'s `GroupedMatmulSwigluQuantV2` entry is the parameter registry — per
tensor only `{name, index, paramType, formatMode}`:

| index | name | paramType |
| --- | --- | --- |
| 0 | `x` | required |
| 1 | `x_scale` | required |
| 2 | `group_list` | required |
| 3 | `weight` | **dynamic** (tensorlist) |
| 4 | `weight_scale` | **dynamic** |
| 5 | `weight_assist_matrix` | **dynamic** |
| 6 | `bias` | optional |
| 7 | `smooth_scale` | optional |

Outputs: `y`, `y_scale`. That matches the production capture's
`INT8;FLOAT;INT64;DT_INT4;UINT64;FLOAT` / `ND;ND;ND;FRACTAL_NZ;ND;ND` exactly, with the two
optional inputs absent. `weight_assist_matrix` is the W4A8 MSD correction term — the same
quantity `tilert/models/glm_5_2_ascend/reference_weights.py:365` checks as
`sum(w) * scale * 8` under the name `scale_bias`.

`aic-ascend910_93-ops-info.json` states admissibility as parallel comma lists of length 7, one
position per supported combination:

```json
 "input3": {"dtype": "int8,int8,int8,int4,int4,int4,int4",
            "format": "FRACTAL_NZ,FRACTAL_NZ,FRACTAL_NZ,ND,FRACTAL_NZ,FRACTAL_NZ,ND",
            "name": "weight", "paramType": "dynamic", "shape": "all"},
 "input4": {"dtype": "float32,bfloat16,float16,uint64,uint64,uint64,uint64", ...},
 "coreType": {"value": "AiCore"},
 "opFile": {"value": "grouped_matmul_swiglu_quant_v2"},
 "opInterface": {"value": "grouped_matmul_swiglu_quant_v2"},
 "prebuildPattern": {"value": "Opaque"},
 "dynamicCompileStatic": {"flag": "true"}, "dynamicFormat": {"flag": "true"},
 "dynamicRankSupport": {"flag": "true"}, "dynamicShapeSupport": {"flag": "true"},
 "needCheckSupport": {"flag": "false"}, "precision_reduce": {"flag": "true"}
```

`opFile.value` is what names the `custom_transformer_impl/dynamic/<name>.py` wrapper.

### 3.2 `simplifiedKey` — generated, and decodable

Verbatim, the V1 entry 0's four keys:

```
GroupedMatmulSwigluQuant/d=0,p=1/2,2/29,29/10,2/0,2/9,2/2,2/0,2
GroupedMatmulSwigluQuant/d=0,p=0/2,2/29,29/10,2/0,2/9,2/2,2/0,2
GroupedMatmulSwigluQuant/d=1,p=1/2,2/29,29/10,2/0,2/9,2/2,2/0,2
GroupedMatmulSwigluQuant/d=1,p=0/2,2/29,29/10,2/0,2/9,2/2,2/0,2
```

and V2 entry 6, the variant production calls and the project rebuilds:

```
GroupedMatmulSwigluQuantV2/d=0,p=1/2,2/0,2/9,2/29,29/10,2/0,2/2,2/0,2
```

The grammar, read off the generator at
`/opt/ascend/cann-9.2.0/python/site-packages/asc_opc_tool/simplified_key_utils.py`,
`assemble_simplified_key` at line 303:

```python
simplified_key_str = op_type + "/" + deterministic_str + "," + impl_mode_str
if input_list:  simplified_key_str = simplified_key_str + input_list[index]
if output_list: simplified_key_str = simplified_key_str + output_list[index]
if attr_str:    simplified_key_str = simplified_key_str + attr_str
```

and `generate_input_or_output_for_simpilified_key` at line 591:

```python
        if isinstance(input_or_output, list):            # dynamic / tensorlist
            dynamic_param_count = len(input_or_output)
            input_or_output = input_or_output[0]         # folded to the FIRST element only
        ...
        if param_type == OpParamType.OPT and (
            simplified_key_mode == 0 or ...):
            continue                                     # optional params OMITTED entirely
        inputs_or_outputs_str = (
            inputs_or_outputs_str + "/" + str(dtype_value) + "," + str(op_format_value))
```

So: `/`-separated fields; each field `<ge::DataType int>,<ge::Format int>`; present inputs in
declaration order, then all outputs. An absent optional parameter contributes **nothing, not even
a placeholder**. A dynamic (tensorlist) parameter collapses to one field. Attrs are dropped
unconditionally, because `generate_simplified_key_str` does `if simplified_key_mode == 0: attr_str = str()`.

The two prefixes:

- `d=0|1` — deterministic. `deterministic == "ignore"` emits **both**.
- `p=N` — impl mode, from the dict in `generate_impl_mode_for_simpilified_key`:
  `default:0, high_performance:1, high_precision:2, super_performance:3,
  support_out_of_bound_index:4, enable_float_32:5, enable_hi_float_32:6, keep_fp_16:7`.

Four keys per entry is therefore `deterministic: "ignore"` crossed with
`--impl_mode=high_performance,optional` — the literal `optional` is not in the dict; it sets
`optional_impl_mode_flag`, which appends `p=0`.

Enum tables are authoritative at
`/opt/ascend/cann-9.2.0/python/site-packages/asc_opc_tool/constant.py:330` and `:431` —
`DT_FLOAT:0, DT_FLOAT16:1, DT_INT8:2, DT_INT64:9, DT_UINT64:10, DT_BF16:27, DT_INT4:29`;
`FORMAT_ND:2, FORMAT_FRACTAL_NZ:29`.

Decoded against the V2 signature, `/2,2/0,2/9,2/29,29/10,2/0,2/2,2/0,2` reads

| field | parameter | dtype,format |
| --- | --- | --- |
| 1 | `x` | int8, ND |
| 2 | `x_scale` | float32, ND |
| 3 | `group_list` | int64, ND |
| 4 | `weight` (folded list) | int4, FRACTAL_NZ |
| 5 | `weight_scale` (folded) | uint64, ND |
| 6 | `weight_assist_matrix` (folded) | float32, ND |
| 7 | `y` | int8, ND |
| 8 | `y_scale` | float32, ND |

Differential confirmation, across entries that differ in exactly one position:

| entries | source difference | key difference |
| --- | --- | --- |
| V1 0 → 4 | `weight` FRACTAL_NZ → ND | `29,29` → `29,2` |
| V1 0 → 1 | `weight` int4→int8, `weight_scale` uint64→float32 | `29,29`→`2,29`; `10,2`→`0,2` |
| V1 1 → 2 | `weight_scale` float32 → float16 | `0,2` → `1,2` |
| V1 2 → 3 | `weight_scale` float16 → bfloat16 | `1,2` → `27,2` |
| V2 0 → 6 | `weight` int4/ND → int4/FRACTAL_NZ | field 4 `29,2` → `29,29` |
| V2 0 → 5 | `x` int8 → int4 | field 1 `2,2` → `29,2` |

### 3.3 `staticKey` — a sha256 per impl mode

Verbatim, V2 entry 6:

```
159cdc7fd700a5bb4d0095e9d240c2d91a918b092898423346ddee46fdf3ba52,e21859ae856a31330a414f48c6228cce71cf58838a0e5b7f30ca8affab764c54
```

Two 64-hex halves, one per impl mode in `--impl_mode=high_performance,optional`. The generator,
`/opt/ascend/cann-9.2.0/python/site-packages/asc_opc_tool/single_op_post_compile.py:255-314`:

```python
        static_key_json = {}
        # should be in alphabetical order to add in static_key_json, because reuse json is in this order
        attrs = self.__generate_op_attrs_static_key(...)
        super().gen_notnone_param(static_key_json, "attrs", attrs)
        build_options = self.__generate_op_build_options(op_info, impl_mode)
        super().gen_notnone_param(static_key_json, "buildOptions", build_options)
        inputs = self.__get_op_inputs_or_outputs_supportinfo(op_info.get(CompileParam.INPUTS), False)
        ...
        super().gen_notnone_param(static_key_json, "inputs", inputs)
        outputs = ...
        super().gen_notnone_param(static_key_json, "outputs", outputs)
        static_key_json_str = json.dumps(static_key_json, separators=(",", ":"))
        static_key = str_to_sha256_hash(static_key_json_str)
```

with `__generate_static_key` looping `impl_mode_str.split(",")` and joining with `","`. Per tensor
the hashed descriptor keeps only `dtype`, `format`, `shape`
(`__generate_single_tensor_supportinfo` with `is_support_info=False`); attrs keep `{dtype, value}`
with `value` nulled for `float`/`list_float` and for any attr carrying `value_range`/`value_list`.

An entry without static shapes still has one: every entry here is `opMode: "dynamic"` with
`shape: [-2]`, and all 5/7 carry a `staticKey`.

Four candidate canonical-JSON reconstructions failed to reproduce the digest byte-exactly in one
session, varying `buildOptions` and the optional-placeholder representation. The practical
consequence is the one that matters: **do not hand-compute either key.** Feed `opc` the same
`input-param.json` and it regenerates both bit-identically, which is exactly what the project's
build script asserts (§4.3).

### 3.4 The per-kernel descriptor

`op_impl/ai_core/tbe/kernel/ascend910_93/grouped_matmul_swiglu_quant_v2/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.json`,
verbatim with `supportInfo` elided:

```json
{
  "binFileName": "GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e",
  "binFileSuffix": ".o",
  "blockDim": -1,
  "coreType": "MIX",
  "intercoreSync": 1,
  "kernelName": "GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e",
  "magic": "RT_DEV_BINARY_MAGIC_ELF",
  "memoryStamping": [],
  "opParaSize": 360,
  "parameters": [null, null, null, null, null, null, null, null, null, null, null],
  "sha256": "baaa18e24c44695737b414d3196414152f1324caf0087c00cfc102162cf16bc5",
  "workspace": {"num": 1, "size": [-1], "type": [0]},
  "kernelList": [
    {"tilingKey": 2, "kernelType": "MIX_AIC", "crossCoreSync": 1, "taskRation": "1:2",
     "kernelName": "GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e_2"},
    {"tilingKey": 3, "kernelType": "MIX_AIC", "crossCoreSync": 1, "taskRation": "1:2",
     "kernelName": "GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e_3"}
  ],
  "taskRation": "tilingKey",
  "optionalInputMode": "gen_placeholder",
  "optionalOutputMode": "gen_placeholder",
  "supportSuperKernel": 1,
  "platformInfo": {"deterministicLevel": 0},
  "compileInfo": {},
  "dynamicParamMode": "folded_with_desc",
  "supportInfo": { ... a verbatim copy of the binList entry, incl. simplifiedKey and staticKey ... },
  "filePath": "ascend910_93/bin/grouped_matmul_swiglu_quant_v2/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.json"
}
```

Notes on the load-bearing fields:

- `blockDim: -1` and `workspace.size: [-1]` — both decided at run time by the host tiling
  function, not by the package.
- `opParaSize: 360` — the TilingData blob size. A substituted kernel must parse that same 360 bytes.
- `sha256` is the digest of the `.o`, computed at
  `/opt/ascend/cann-9.2.0/python/site-packages/tbe/tikcpp/compile_op.py:516`:
  `js["sha256"] = hashlib.sha256(obj_file.read()).hexdigest()`. Cross-checked:
  `prod-kernel/artifacts.sha256` records `baaa18e2…` for the `.o`, matching the json. **Recompute
  it on any `.o` swap.**
- `filePath` is added by the installer, not by `opc`, and its `ascend910_93/bin/...` component does
  not exist in this layout. It is vestigial: a key-by-key diff of raw `opc` output against the
  installed copy shows `filePath` as the only difference.

**Inference on whether `sha256` is enforced.** `libnnopbase.so` — the aclnn loader that resolves
`simplifiedKey` → `.o` — carries the strings `binFileName`, `binList`, `binInfo`, `jsonFilePath`,
`kernelList`, `simplifiedKey`, `taskRation`, but **not** `sha256`. The compile-side
`libop_compile_adapter.so` and `libfe.so` do carry it. Reading: the pure-aclnn dispatch path
probably does not verify it, the graph-compile path may. This is a string inventory, not
disassembly. Recompute regardless.

### 3.5 The filename hash is an input, not a derivation

`fa3d6d3de6e1f32e170e39d2ddd3a20e` is not a function of the signature. The built-in
`grouped_matmul_swiglu_quant_v2.json` carries the **identical seven `simplifiedKey` strings** under
seven completely different hashes (`02481f20…`, `ed7aa5e4…`, …), and `fa3d6d3d…` appears nowhere in
the built-in tree. `opc` takes the name as a parameter,
`/opt/ascend/cann-9.2.0/python/site-packages/asc_opc_tool/op_info_parser.py:594-599`:

```python
            bin_filename = self.__op_info_dict.get(OpcOptions.BIN_FILENAME)
            if bin_filename is not None:
                logger.debug("Bin_filename is not none, use it as kernel name: %s.", bin_filename)
                return bin_filename
```

which is why a rebuild keeps the name: the build script copies `binFileName` out of the original
descriptor into `input-param.json`.

---

## 4. What produces the `.o` and its `.json`: `opc`, and nothing else

### 4.1 The tool

`opc` is the whole toolchain for overriding an operator that already exists. No CMake, no
`msopgen` project, no `.run` rebuild. It takes the wrapper `.py` from inside the vendor package,
compiles the AscendC sources from inside the vendor package, and emits one `.o` plus one `.json`.

From `opc --help`, the load-bearing lines verbatim:

```
[Output]
  --output            Output file path(needn't suffix, will add .o and .json automatically).
[Input]
  op_path             The path of operator file.
  --input_param       The path of input json file.
  --main_func         The compilation function in operator's implementation file.
  --bin_filename      The name of op kernel.
  --simplified_key_mode       Set how to generate simplified_key.
                              None(default): Don't generate simplified_key.
                              0: generate simplified_key, simple mode
                              1: generate simplified key, compitable mode
  --dynamic_param_mode        Whether to add count for dynamic input/output in simplified key.
                              None: default / unfolded: add count / folded_with_desc: with shape, don't add count
  --op_mode           Set op mode (Options are dynamic, static and pre-static).
  --tiling_key        Set tiling key list for op, default is None.  For expl: --tiling_key=1,2,3,4
  --op_super_kernel_options         Set super kernel options.
  --op_relocatable_kernel_binary    Set relocatable bin mode.
```

`--output` producing both the `.o` and its paired `.json` is the direct answer to "what produces
them". `--simplified_key_mode` being an `opc` flag is the direct answer to whether `simplifiedKey`
must be written by hand: it must not be.

**`opc` needs three things beyond its arguments, and one of them is expensive.** The first
positional is a generated Python wrapper, not the kernel `.cpp`. In addition:

- `BUILD_KERNEL_SRC=<kernel.cpp>` must be in the environment — parsed from `--kernel-src=` by
  `cmake/util/const_var.py:43` (`SRC_ENV`).
- `ASCEND_CUSTOM_OPP_PATH` must point at a tree containing
  `op_impl/ai_core/tbe/op_tiling/liboptiling.so`, because
  `/opt/ascend/cann-9.2.0/python/site-packages/tbe/tikcpp/get_op_tiling.py:87` sets `_TILING_SO_PATH`
  to exactly that, `ctypes.CDLL`s it, and calls `do_op_tiling` **during the compile** to obtain
  `tiling_data_size`. So a working host tiling library is on the critical path even when all you
  want is the `.o`. For an override that reuses the vendor's TilingData this is free — the library
  is already in the package. For a kernel with a *different* TilingData it is not: you must build
  `liboptiling.so` first, which means the `op_host/` half of a project.
- `aic-<soc>-ops-info.json` must be staged under `op_impl/ai_core/tbe/config/<soc>/` in the same tree.

And there are two shims, not one. `/opt/ascend/cann-9.2.0/bin/opc` → `opc_tool/opc.py` is the legacy
driver (called at `fwk_modules/util/ascendc_bin_param_build.py:425`);
`/opt/ascend/cann-9.2.0/bin/asc_opc` → `asc_opc_tool/opc.py` is the modern one (`:527`). The
project's build scripts use `opc`; the generators quoted in §3.2-§3.3 live under `asc_opc_tool`.

**A trap worth its own line.** `which opc` in a bare shell resolves to
`/usr/local/Ascend/cann-9.1.0/bin/opc` — the container has two CANN installs. After
`source /workspace/environment/activate.sh` it resolves to `/opt/ascend/cann-9.2.0/bin/opc` with
`ASCEND_HOME_PATH=/opt/ascend/cann-9.2.0`. The activation unsets the toolkit variables first
(`activate.sh:6`) and then sources `/opt/ascend/set_env.sh` (`:12`). Both project build scripts
guard on this: `: "${ASCEND_HOME_PATH:?Source the CANN/Python activation before this script}"`.

### 4.2 The invocation, verbatim from the project's own script

`tests/glm52/build_gmm1_barrier.sh` is the maintained recipe; the copy that built the live overlay
is at `/workspace/experiments/gmm-compact-20261020/build-tree/tests/glm52/build_gmm1_barrier.sh`.

```bash
op=grouped_matmul_swiglu_quant_v2
variant=GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e
original_vendor="$source_dir/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
binary_dir="op_impl/ai_core/tbe/kernel/ascend910_93/$op"
installed_headers="op_impl/ai_core/tbe/custom_transformer_impl/ascendc/$op"
wrapper="op_impl/ai_core/tbe/custom_transformer_impl/dynamic/$op.py"

# 1. copy the whole vendor package, then overwrite the AscendC sources inside it
overlay="$output_dir/overlay/vendors/custom_transformer"
cp -a --reflink=auto "$original_vendor" "$overlay"
for header in "${headers[@]}"; do cp "$build_source/$header" "$overlay/$installed_headers/"; done
# opc compiles the entry from the installed copy, so a patch that touches it
# takes effect only once that copy is replaced too. Without this the build
# succeeds, the kernel is byte-identical to the unpatched one, and the
# measurement reports no difference.
cp "$build_source/gmm/$op/op_kernel/$op.cpp" "$overlay/$installed_headers/"

# 2. derive input-param.json from the EXISTING kernel descriptor
descriptor = json.loads(Path(sys.argv[1]).read_text())
support = descriptor["supportInfo"]
parameters = {"op_type": "GroupedMatmulSwigluQuantV2", "op_list": [{
    "bin_filename": descriptor["binFileName"],
    **{key: support[key] for key in ("inputs", "outputs", "attrs")}}]}

# 3. recompile exactly one kernel binary
(
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    export CPATH="$build_source/common/include:$build_source/third_party/catlass/include${CPATH:+:$CPATH}"
    export ASCEND_CUSTOM_OPP_PATH="$overlay${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export HI_PYTHON=python3 TILINGKEY_PAR_COMPILE=1 BIN_FILENAME_HASHED=1
    opc "$overlay/$wrapper" --main_func="$op" \
        --input_param="$output_dir/input-param.json" --soc_version=Ascend910_9391 \
        --output="$kernel_output" --impl_mode=high_performance,optional \
        --simplified_key_mode=0 --op_mode=dynamic
) 2>&1 | tee "$output_dir/build.log"
```

Reading it out:

- The first positional argument is the **wrapper `.py` inside the overlay**, and `--main_func` is
  the function in it.
- `input-param.json` is `{op_type, op_list:[{bin_filename, inputs, outputs, attrs}]}`, lifted
  straight out of the original descriptor's `supportInfo`. Nothing is authored.
- `--soc_version=Ascend910_9391` is A3.
- `--impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic` are exactly
  the knobs that produce the four `simplifiedKey` strings and the two-half `staticKey`.
- `BIN_FILENAME_HASHED=1` suppresses the `_<pid>` suffix
  (`asc_opc_tool/ascendc_common_utility.py:473-480`, `get_distinct_filename_tag`), making output
  names deterministic.
- `ASCEND_CUSTOM_OPP_PATH` must point at the overlay **during the build**: `opc` resolves the op's
  tiling and info from the package it is extending.
- `CPATH` carries the vendor's own include trees, here `common/include` and
  `third_party/catlass/include`.

### 4.3 The graft, and the assertion that is the real contract

```python
original, compiled = (json.loads(Path(path).read_text()) for path in sys.argv[1:3])
with Path(sys.argv[3]).open("rb") as stream:
    elf_sha = hashlib.file_digest(stream, "sha256").hexdigest()
assert compiled["sha256"] == elf_sha
for key in original.keys() | compiled.keys():
    if key not in ("sha256", "filePath"):
        assert original.get(key) == compiled.get(key), key
original["sha256"] = elf_sha
Path(sys.argv[4]).write_text(json.dumps(original, indent=2) + "\n")
```

Every key of the descriptor must come out identical except `sha256` and `filePath`. The build does
not edit `simplifiedKey` or `staticKey`; it **proves `opc` reproduced them**. That assertion is
the thing to copy into any new build script.

### 4.4 The minimal override, as a list

1. `cp -a` the vendor package to a new directory.
2. Overwrite `op_impl/ai_core/tbe/custom_transformer_impl/ascendc/<op>/` — **including the `.cpp`
   entry file, not only the headers**.
3. Build `input-param.json` from the existing descriptor's `supportInfo` plus its `binFileName`.
4. Run `opc` as above, with the overlay on `ASCEND_CUSTOM_OPP_PATH`.
5. Assert every descriptor key but `sha256`/`filePath` is unchanged; write the new `sha256`.
6. Copy the `.o` in.
7. Prepend the copy to `ASCEND_CUSTOM_OPP_PATH`.

Nothing else in the 54 MB changes: not `op_proto`, not `op_api`, not `liboptiling.so`, not
`aic-ascend910_93-ops-info.json`, not `binary_info_config.json`.

---

## 5. The mix ratio lives in the device source, not in the op definition

Three places carry 1 cube : 2 vector; only one is the source of truth.

Source of truth —
`op_impl/ai_core/tbe/custom_transformer_impl/ascendc/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2.cpp`,
one macro per tiling key at lines 37, 58, 76 and 97 for keys 2, 4, 5, 3:

```cpp
extern "C" __global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, ..., GM_ADDR tiling)
{
    ...
    if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
```

Recorded by `opc` as `kernelList[i].kernelType = "MIX_AIC"` with `taskRation = "1:2"`, plus
top-level `taskRation = "tilingKey"` meaning per-key and `coreType = "MIX"`.

And **not** declared on the host side at all. `source/csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_def.cpp`
ends:

```cpp
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        ...
        config950. ... .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value","grouped_matmul_swiglu_quant_v2_apt");
        this->AICore().AddConfig("ascend950", config950);
    }
};
OP_ADD(GroupedMatmulSwigluQuantV2);
```

No mix-ratio field. The ratio rides along in the `.cpp` and is re-derived into the descriptor.

Package-wide inventory: `taskRation` appears 256× as `"1:2"`, 80× as `"tilingKey"`, 20× as
`"0:1"`; `kernelType` is `"MIX_AIC"` in all 276 occurrences; `coreType` is `"VectorCore"` 90× and
`"MIX"` 80×. `mixType` and `MIX_AIV` appear nowhere except `transpose_kv_cache_by_block`.
`AiCoreType` appears nowhere; the related spelling is `coreType.value = "AiCore"`.

Across every SoC in `/opt/ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe/config/`,
`coreType.value` is `AiCore` 2053 times, `AiCore,VectorCore` 16 times, `VectorCore` once, and
`MIX` **never**; `taskRation.value` does not appear anywhere in the installation. The host side has
no say in mix, and no `ExtendCfgInfo` substitutes for the in-body macro.

### 5.1 The op-project build does not compile your entry as a kernel — so `__mix__` must go

This is the most consequential single fact about moving a `<<<>>>` kernel into an operator project,
and it runs opposite to the obvious expectation.
`/opt/ascend/cann-9.2.0/python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py:784-830`
synthesises the translation unit it actually compiles, verbatim fragments:

```python
source += "#undef __global__\n"
source += "#define __global__ inline\n"
source += f'#include "{src_file}"\n'
source += '#include "kernel_common.h"\n'
source += "#undef __global__\n"
...
source += "#define __global__ __attribute__((cce_kernel))\n"
...
auto_gen_kernel_func = f"auto_gen_{func_name}_kernel"
gen_func_attributes = "__global__"
...
    gen_func_attributes += " [aicore]"
kernel_func_dec = f'extern "C" {gen_func_attributes} void {auto_gen_kernel_func}('
compile_info.global_kernel_attribute = gen_func_attributes
kernel_func_dec_pub = f"__aicore__ inline __attribute__((always_inline)) void ascendc_{auto_gen_kernel_func}("
```

Your `.cpp` is textually included with `__global__` redefined to `inline`, so your entry becomes an
always-inlined device function. The real `cce_kernel` is the generated
`auto_gen_<func>_kernel`, whose attribute list is exactly `__global__ [aicore]`.
`global_kernel_attribute` is written once at `:828` and thereafter only read (`:1711-1717`,
`static_compile_resource_generator.py:876-877`); no path appends `core_ratio` or
`cce_schedule_mode`.

Consequences, both of which are costs the project pays and not conveniences it gains:

- `__mix__(1, 2)` ends up on a non-kernel function, which bisheng rejects by its own diagnostic —
  「`__attribute__((aic)), __attribute__((aiv)), bisheng::core_ratio attribute needs to appear with
  kernel func.`」 Expect a hard error, not a silent downgrade. Delete it.
- `__schedmode__(1)` on an inlined callee has nothing to propagate it to the generated entry. It is
  dropped. Delete it. The framework conveys the equivalent through the descriptor instead —
  `intercoreSync: 1` at top level and `kernelList[].crossCoreSync: 1` per key (§3.4) — which is why
  the vendor's own `CrossCoreSetFlag` kernel works without the attribute. **That substitution is
  inference from the descriptor's fields, and it is the first thing to verify on a first build**,
  because `README.md`'s correction list records `__schedmode__(1)` as *required* for cross-core
  sync, with the failure signature "a launch that never returns at 250-360 W and 0% utilization".
- The ratio is not handed to the kernel body by a macro either. `ascendc_compile_v220.py:309-315`
  defines `-D__MIX_CORE_MACRO__=1` for both `MIX_AIC_1_1` and `MIX_AIC_1_2`, but
  `-D__MIX_CORE_AIC_RATION__=1` only for `1_1`. It is read from hardware,
  `/opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_sys_var_impl.h:43-54`:

  ```c
  __aicore__ inline int64_t GetTaskRationImpl()
  {
      if ASCEND_IS_AIC { return 1; }
      else { return get_subblockdim(); }   // 1 in aic, 2 in aiv
  }
  ```

  so inside a repackaged kernel, `ASCEND_IS_AIC` / `ASCEND_IS_AIV` plus `GetTaskRationImpl()` and
  `GetBlockIdxImpl()` replace anything derived from `__mix__`'s operands. `GetBlockIdxImpl` already
  folds the ratio in: `get_block_idx() * GetTaskRationImpl() + get_subblockid()` on AIV.

Verification order on a first build, cheapest first: the per-kernel `.json` must carry
`"coreType": "MIX"`, `"taskRation": "tilingKey"` (or `"1:2"` if no per-key types are set), and a
`kernelList` entry per tiling key with `"kernelType": "MIX_AIC"`, `"taskRation": "1:2"`,
`"crossCoreSync": 1`; the `.o` must contain the
`<binFileName>_<tilingKey>_mix_aic` / `_mix_aiv` symbol pair and the matching `.ascend.meta.*`
sections. Build once with `__mix__(1,2)` retained and once removed, and diff those fields: the
macro-driven values are identical either way, which is the point.

### 5.2 Both `KERNEL_TASK_TYPE` forms are no-ops outside a project — and the keyed one is narrower

The macros are in
`/opt/ascend/cann-9.2.0/aarch64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h`, reached
transitively from `kernel_operator.h`:

```c
// :89-101, guarded by __PLUGIN__KERNEL_META_TYPE_ENUME_DEFINED__
enum KernelMetaType : uint8_t { KERNEL_TYPE_AIV_ONLY, KERNEL_TYPE_AIC_ONLY,
  KERNEL_TYPE_MIX_AIV_1_0, KERNEL_TYPE_MIX_AIC_1_0, KERNEL_TYPE_MIX_AIC_1_1,
  KERNEL_TYPE_MIX_AIC_1_2, KERNEL_TYPE_AICORE, KERNEL_TYPE_VECTORCORE,
  KERNEL_TYPE_MIX_AICORE, KERNEL_TYPE_MIX_VECTOR_CORE, KERNEL_TYPE_MAX, };

// :365-373
#ifdef __CHECK_FEATURE_AT_PRECOMPILE
#define ENABLE_FEATURE_FOR_COMPILE(f, val) auto __enable_feature_for_compile_##f = val
#else
#define ENABLE_FEATURE_FOR_COMPILE(f, val)
#endif

// :377-387
#define KERNEL_TASK_TYPE(key, value)    ENABLE_FEATURE_FOR_COMPILE(key, value)
#define KERNEL_TASK_TYPE_DEFAULT(value) ENABLE_FEATURE_FOR_COMPILE(default, value)
```

also `:104-113` `enum KernelType { K_TYPE_AICORE=1, K_TYPE_AIC=2, K_TYPE_AIV=3,
K_TYPE_MIX_AIC_MAIN=4, K_TYPE_MIX_AIV_MAIN=5, … }` — the value that reaches the ELF — and
`:156-168` / `:304-318`, `FuncMetaType` with `F_TYPE_MIX_TASK_RATION=3` and
`struct FunMetaMixCoreType { BaseTlv head; unsigned short taskRation0; unsigned short taskRation1; }`.
Both macros are defined empty when `__NPU_ARCH__` is 5101/5161/5165/5163.

A second, competing definition sits in the compiler's own resource directory,
`/opt/ascend/cann-9.2.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_aicore_functions.h:40-58`:
a **differently numbered** `KernelMetaType` (`MIX_AIC_1_2 = 5` there), `KERNEL_TASK_TYPE_DEFAULT(val)
→ __builtin_cce_kernel_type_set(val)`, no two-argument form at all, and the whole block gated on
`__CCE_ENABLE_AUTO_INFER__` (predefined only under `-fcce-auto-infer-kernel-type`, which nothing in
CANN's shipped build passes). Both files share the `__PLUGIN__*_DEFINED__` guards, so whichever is
included first wins.

**The project-only restriction is mechanism, not documentation.** Neither header records it; the
enum's only annotation is the misleading `// define macro for deterministic compile options` at
`:88`. `ENABLE_FEATURE_FOR_COMPILE` expands to nothing unless `__CHECK_FEATURE_AT_PRECOMPILE` is
defined, and only the project driver defines it —
`asc_op_compile_base/asc_op_compiler/kernel_info_infer.py:779-785` runs a preprocess-only pass with
`["-E", "-D__CHECK_FEATURE_AT_PRECOMPILE", "-includestdio.h", "-DASCENDC_TPL_PRE"]` and then
recovers the value **by regex from the `.i` text** (`:99-111`,
`__enable_feature_for_compile_default\s*=\s*([0-9a-zA-Z_]{1,})\s*;`).

So, correcting what this repository has assumed: in a direct `bisheng -xcce` build **both** forms
expand to literally nothing. `tests/glm52/routed/routed_expert_kernel.cpp:330`'s
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` is a silent no-op today; the kernel's 1:2 ratio
comes entirely from `__mix__(1, 2)`. The two mechanisms are exactly complementary and neither works
in the other's world: `__mix__` + `__schedmode__` for `<<<>>>`, `KERNEL_TASK_TYPE*` for a project.

Two further constraints the header does not state, both consequences of the regex: the keyed form's
first argument must be a **numeric literal**, not a C expression; and `kernel_info_infer.py:218-238`
**rejects anything other than `KERNEL_TYPE_MIX_AIC_1_1` or `KERNEL_TYPE_MIX_AIC_1_2`** as its second
argument. `KERNEL_TASK_TYPE_DEFAULT` accepts the full enum.

And the choice between them changes the artifacts. `KERNEL_TASK_TYPE_DEFAULT` alone is sufficient
and is what `grouped_matmul_swiglu_quant.cpp:71` leads with; per-key types make
`ascendc_compile_gen_json.py:477` write `"taskRation": "tilingKey"` at top level with a per-key
`kernelList`, whereas default-only writes a literal `"taskRation": "1:2"`. Downstream,
`ascendc_ops_config.py:156-170` emits `"multiKernelType": 1` into `binary_info_config.json` only
when `coreType == 0 && taskRation == "tilingKey"`. Getting this wrong changes the config json.

**What this means for the `KERNEL_TASK_TYPE` question.** Moving into an operator project buys
**per-tiling-key switching among `MIX_AIC_1_1` and `MIX_AIC_1_2`, and nothing else**, at the price of
deleting `__mix__(1,2)` and `__schedmode__(1)` (§5.1). It does not buy the mix ratio, which
`__mix__` already supplies.

This contradicts `README.md` finding 12's premise. That finding reads Huawei's `swiglu_gated_mlp`
as having "twelve per-tiling-key overrides, nine `AIC_ONLY` and three `AIV_ONLY`", and prices a
tiling-key-switched binary at 4.5x on launch cost for an AIC_ONLY + AIV_ONLY pair against one
MIX_AIC. If `kernel_info_infer.py` rejects every non-MIX type in the keyed form on CANN 9.2.0, then
that design is not expressible here and the 4.5x lever needs **separate binaries**, which is the
third of finding 12's three designs rather than the second. The conflict is unresolved: the
`swiglu_gated_mlp` reading came from a public repository, not from this installation. Resolving it
means grepping that kernel's actual source for whether the nine `AIC_ONLY` sites use
`KERNEL_TASK_TYPE` or `KERNEL_TASK_TYPE_DEFAULT`.

---

## 6. `op_proto` — nothing to rebuild for a reused prototype

`op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so` defines 60 dynamic symbols, all of them the
C registry ABI and none per-operator:

```
GetOpImplFunctions   GetOpImplFunctionsV2   GetOpCtImplFunctions
GetRegisteredOpNum   GetRegisteredOpCtNum   GetOpSpecificInfo
DestroyCustomOpRegistry   GenSimplifiedKey   CheckSupport   CalcOpParam ...
```

`nm -D --defined-only ... | grep -i groupedmatmul` returns nothing: registration is data-driven
through static initializers, and the operators appear only as strings —
`GroupedMatmulSwigluQuant`, `GroupedMatmulSwigluQuantV2`,
`GroupedMatmulSwigluQuantWeightNzTensorList`, `InferShape4GroupedMatmulSwigluQuantV2`,
`InferDataType4GroupedMatmulSwigluQuantV2`.

`op_proto/inc/grouped_matmul_swiglu_quant_v2_proto.h`, verbatim:

```cpp
REG_OP(GroupedMatmulSwigluQuantV2)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(x_scale, ge::TensorType::ALL())
    .INPUT(group_list, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight_scale, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight_assist_matrix, ge::TensorType::ALL())
    .OPTIONAL_INPUT(bias, ge::TensorType::ALL())
    .OPTIONAL_INPUT(smooth_scale, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(y_scale, ge::TensorType::ALL())
    .ATTR(dequant_mode, Int, 0) ... .ATTR(swiglu_limit, Float, 0)
    .OP_END_FACTORY_REG(GroupedMatmulSwigluQuantV2);
```

`ge::TensorType::ALL()` throughout. Dtype and format admissibility live entirely in
`aic-ascend910_93-ops-info.json` and the `simplifiedKey` table; the prototype pins only arity and
attr names. So a new kernel that reuses an existing prototype — same op type, same tensor arity,
same attrs — needs **no `op_proto` rebuild**. A kernel with a different arity needs one.

---

## 7. `op_api` — the aclnn entry points ship with the vendor, and their names are decoupled from op types

`op_api/include/aclnnop/` holds 57 public headers. `op_api/lib/libcust_opapi.so` (1142864 B,
stripped) exports 459 dynamic symbols, 114 of them `aclnn*` — 57 two-phase
`aclnnX`/`aclnnXGetWorkspaceSize` pairs. The torch/aclnn-level entry points are part of the
package, not of the base CANN install.

For this family:

```
T aclnnGroupedMatmulSwigluQuant                     T aclnnGroupedMatmulSwigluQuantGetWorkspaceSize
T aclnnGroupedMatmulSwigluQuantV2                   T aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize
T aclnnGroupedMatmulSwigluQuantWeightNZ             T aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize
T aclnnGroupedMatmulSwigluQuantWeightNzV2           T aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize
T aclnnGroupedMatmulSwigluQuantWeightNzTensorList   (+ GetWorkspaceSize)
T _ZN4l0op26GroupedMatmulSwigluQuantV2EPK9aclTensorPK13aclTensorListS5_S2_S5_S2_S2_S2_llllblPK11aclIntArraydP13aclOpExecutor
W _ZN4l0op34GroupedMatmulSwigluQuantV2OpTypeIdEv
```

**Five aclnn names, three op-type strings.** aclnn façades and GE op types are decoupled: several
façades funnel into one `l0op::<OpType>` and therefore into one kernel-binary table. That is the
lever for adding a torch-visible entry point without touching `op_proto` or the op-info json — at
the cost of rebuilding this 1.1 MB library from `source/csrc`.

### The live dispatch chain, end to end

```
torch.ops…grouped_matmul_swiglu_quant_v2
  ← torch_binding.cpp:3070  ops.def("grouped_matmul_swiglu_quant_v2(Tensor x, Tensor[] weight, …)")
  ← torch_binding.cpp:3075  ops.impl(…, torch::kPrivateUse1, &vllm_ascend::grouped_matmul_swiglu_quant_v2)
  → gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h:53
       EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantWeightNzV2, x, weight, weight_scale, …)
  → libcust_opapi.so :: aclnnGroupedMatmulSwigluQuantWeightNzV2   (dlsym'd at run time)
  → l0op::GroupedMatmulSwigluQuantV2  ⇒ OpType "GroupedMatmulSwigluQuantV2"
  → libnnopbase.so: simplifiedKey lookup in kernel/config/ascend910_93/grouped_matmul_swiglu_quant_v2.json
  → binInfo.jsonFilePath → …/GroupedMatmulSwigluQuantV2_fa3d6d3de6e1f32e170e39d2ddd3a20e.{json,o}
```

The top of that chain is an ordinary `TORCH_LIBRARY` registration on `kPrivateUse1`. Which is the
same mechanism §9 uses, and is why §9 exists.

---

## 8. The full-package path, and `msopgen`

For an operator that does not exist yet, the package is built from a project, not patched.

`source/csrc/build.sh` (57936 B) drives it: `RELEASE_TARGETS=("ophost" "opapi" "opgraph" "onnxplugin")`,
invoked as

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3
```

`--vendor_name` becomes `-DVENDOR_NAME=` (line 1183) and the output vendor directory is named
`${vendor_name}_transformer` (lines 464-469) — which is where `custom_transformer` comes from.
Supporting machinery is in `source/csrc/scripts/`: `opgen/opgen_standalone.py`,
`package/package.py`, `util/insert_op_info.py`, `util/merge_proto.py`,
`util/modify_gen_aclnn.py`, `util/build_opp_kernel_static.py`.

### What `msopgen` actually emits

A premise correction first: `msopgen` 9.2.0 with `-lan cpp` does **not** instantiate
`tools/op_project_templates/ascendc/customize/`.
`/opt/ascend/cann-9.2.0/python/site-packages/msopgen/interface/const_manager.py:79` sets
`OP_TEMPLATE_ASCENDC_PATH_LINUX = "./new_op_project_template/custom_op"`, and that tree is thin:
`CMakeLists.txt`, `CMakePresets.json`, `build.sh`,
`framework/{onnx_plugin,tf_plugin}/CMakeLists.txt`, `op_host/CMakeLists.txt`,
`op_kernel/CMakeLists.txt`, and **no `cmake/` directory at all**. Its top-level `CMakeLists.txt` is
28 lines — `find_package(ASC REQUIRED)`, `npu_op_package(${package_name} TYPE RUN CONFIG
INSTALL_PATH ${CMAKE_BINARY_DIR}/)`, three conditional `add_subdirectory` — and all the machinery
lives in the installed package at `/opt/ascend/cann-9.2.0/tools/tikcpp/ascendc_kernel_cmake/`,
resolved by `asc-config.cmake` as `$ENV{ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake`
(`tools/tikcpp` and `compiler/tikcpp` are both symlinks to `aarch64-linux/tikcpp`).
`ASCEND_HOME_PATH` must be set or `asc-config.cmake` hard-fails.

`tools/op_project_templates/ascendc/customize/` is the full self-contained reference project — what
the documentation describes and what `-m 1` extension assumes. Its layout, with roles:

| file | lines | role |
| --- | --- | --- |
| `CMakeLists.txt` | 76 | includes `cmake/{config,func,intf}.cmake`, conditional `add_subdirectory(framework\|op_host\|op_kernel\|testcases)`, seds `vendor_name=customize` into `scripts/*`, runs `cmake/util/gen_version_info.sh`, CPack with `CPACK_GENERATOR External` + `cmake/makeself.cmake` → `custom_opp_<system>_<arch>.run` |
| `CMakePresets.json` | — | the only file a user edits. `ASCEND_COMPUTE_UNIT` ships as the literal placeholder `__ASCNED_COMPUTE_UNIT__` (typo included; msopgen substitutes it), plus `vendor_name=customize`, `ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/latest`, `ENABLE_BINARY_PACKAGE=True`, `ENABLE_SOURCE_PACKAGE=True`, `ENABLE_TEST=True`, `ENABLE_CROSS_COMPILE=False`, `ASCEND_PACK_SHARED_LIBRARY=False` |
| `cmake/config.cmake` | 50 | defaults; `ASCEND_AUTOGEN_PATH=${CMAKE_BINARY_DIR}/autogen`; touches `autogen/custom_compile_options.ini` and `autogen/custom_opc_options.ini` at configure time |
| `cmake/func.cmake` | 427 | all the logic: `opbuild()`, `add_ops_info_target()`, `add_ops_compile_options()`, `add_simple_kernel_compile()`, `add_kernels_compile()`, `add_kernels_install()` |
| `cmake/intf.cmake` | 28 | the `intf_pub` INTERFACE target: `-fPIC -fvisibility=hidden -std=c++11`, **`_GLIBCXX_USE_CXX11_ABI=0`**, `-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack` |
| `cmake/makeself.cmake`, `cmake/device_task.cmake` | 31, 50 | packaging; tiling-sink `ExternalProject` |
| `cmake/util/` | 38 files | the generators. Load-bearing: `ascendc_compile_kernel.py`, `ascendc_bin_param_build.py`, `ascendc_impl_build.py`, `insert_simplified_keys.py`, `ascendc_ops_config.py`, `parse_ini_to_json.py`, `opdesc_parser.py`, `ascendc_get_op_name.py`, `ascendc_gen_options.py`, `const_var.py`, `makeself/` |
| `op_host/CMakeLists.txt` | 221 | `aux_source_directory` + `opbuild()`, then three libraries |
| `op_kernel/CMakeLists.txt` | 5 | `add_kernels_compile()`, plus an optional `add_ops_compile_options(ALL OPTIONS -g -O0)` under Debug. Nothing else |
| `framework/CMakeLists.txt` | — | TF/ONNX/Caffe parser plugins, optional |
| `scripts/` | — | `help.info`, `install.sh` (the `config.ini` writer quoted in §1), `upgrade.sh` |

Note `cmake/intf.cmake`'s `_GLIBCXX_USE_CXX11_ABI=0` against this host's `torch` built with ABI 1
(§9): an operator project and a torch extension cannot share one compiled object.

Which file holds what, which is the part that decides the cost:

- **Kernel** → `op_kernel/<op_file>.cpp`, and the filename is not free. `cmake/func.cmake:402` is
  `SRC ${CMAKE_SOURCE_DIR}/op_kernel/${op_file}.cpp`, where `op_file` comes from
  `ascendc_get_op_name.py --ini-file=autogen/aic-<soc>-ops-info.ini` reading `opFile.value`, which
  `opbuild` derives by snake-casing the `OpDef` type name. So it must be
  `op_kernel/<snake(OpType)>.cpp` or the kernel target is never created. Contents:
  `extern "C" __global__ __aicore__ void <op_file>(…, GM_ADDR workspace, GM_ADDR tiling)`, with
  `KERNEL_TASK_TYPE_DEFAULT(...)` as the first body statement — and without `__mix__` or
  `__schedmode__` (§5.1).
- **Host tiling and the op definition** → the *same single file*, `op_host/<op_file>.cpp`.
  `op_host/CMakeLists.txt:2` is `aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)`, and
  that identical source list is compiled three times with different macros: plain into
  `autogen/libascend_all_ops.so` for `opbuild`; with `-DOP_PROTO_LIB` into
  `libcust_opsproto_rt2.0.so`; with `-DOP_TILING_LIB` into `libcust_opmaster_rt2.0.so`, installed as
  `op_impl/ai_core/tbe/op_tiling/liboptiling.so`. `OP_ADD` in
  `/opt/ascend/cann-9.2.0/aarch64-linux/include/register/op_def_registry.h:19/30/54` switches on
  exactly those two macros — the proto branch registers InferShape / InferShapeRange /
  InferDataType, the tiling branch registers `impl.Tiling(op.AICore().GetTiling())` plus
  `TilingParse` and the OpCheckFunc helpers. So one translation unit carries
  `class MyOp : public OpDef`, `OP_ADD(MyOp)`, `IMPL_OP_OPTILING(MyOp)`, `IMPL_OP_INFERSHAPE(MyOp)`,
  and the tiling struct via `BEGIN_TILING_DATA_DEF`.
- **aclnn wrapper** → generated, never written: `autogen/aclnn_*.cpp/.h` from `opbuild`, built into
  `libcust_opapi.so`, installed to `op_api/lib` and `op_api/include`.

`build.sh` runs, in order: `python3 cmake/util/preset_parse.py CMakePresets.json` (turning the
preset into `-D` flags for cmake < 3.19); `cmake -S . -B build_out --preset=default`;
`cmake --build build_out --target binary -j$(nproc)`; `cmake --build build_out --target package
-j$(nproc)` — or `--target install` when `ASCEND_PACK_SHARED_LIBRARY=True`. With
`ENABLE_CROSS_COMPILE` and `ENABLE_BINARY_PACKAGE` both set it does a two-phase dance: configure
with `-DENABLE_CROSS_COMPILE=0`, build `--target cust_optiling`, stash
`libcust_opmaster_rt2.0.so` into `build_out/host_native_tiling/`, wipe, reconfigure with
`-DHOST_NATIVE_TILING_LIB=<that path>`, then `binary` and `package`. The reason is §4.1's: the
binary kernel build dlopens the tiling library on the build host, so it must be a native object
even for a cross build.

The thin template's `build.sh` instead calls
`$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py`, then
`cmake -S . -B build_out --preset=default`, then
`cmake --build build_out --target binary package -j$(nproc)`.

What a project buys over a patch, and what it costs, is §10's subject.

---

## 8b. Where the two operators are actually called from, and what reaches them

Paths below are in the live checkout `/workspace/src/vllm-ascend-unified`.

The routed experts do **not** go through `torch_npu` for the first operator. Both calls sit in one
class, `AscendW4A8DynamicFusedMoEMethod` in `vllm_ascend/quantization/methods/w4a8/w4a8.py`,
decorated `@register_scheme("W4A8_DYNAMIC", "moe")` at `:118`:

```python
# :599-607, inside apply_gmm1_act_quant (:521)
        hidden_states, swiglu_out_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=hidden_states, weight=w1, weight_scale=w1_scale, x_scale=pertoken_scale,
            group_list=group_list, weight_assist_matrix=bias1,
            dequant_mode=0, group_list_type=group_list_type,
            swiglu_limit=mlp_compute_input.swiglu_limit,
        )

# :639-654, apply_gmm2
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states], weight=w2, scale=w2_scale, bias=bias2,
            per_token_scale=[act_out_scale], split_item=2,
            group_list_type=group_list_type, group_type=0, group_list=group_list,
            output_dtype=torch.bfloat16,
        )[0]
```

The caller is `vllm_ascend/ops/fused_moe/moe_mlp.py:70-71`:

```python
    if quant_method.supports_fused_activation(mlp_compute_input.activation):
        hidden_states, act_out_scale = quant_method.apply_gmm1_act_quant(mlp_compute_input)
    ...
    hidden_states = quant_method.apply_gmm2(mlp_compute_input, hidden_states, act_out_scale)
```

reached via `quantization/method_adapters.py:319` → `w4a8.py:280` →
`ops/fused_moe/moe_comm_method.py:175` → `moe_mlp.py:49`. `supports_fused_activation` is
`quantization/methods/base.py:353-356` testing `fused_activations = frozenset({"silu", "situ"})`
(`w4a8.py:192`), and GLM-5.2's `hidden_act` is `silu`.

Both are **attribute lookups performed per call**, so a class- or instance-level override of
`apply_gmm1_act_quant` / `apply_gmm2` is honoured. The structural analogue already in this repo is
`tilert/models/glm_5_2_ascend/cann_head.py:14-28`, which copies the `quant_method` object,
`MethodType`-assigns onto the copy and swaps the copy in, leaving the shared class untouched.

The `_C_ascend` op's own stack, for completeness: schema at `csrc/torch_binding.cpp:3069-3075`
inside `TORCH_LIBRARY_EXPAND(CONCAT(_C, _ascend), ops)` (`:2799`), Meta kernel at
`csrc/torch_binding_meta.cpp:186` registered at `:1846`, body at
`csrc/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h:20-93` which
allocates `at::empty({m, n/2}, kChar)` and `at::empty({m}, kFloat)` and issues
`EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantWeightNzV2, ...)` — the **WeightNz** façade, not the ND
one; `OP_TYPE_REGISTER(GroupedMatmulSwigluQuantV2)` at
`csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/grouped_matmul_swiglu_quant_v2.cpp:22`
collapses both onto one op type.

### The MoE is already an opaque custom op, which is why a Python override is captured

The compiled FX graph does not contain either grouped matmul. The dump
`/workspace/experiments/lever-20261020/cache-tail8/torch_compile_cache/014f8d7d08/rank_0_0/backbone/computation_graph.py:588`
reads

```python
moe_forward_shared = torch.ops.vllm.moe_forward_shared(view_4, view_4, view_4, None, synthetic_local_tmp_0_, 0)
```

registered by `direct_register_custom_op` at
`/workspace/src/vllm-unified/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:209-214`,
with zero occurrences of `grouped_matmul_swiglu_quant_v2` among the graph's 78 `mla_forward` /
75 `moe_forward_shared` / 156 `npu_add_rms_norm_bias` nodes. Because the boundary is opaque, Python
re-enters `apply_gmm1_act_quant` on every eager invocation of that node — and once, with any patch
in force, during capture.

Capture is a genuine ACL graph, `vllm_ascend/compilation/acl_graph.py:208-210` and `:297-301`.
The arm's own record `/workspace/experiments/lever-20261020/tail8/rank-0.json` reads `mode "3"`
(VLLM_COMPILE), `cudagraph_mode FULL_DECODE_ONLY`, `capture_sizes [32]`, `enable_static_kernel
true`, `enable_super_kernel true`, `enable_npugraph_ex true`, `num_graphs_seen 1`,
`num_inductor_compiles 0`, `graph: {width 32, captures 1, replays 1164}`. Not torchair: there is no
`torchair` in site-packages and the `except ImportError` fallback at
`vllm_ascend/compilation/compiler_interface.py:170-225` is dead because
`import npugraph_ex` resolves to `torch_npu/dynamo/npugraph_ex`.

The profile confirms the operators are inside the captured region: of 247
`GroupedMatmulSwigluQuantV2` launches, 225 carry Model ID 48 (the captured verify graph, 7467
kernels), 18 Model ID 49 (draft), 3 Model ID 47, and 1 `4294967295` (eager). Same split for
`GroupedMatmul`. **So the patch window is after `worker.load_model()` and before capture, and a
patch installed after capture is ignored outright for the captured width** — replay never enters
Python.

One more trap vLLM supplies by default: `/workspace/src/vllm-unified/vllm/config/vllm.py:316` sets
O2 → `CUDAGraphMode.FULL_AND_PIECEWISE`, which contains PIECEWISE, which drives
`vllm_ascend/platform.py:1337` and silently zeroes npugraph_ex, static kernel and super kernel.
Every arm here overrides it to FULL_DECODE_ONLY, taking the `:1361-1363` branch, which empties
`splitting_ops`: one graph over the whole model forward.

### The measured call shape

From `profiles/width8-20261019/rank-0/kernel_details.csv`, the kernel named
`aclnnGroupedMatmulSwigluQuantWeightNzV2_GroupedMatmulSwigluQuantV2_GroupedMatmulSwigluQuantV2`
(247 launches, 22419.4 µs) and `aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul`
(247 launches, 18515.3 µs) — 6.81 and 5.63 ms a round at 247/75 = 3.29 forwards.

| parameter | shape | dtype | format |
| --- | --- | --- | --- |
| `x` | `8,6144` | INT8 | ND |
| `x_scale` | `8` | FLOAT | ND |
| `group_list` | `256` | INT64 | ND |
| `weight` | `256,4,384,16,64` | DT_INT4 | FRACTAL_NZ |
| `weight_scale` | `256,256` | UINT64 | ND |
| `weight_assist_matrix` | `256,256` | FLOAT | ND |
| `y` | `8,128` | INT8 | ND |
| `y_scale` | `8` | FLOAT | ND |

`MIX_AIC`, 24 blocks. **M = 8**, i.e. eight permuted rows, with a 256-slot group-list table and all
256 experts declared — pure TP16, no expert parallelism (`n = 2 × 2048/16 = 256`, `k = 6144`;
`use_cann_megamoe` at `ascend_forward_context.py:87-97` requires `enable_expert_parallel`, which
is off, so `select_moe_comm_method` picks `MoECommType.ALLGATHER`). That matches
`tests/glm52/replay_cann_gmm1.py:44`'s frozen capture exactly: 8 active slots, 8 rows, every count
0 or 1.

### The weight pipeline is one-way, and the one call site that decides it

`w4a8.py:225-241` allocates `w13_weight [E, intermediate_per_partition, hidden]` INT8 with two int4
packed per byte along the output dimension, and `w2_weight [E, hidden//2, intermediate]`. Then
`:373-382`:

```python
    def process_weights_after_loading(self, layer):
        if self.quant_method == COMPRESSED_TENSORS_METHOD:
            self.process_weights_after_loading_compressed_tensors(layer)
        else:
            self.process_weights_after_loading_modelslim(layer)
        layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
        layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
```

ending at `:410-411` with `_pack_to_int32` (`:348-353`, a `.view(torch.int32)`).
`process_weights_after_loading_modelslim` (`:656-665`) transposes both weights `(1,2)`, runs
`_process_scale` on both scales, and reduces both `scale_bias` by
`transpose(1,2).contiguous().sum(axis=1)` — which is where the FLOAT `weight_assist_matrix [256,256]`
comes from. `_process_scale` (`:355-360`) transposes, reinterprets the float32 bit pattern as
uint32 through numpy and widens to int64 — which is where the UINT64 `weight_scale` comes from.
The cast itself is `vllm_ascend/utils.py:394-406`,
`torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)` with `ACL_FORMAT_FRACTAL_NZ = 29` at
`:63`, gated by `_should_trans_nz` (`:350-387`) on `config.weight_nz_mode`, default 1
(`ascend_config.py:618`); A3 is `WeightLayoutPolicy.CONFIGURABLE`
(`device/hardware_profile.py:318`), and quantized dtypes convert at mode 1.

**Once at load, and not reversible.** `apply_gmm1_act_quant` performs no format cast; the only
per-step host work is `_maybe_convert_group_list` (`:471-479`),
`torch.cat([group_list[:1], torch.diff(group_list, dim=0)])` when `group_list_type == 0`. And the
layout is enforced, not advisory:
`csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp:112-144`
computes `storageShape = {expertNum, n / 64, k / 16, 16, 8}`, calls `SetStorageShape` and
`SetStorageFormat(FORMAT_FRACTAL_NZ)` unconditionally — 「weight在该场景下强制绑定StorageFormat
和 ViewFormat 为NZ」 — and asserts the *incoming* storage shape already had `dimNum == 5`. So the
fractal tile is k0=16 by n0=64 int4, and without the load-time cast the operator rejects the tensor.

`patches/cann-gmm-swiglu-v2/w4a8-compact.patch` records what cannot be undone later:
`index_select` on an NZ weight fails inside TransData, a per-expert `copy_` into an NZ destination
fails the same way, and `npu_format_cast` back to ND fails. The global escape is
`additional_config` `weight_nz_mode: 0`, which turns the cast off for **every** quantized weight in
the model, linear layers included. Per-layer or per-tensor means patching
`process_weights_after_loading`, which is exactly where that patch already edits.

The installed tree is stock: `grep -c _compact_expert_declaration` on the installed `w4a8.py`
returns 0, and none of `patches/vllm-ascend/*.patch` is applied on the host.

### Two traps that will silently invalidate a measurement

**Static-kernel cache.** With `enable_static_kernel` on, npugraph_ex recompiles the captured
operators into a static-kernel package, and `_get_static_kernel_cache_key` keys on dumped operator
params, the super-kernel option string and the determinism flag — **not** on
`ASCEND_CUSTOM_OPP_PATH` and not on the vendor headers — with its index below the process cwd.
That is why `tests/glm52/launch_arm.sh:38-41` gives each arm its own `VLLM_CACHE_ROOT` and why
`:55` does `cd "$out"` with the comment "Static kernel packages are cached below cwd, outside
VLLM_CACHE_ROOT." The live mitigation is a `sys.meta_path` hook,
`/workspace/experiments/w4a8-race-20261019/fixhook/sitecustomize.py`, selected by
`FIX_ARM=cache-off`, which wraps `_configure_backend` post-import and injects
`options["disable_static_kernel_compile_cache"] = True`. The same file offers
`_set_static_kernel_blacklist(["GroupedMatmulSwigluQuantV2", ...])`, a documented torch_npu API at
`/opt/trt-asc/venv/lib/python3.11/site-packages/torch_npu/dynamo/npugraph_ex/_acl_concrete_graph/static_kernel.py:1168`
that excludes operators from static-kernel compilation by JSON-filename substring.

**A Python overlay of `vllm_ascend` re-arms the vendor path.** `utils.py:454-458` derives the
vendor path from `dirname(vllm_ascend.__file__)` and prepends it, so an overlay's *unpatched*
vendor copy would outrank a patched one. It is harmless today only because `_prepend_env_path`
guards with `if path not in path_entries` (`:449`) and `activate.sh` already lists that exact path
second.

### Where TPOT comes from

Not `vllm serve`: neither that string nor `vllm bench serve` occurs anywhere under
`/workspace/experiments` or the TileRT tests. TPOT is `tests/glm52/bench_cann_mtp.py`, scope
「internal TP16 session delivery; HTTP/HIXL excluded」, clock `perf_counter_ns`, formula
`(last_callback-first_callback)/(output_tokens-1)` (`:618-624`), aggregated by rank 0 into
`summary.json` (`:561-592`) with cross-rank digest agreement enforced at `:545-560` and
`graph.captures == 1` cross-checked at `:264-281`.

The arm that both uses the gmm-fastpath overlay and produced a number:

```bash
TAG=tail8 WIDTH=32 UTILISATION=0.78 PORT_BASE=21000 \
TREE=/workspace/experiments/shared-fia-20261020/tree \
OUT=/workspace/experiments/lever-20261020 \
OVERLAY=/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer \
bash /workspace/experiments/shared-fia-20261020/tree/tests/glm52/launch_arm.sh --lookup
```

`lever-20261020/tail8/summary.json` median 10.9224 ms, `tail32/summary.json` 11.1642 ms, both with
`target_graph true, compile_target true, super_kernel true, attention_kernel sfa, fp32_head true,
drafter lookup`. The resolved `ASCEND_CUSTOM_OPP_PATH` in that arm has **six** entries: three
`fla_npu` paths (prepended before user code by
`/opt/trt-asc/venv/lib/python3.11/site-packages/fla_npu/__init__.py:94-95` via a `.pth`), then the
gmm-fastpath overlay, then the owner kernel-overlay, then the in-tree vendor. `activate.sh:7`
`unset`s the variable first, so anything exported before sourcing is discarded.

## 9. The other channel: a torch custom op, with no OPP packaging at all

The vendor package is not the only way a hand-written kernel reaches the captured decode graph, and
on this stack it is not the one with the most precedent.

vLLM-Ascend itself registers kernels that are in no vendor package. `direct_register_custom_op` and
`register_fake` appear in `vllm_ascend/attention/dsa_v41.py`, `vllm_ascend/ops/triton/dcp/dcp_a2a.py`
and five `vllm_ascend/ops/triton/linearnorm/*.py`. The registration, from
`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope_vnorm.py:473-478`:

```python
from vllm.utils.torch_utils import direct_register_custom_op   # line 32

direct_register_custom_op(
    op_name="qkv_rmsnorm_rope_vnorm",
    op_func=split_qkv_rmsnorm_rope_vnorm_impl,
    fake_impl=split_qkv_rmsnorm_rope_vnorm_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
```

`dispatch_key="PrivateUse1"` is the NPU key; the `fake_impl` is what keeps Dynamo from breaking the
graph under `CompilationMode.VLLM_COMPILE`.

This project has already walked that path once with its own AscendC kernel. `docs/evidence/ascend-decode/torch-allreduce-20261018/README.md`
records the one-sided all-reduce reaching the production graph as `aclshmem_torch.so` plus
`libtp_push_kernel.so`, loaded by `tilert/models/glm_5_2_ascend/onesided_reduce.py:33-34`:

```python
    ctypes.CDLL(f"{LIBRARY_DIR}/libtp_push_kernel.so", mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(f"{LIBRARY_DIR}/aclshmem_torch.so")
```

with three build facts from that evidence file worth keeping: the binding links
`${ACLSHMEM_ALL_KERNEL_TARGETS}` but does **not** record the kernel library as a runtime
dependency, so the kernel `.so` must be `dlopen`ed `RTLD_GLOBAL` first or the binding fails with
`undefined symbol`; `attr_init` reads the **calling thread's** ACL device, which inside a vLLM
worker may be unset; and the registered class namespace comes from the `torch::jit::class_` call,
not from the directory name.

The attachment point is already in the engine. `tilert/models/glm_5_2_ascend/cann_model.py:257-260`
installs a patch after `load_model()` and before any graph capture:

```python
        if plain_moe_combine:
            from .moe_combine import install

            self._restore_combine = install()
```

alongside `_install_tp_sum` (`:256`, body at `:262`), `install_fp32_head` (`:248-250`) and
`onesided_reduce.install` (`:273`). Four overrides already live at that hook.

Toolchain facts for building such a `.so` on this host: `torch 2.12.0+cpu`, `torch_npu 2.12.0`,
`torch._C._GLIBCXX_USE_CXX11_ABI is True` (so `-D_GLIBCXX_USE_CXX11_ABI=1`), site-packages at
`/opt/trt-asc/venv/lib/python3.11/site-packages/torch`. `tests/glm52/routed/build.sh` already
produces the kernel object with
`bisheng -xcce --npu-arch=dav-2201 -std=c++17 -O2 -c …`; it needs `-fPIC` added and a `-shared`
link rather than an executable one.

### Four registration routes, each with a template on this box

| route | template | cost |
| --- | --- | --- |
| pure Python `torch.library` | `vllm_ascend/ops/packaged_attention.py:40,83,130` — `define`, `register_fake`, `impl(..., "PrivateUse1")`, with the DSL imported lazily so tracing never pulls the compiler in. Modern one-decorator form at `vllm_ascend/lora/lora_ops.py:460,504` | no compiler; needs an existing callable |
| C++ `TORCH_LIBRARY` extension | `csrc/third_party/catlass/examples/python_extension/src/bindings/torch_bindings.cpp` (31 lines) with `CMakeLists.txt:35,41,43` and loader `torch_catlass/__init__.py`; build helper `NpuExtension` at `torch_npu/utils/cpp_extension.py:12` | one `.so` |
| stable-ABI `.so` + `torch.ops.load_library` | in production as `/opt/trt-asc/venv/lib/python3.11/site-packages/fla_npu/libfla_npu_stable.so`, consumed at `fla_npu/ops/ascendc/_stable.py:386,476`, stream passed as a trailing `int` | one `.so`, no torch headers |
| override `torch_npu`'s own op | `torch.library.Library("npu", "IMPL").impl("npu_grouped_matmul_swiglu_quant_v2", fn, "NPU", allow_override=True)` — `allow_override=True` is mandatory, torch_npu already holds that key. Precedent against `aten` at `vllm_ascend/batch_invariant.py:126-163` | keeps existing call sites |

A second vendor-package precedent, outside vLLM-Ascend entirely: the `fla_npu` wheel ships
`/opt/trt-asc/venv/lib/python3.11/site-packages/fla_npu/opp/vendors/fla_npu_transformer/` and
injects it before CANN initialises via `fla_npu_opp_env.py` plus a `.pth`.

### The stream accessor is a landmine

`torch_npu._C._npu_getCurrentRawStream(idx) -> int` is what torch_npu's own inductor codegen emits
(`torch_npu/utils/_inductor.py:10`), and it **drains the task queue first**.
`fla_npu/ops/ascendc/_stable.py:248-272` documents the cost: roughly **1 ms per call on a busy
vLLM worker against 2 µs for the plain read**, and it records that a process-global cached stream
pointer previously sent kernels onto another thread's stream. Use
`_npu_getCurrentRawStreamNoWait`, and do not cache the pointer across threads.
`torch_npu.npu.current_stream().npu_stream` and `Stream._as_parameter_` are at
`torch_npu/npu/streams.py:97-98`.

### One ABI hazard, found in passing

The in-house vendor's `aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize` is **not** ABI-identical to
CANN's. `csrc/gmm/grouped_matmul_swiglu_quant_v2/op_host/op_api/aclnn_grouped_matmul_swiglu_quant_v2.cpp:66-95`
carries an extra `double swigluLimit` before the outputs (generated header
`op_api/include/aclnnop/aclnn_grouped_matmul_swiglu_quant_v2.h:52`), which torch_npu's schema has no
field for. **Inference from AAPCS64**, not measured: a `double` travels in `d0` and consumes no
integer or stack slot, so every pointer still lines up and the call will not fault — it will read
whatever is in `d0` as the SwiGLU clamp. The `_C_ascend` path is self-consistent because it
declares and passes `float swiglu_limit=0.0` explicitly. Relevant because `GetOpApiFuncAddr`
(`torch_npu/include/third_party/op-plugin/op_plugin/utils/op_api_common.h:51-58`) walks the custom
vendor paths building `<dir>/op_api/lib/libcust_opapi.so`, `dlopen` + `dlsym`, first hit wins, and
only then falls back to the built-in handles — so a custom vendor exporting that symbol shadows
CANN's for `torch_npu`'s operator too, with no torch-level registration anywhere.

---

## 10. Superseded assumptions

Each of these is something this project has held and the material above contradicts.

1. **"A direct `bisheng` build with `<<<>>>` has no path into the running inference graph."**
   Superseded. It has no path in *as a CANN operator* — correct — but it reaches the captured
   decode graph as a torch custom op, and this project has already done exactly that with its own
   AscendC kernel (§9, `torch-allreduce-20261018`). The missing path was never the only path; it
   was the one the project had not reused.

2. **"Building a custom operator so the graph dispatches to my kernel needs `msopgen`."**
   Wrong for an operator that already exists. `opc` alone, against sources that already ship inside
   the package. `msopgen` and `build.sh --pkg` are for a genuinely new operator (§8).

3. **"`KERNEL_TASK_TYPE_DEFAULT` is doing something in our `<<<>>>` build."** It is not.
   `ENABLE_FEATURE_FOR_COMPILE` expands to nothing unless `__CHECK_FEATURE_AT_PRECOMPILE` is
   defined, and only the project driver defines it — in a preprocess-only pass whose `.i` output it
   scrapes by regex (`kernel_info_infer.py:779-785`, `:99-111`). So
   `routed_expert_kernel.cpp:330` is a silent no-op and the kernel's 1:2 ratio comes entirely from
   `__mix__(1, 2)`. The corollary for `README.md` finding 12's "With `<<<>>>` only
   `KERNEL_TASK_TYPE_DEFAULT` applies": under `<<<>>>` *neither* form applies.

4. **"`simplifiedKey` and `staticKey` must be reproduced by hand."** Wrong. `opc` generates both
   from `input-param.json` under `--simplified_key_mode` and `--impl_mode`
   (§3.2, §3.3), and the project's build script *asserts* that it reproduced them rather than
   writing them. Four attempts to recompute `staticKey` by hand in one session all failed; nothing
   needs it.

5. **"`op_proto` must be rebuilt."** Not for a reused prototype. The prototype is
   `ge::TensorType::ALL()` on every parameter and pins only arity and attr names (§6).

6. **"The existing overlay is a precedent someone else set, which I have never used."** The overlay
   at `/workspace/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer`
   is the **output of this repository's own `tests/glm52/build_gmm1_barrier.sh`**, and its
   `inputs.sha256` names the script. The operator it overrides is `GroupedMatmulSwigluQuantV2`, not
   the V1 `GroupedMatmulSwigluQuant` whose files sit beside it untouched.

7. **"Moving to an operator project buys the `KERNEL_TASK_TYPE` form, and so buys more than
   dispatch."** Inverted. It buys per-tiling-key switching among exactly two types
   (`MIX_AIC_1_1`, `MIX_AIC_1_2` — `kernel_info_infer.py:218-238` rejects the rest) and it *costs*
   the deletion of `__mix__(1, 2)` and `__schedmode__(1)`, because the op-project build textually
   includes the source with `#define __global__ inline` and generates its own
   `auto_gen_<func>_kernel` as the real entry, so a `core_ratio` attribute on a now-inlined function
   is a hard bisheng error and a `cce_schedule_mode` on it has nothing to propagate to (§5.1).
   `KERNEL_TASK_TYPE_DEFAULT` already works under `<<<>>>` and `routed_expert_kernel.cpp:330`
   already uses it to get `MIX_AIC_1_2`; the vendor declares its 1:2 ratio nowhere on the host side
   either (§5). The ratio comes from hardware through `GetTaskRationImpl()`.

8. **"Sixteen experts on sixteen blocks is the per-rank shape at TP16."** Not what the capture
   shows. `profiles/width8-20261019/rank-0/kernel_details.csv` has the operator receiving
   `x` of `[8, 6144]` and emitting `y` of `[8, 128]` — **M = 8**, eight permuted rows, with all 256
   experts declared in a 256-slot `group_list`. `tests/glm52/replay_cann_gmm1.py:44` asserts the
   same of its frozen capture: 8 active slots, 8 rows, every count 0 or 1. The kernel's 41.990 µs
   was measured at 16 experts × 2 rows = M = 32, i.e. **four times the rows and twice the experts of
   the shape the vendor's 165.2 µs came from**. Which makes the advantage larger than claimed, not
   smaller: the kernel's own sweep gives 40.000 µs at eight experts, and `ROWS_PER_EXPERT=1` is
   below that on the 47.25 µs + 1.02 µs/row fit. But it is a comparison across two shapes until one
   `group_list` is read at the arm's actual width, and `docs/evidence/ascend-decode/int4-weights-20261020/README.md`'s
   "sixteen experts of 2,359,296 elements: 18.9 MB at INT4" rests on the same unchecked assumption.

9. **"A binary override of `GroupedMatmulSwigluQuantV2` can carry this kernel."** It cannot. The
   routed kernel replaces `GroupedMatmulSwigluQuantV2` **and** `GroupedMatmul` and keeps the
   intermediate in UB; an `.o` swap preserves the operator's prototype by construction, so the
   graph still issues the second operator and the GM round trip between them survives. A per-op
   override can only deliver a win that is internal to one operator — which is what the four landed
   `patches/cann-gmm-swiglu-v2/` patches did, measured at −1.468 ms a round. Fusing two operators
   requires changing the caller.

---

## Gaps

- `staticKey` was not reproduced byte-exactly from its generator. The canonical JSON's
  `buildOptions` content and the optional-parameter placeholder representation are the two unknowns.
  Nothing depends on closing this, since `opc` regenerates it.
- Whether `sha256` in the per-kernel descriptor is verified at load is inference from a string
  inventory of `libnnopbase.so` / `libfe.so`, not from disassembly.
- Whether `ASCEND_CUSTOM_OPP_PATH` outranks `${ASCEND_OPP_PATH}/vendors/config.ini` was not
  observable here: no base `config.ini` exists on this host.
- The reverse-order walk at `op_kernel_lib.cpp:372` against the forward walk at
  `fusion_binary_info.cc:69` is unexplained. If the reversal is incidental rather than compensating,
  op-info registration and kernel-binary registration could resolve a two-vendor conflict in favour
  of different vendors in one process. Separating them needs a controlled two-vendor run with
  OP-module logging at level 0.
- `op_impl/cpu/aicpu_kernel/` was inventoried (`libtransformer_aicpu_kernels.so`, 5408584 B;
  `cust_aicpu_kernel.json`, 3 ops, engine `DNN_VM_AICPU`) but not dissected. No AI-core operator
  override touches it.
- The `msopgen` template tree under `/opt/ascend/cann-9.2.0/tools/op_project_templates/ascendc/customize/`
  was read only for `scripts/install.sh`. The generated project's `CMakeLists.txt`, `op_host/` and
  `op_kernel/` skeletons are not quoted here. For an operator that already exists this does not
  matter — §4 is the whole path — but it is the gap to close before adding a genuinely new op type.
- The routed kernel's numerics have never been compared against the vendor operator, only against
  its own host reference (`routed_expert_main.cpp`, 0 of 196608). The vendor dequantises through the
  A3 Fixpipe's 1+8+10-bit scale with separate FP32 additions, emulated in this repo at
  `tilert/models/glm_5_2_ascend/reference_weights.py` `w4_msd_product` ("A3 Fixpipe uses a 1+8+10-bit
  scale, truncating the low 13 FP32 bits"), while the kernel does a plain FP32 `Muls` and a
  `CAST_RINT`. The integer products agree by construction — a losslessly widened int4 weight in an
  int8 container gives the same `mad`, and the vendor's MSD split plus `weight_assist_matrix`
  correction is exact — so the divergence is confined to the dequantisation rounding, and its size
  is unmeasured. Because this engine's TPOT is acceptance-coupled, that size is what decides whether
  any end-to-end number is interpretable.
- Resident HBM: widening the routed experts to INT8 costs an extra 302 MB a layer
  (256 experts × 2,359,296 elements × 0.5 B), i.e. +22.65 GB over 75 layers against a 65536 MB part
  already reporting ~50 GB in use at `gpu_memory_utilization 0.78`. A full-model INT8 arm does not
  fit; a partial-layer arm does. The INT8 kernel is therefore a per-layer experiment by
  construction, and the production path needs
  `docs/evidence/ascend-decode/int4-weights-20261020/README.md`'s INT4 form, whose remaining defect
  is 49105 of 49152 values deviating after the first projection became exact.
