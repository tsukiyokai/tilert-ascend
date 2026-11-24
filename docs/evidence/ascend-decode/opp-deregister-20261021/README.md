# Deregistering the custom grouped matmul: four attempts, the fourth works

2026-10-21, a3-84, one die, 96 rows, sixteen experts. The same call under four
treatments of a copy of `_cann_ops_custom/vendors/custom_transformer`:

| treatment | grouped matmul | norm+quant | gating |
| --- | --- | --- | --- |
| the package as shipped | 104.9 to 124.0 us | 43.038 | 26.455 |
| TBE kernel dir and config removed | 116.376 | 50.875 | 30.032 |
| its `binary_info_config.json` entry removed | 115.823 | 46.579 | 36.479 |
| its symbols localised with objcopy | 122.097 | 45.707 | 25.679 |
| **`libcust_opapi.so` withheld** | **70.147** | 48.757 | 28.016 |
| no custom path at all (CANN) | 66.813 | 56.049 | 33.732 |

## Why the first three could not work

Dispatch resolves `aclnnGroupedMatmulSwigluQuantWeightNzV2` as a dynamic
symbol, and the package's `op_api/lib/libcust_opapi.so` exports it -- two
symbols, the entry and its `GetWorkspaceSize`. Removing TBE kernels, removing
the manifest entry and localising symbols with `objcopy` all leave that
export in place: `objcopy --localize-symbol` edits the static symbol table,
not `.dynsym`, which needs a relink and a version script the package's source
would be required for.

The third failure is what located the dispatch layer, and the fourth attempt
follows from it.

## What withholding the library costs and buys

The grouped matmul falls to 70.147 us, near CANN's 66.813. The norm and the
gating stay close to their custom figures -- 48.757 and 28.016 against 43.038
and 26.455 -- because their TBE kernels and protos remain and only the aclnn
entry falls back.

    grouped matmul   34.8 us a layer saved      2.71 ms a round
    norm and gating   7.3 us a layer lost       0.57 ms a round
    net                                         2.14 ms a round

Against 1.4 ms for unsetting the path wholesale. Better by half, and it is a
file move in a copied package rather than a change to anything shared.
