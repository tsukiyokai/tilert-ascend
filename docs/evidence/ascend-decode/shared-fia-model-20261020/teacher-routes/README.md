# Teacher route prior from frozen M1 residual

A3 a3-194/trt-asc, CANN9.2.0 B114; norm on die15 and routes on die0. Torch peaks were16.1MiB and23.9MiB respectively, with a256MiB quota.

The current M1 kernels reconstructed rank-0 teacher routes for MoE layers 3–77 and positions 8192–8223. The successful device attempt is `routes-retry01-exit.json` (`returncode: 0`). This is not the historical run's original routes, a live M32 forward, or the rejected suffixes of a lookup request. TP16 route agreement was not measured.

| Window width | Layer windows | Expert union min / mean / median / max | Largest expert count min / mean / median / max | All expert counts ≤16 |
| --- | ---: | --- | --- | ---: |
| 8 | 300 | 33 / 44.35 / 44 / 60 | 2 / 4.71 / 5 / 8 | 300/300 |
| 16 | 150 | 51 / 68.067 / 67 / 100 | 3 / 8.613 / 9 / 16 | 150/150 |
| 32 | 75 | 74 / 99.36 / 96 / 135 | 5 / 16.16 / 16 / 30 | 40/75 |

The M32 count predicate rejects 35/75 layer windows. Widths 8 and 16 satisfy it by construction because each teacher row selects eight distinct experts. These are inputs to the current GMM2 adaptive count predicate, not observations of GMM2 dispatch or timing.

## Recovery and binding

`collect.py` is frozen at `b3be9f705e85121f4547ef9c526c16b94088cc2ce75b5ffc34643be52085a567`. `cpu-gate-result.json` independently verifies all 75 file hashes against the old rank-0 report, identical checkpoint metadata, teacher input tokens, prompt/prefix identity, positions, and finite BF16 residual shape `[32,6144]`.

Six current-device norm cases passed, including rounding, cancellation, signed zero, and three real residual rows. CPU inspection of the saved outputs confirms byte-identical normalized outputs and expected BF16 add residuals. Recovery calls the same `npu_add_rms_norm_bias` with the saved residual and a signed zero; it does not substitute ordinary RMSNorm. Actual launch evidence records twelve key-33, one-block launches of `AddRmsNormBias_352d2859a07d64c080e8dfc836d9897f_33`. This binds the current execution; the historical norm ELF/key remains unknown.

The router uses current M1 BF16 normalized inputs, BF16 gate weights, FP32 GEMM output, and the actual `DeviceOperator.moe_gating_top_k` with sigmoid, correction bias, eight selections, one group, renormalization, and scale 2.5. `teacher-routes.pt` stores `[75,32,8]` IDs and `[75,32,256]` FP32 logits. `summarize.py` independently recomputes every histogram and checks `teacher-unions.json`.

## Logical weight budget

For TP16, each distinct expert has two gate/up W4 matrices and one down W4 matrix with local intermediate width 128: `3 × 6144 × 128 / 2 = 1,179,648 bytes` per rank. Summing each expert once per layer-window gives the following W4 payloads; scale metadata, repeated kernel loads, cache residency, and all other model work are excluded.

| Width | W4 bytes across all 32 teacher rows | Mean W4 GiB per 75-layer window |
| --- | ---: | ---: |
| 8 | 15,695,216,640 | 3.6543 |
| 16 | 12,044,206,080 | 5.6085 |
| 32 | 8,790,736,896 | 8.1870 |

These are logical payload counts, not measured HBM traffic or TPOT. The existing estimator's scale terms are kept separately in the raw union file and are not asserted as the current fused operator's traffic.

## Expert-local tile geometry

In the 35 M32 windows rejected by the global count guard, 49 of 3,198 active `(layer, expert)` pairs have more than 16 rows: 1.5322%. The other 3,149 pairs meet the wide-tile row bound. For positive expert count `c≤32`, the old geometry uses `12×ceil(c/32)` tiles; allowing wide only for `c≤16` uses six tiles there and preserves the old geometry for every larger expert.

This gives 38,376→19,482 tiles in those 35 windows (−49.2339%), and 63,900→45,006 across all 75 windows relative to the current global guard (−29.5681%). These are exact tile counts under the stated formulas, not runtime or HBM savings. This counterfactual retains the old geometry for a 17-row expert. A separate fixed-wide experiment may allow its two M tiles and measure the extra weight read against the saved tiles of other experts. `group-budget.py` and `group-budget.json` retain the per-layer counts and input hash.

## Files

The server directory is `/workspace/experiments/teacher-input-router-20261020`. Raw tensors remain there. This directory retains the scripts and small receipts: `norm-result.json`, `cpu-gate-result.json`, `import-order-cpu.json`, `routes-result.json`, `routes-prelude.json`, and `teacher-summary.json`.

The first route attempt failed before any route kernel because directly importing `DeviceOperator` entered an import cycle. `preload_routes.py` (`087522f4…`) imports `vllm_ascend.ops` first, then executes the unchanged collector. A real CPU import check observed `torch_npu.npu._initialized == False` before and after. The failed first attempt remains under `routes.log`, `routes-exit.json`, and `routes-failure.json`; use the `routes-retry01` files for the successful attempt.

Raw tensor and full debug-log hashes remain in `server-manifest.json`; those large files stay on the server.
