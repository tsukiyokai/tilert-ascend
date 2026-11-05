# GMM2 adaptive tiles

A3 a3-194/trt-asc, shared die15, CANN9.2.0 B114. The opt-in
`build_gmm2.sh NEW_OUTPUT --adaptive-tiles` adds tile selection to compact GMM2.
The default and `--compact-experts` modes retain their previous behavior.

## Mechanism

For E256/K128/N6144 COUNT routing, the publisher records whether every active
expert has at most16 rows. That case uses32×1024 tiles from the installed
CANN tiling API; otherwise it retains64×512. The table's low32 header bits
remain the active count. Shape guards and COUNT/BF16 template restrictions
are recorded in [bounds](bounds.json).

For an expert with c rows, tile count changes from12 ceil(c/32) to
6 ceil(c/16). Small groups halve the tile count;32 rows do not. Fixed wide
tiles made shared8 slower by10.00µs in the [same-ELF comparison](retile-bench_retile.json),
which motivates the dynamic fallback. NZ weight addresses and MSD arithmetic
remain unchanged. [CANN-generated tiling](tiling-32x1024.txt).

The existing AIC barrier moves before Matmul initialization so all cores see
the same decision. Its synchronization count is unchanged, but its timing is
not free. The FP32 temporary allocation now covers all four accessed regions;
wide/fallback UB peaks are184448/182400B against196352B available. The unused
offset view is inherited and not accessed under the guarded withOffset=0.

## Correctness

- [CPU schedule](schedule-cpu.json):299 cases cover output ownership, addresses,
  all33 tiling fields and AIV phase order. [Dynamic metadata](dynamic-cpu.json):
  12 cases cover header transitions, unchanged active counts and expert migration.
- [Production release](release-validate.json):46 changing-route Graph cases
  match independent CPU MSD arithmetic bit for bit, including16→17→16 rows
  per expert and changed IDs with unchanged active count.
- [Real checkpoint](release-real.json):layer4/position8192/rank0, four eager/
  Graph calls match CPU arithmetic and the frozen CANN output.
- [Loaded ELF](release-validate-access.json):SHA16e8bea5f250b048da83cbcecac5f0e77f0dd66886a0fc1412144c1d6f47a114.
  The real-weight run has its own [access receipt](release-real-access.json).
  [Patch audit](production-patch-audit.json) binds exact release sources.

## Whole-package timing

[Four separate process arms](release-summary.json), A/B/B/A: A is the released
compact ELF0fe4dbd8…; B is the adaptive release above. Each arm uses eight
full weight banks,128 calls/Graph,10 warmups and30 NPU-event samples, with
input restoration included. Profiler off, clocks unlocked, no samples removed;
peak reserved1796MiB. Every arm's output and route hashes match. Foreign Ray
weights remain resident; these are single-die screening measurements.

| Width / routing | A1 µs | B1 µs | B2 µs | A2 µs | B−A, adjacent pairs µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / random | 33.13 | 34.31 | 33.39 | 31.58 | +1.18 / +1.82 |
| 8 / random | 77.48 | 57.06 | 54.66 | 71.43 | −20.42 / −16.77 |
| 32 / random | 146.98 | 107.86 | 106.18 | 143.90 | −39.12 / −37.72 |
| 32 / shared8 | 38.09 | 45.09 | 40.77 | 34.99 | +7.01 / +5.79 |
| 32 / all256 | 212.81 | 153.04 | 142.95 | 210.09 | −59.78 / −67.14 |

M8/M32 dispersed routing improves; M1 and shared8 regress. Retaining the old
tile shape does not retain its original schedule. Keep this option disabled
until the actual lookup route distribution and whole-model teacher checks
select it. No request TPOT improvement is established.

The earlier adaptive test ELF used a pointer branch that skipped the header
read only in its control arm. Its shared8 timing is confounded and is not used
as the release benefit. Raw file `adaptive-bench_retile.json` remains on the
server. The preceding fixed-wide experiment keeps the same barrier placement
in both arms and remains a tile-shape comparison.

Code commit `cb75541`; source and launch copies are in [source](source/). Build identities are in
[inputs](inputs.sha256) and [artifacts](artifacts.sha256). Raw server root:
`/home/shanshan/trt-asc-194/experiments/gmm2-compact-20261020`.
