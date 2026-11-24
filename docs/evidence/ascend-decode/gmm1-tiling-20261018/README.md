# Existing CANN GMM1 screens

## baseN=128

A3 shared die 0, one TP16 weight shard, real layer-3 M4 routed inputs:
32 active experts, each with one row. The [private host patch](base-n128.patch)
changes only `baseParams.baseN` and `mmTiling.baseN` from 256 to 128 for
the fixed A8W4 MSD shape K=6144/N=256. The validated SwiGLU device binary
and its barrier remain unchanged; the other SDK tiling fields are preserved.

| Clone + complete GMM1 graph | p50 (µs/call) | p95 |
| --- | --- | --- |
| Existing baseN=256 | 121.525 | 482.393 |
| Candidate baseN=128 | 124.588 | 488.564 |

Each case has ten calls/graph, ten warmup graphs and thirty timing samples.
MSD mutates its input, so each captured call clones the same frozen A8 rows.
Eight eager checks and the last saved graph output match the actual M4
fixture exactly. The [audit](m4-audit.json) independently recomputes timings
and verifies 4,128 output elements per saved graph; other timed outputs
were not archived. A separate debug execution confirms final `baseN=128`
and passes all eight functional replays. SDK logs printed before the
override still show 256; the final operator tiling is 128.

There is no measured gain. Retain the existing tiling and stop this branch.
These shared-device times do not measure model TPOT. Raw fixtures, the
58-check source audit, vendor package, launches and logs remain under
`/home/shanshan/trt-asc/offline/cann-gmm1-basen128-20261018-01/` on a3-gs.

## Single-chunk synchronization

The [single-chunk audit](single-chunk-audit.json) keeps tiling, arithmetic,
MSD boundaries and the SwiGLU barrier unchanged. Only `loopCount==1` skips
the next-prefetch AIV barrier and the duplicate mixed barrier before final
postprocessing. The pre→matmul and matmul→post mixed barriers remain.
At K=6144/N=256, `mLimit=4681`; M1/M4 dispatch eight/thirty-two rows, so
both execute one chunk. Other loop counts retain the original barriers.

Four separate processes run baseline/candidate/candidate/baseline. Each
records thirty event samples of ten captured clone+GMM1 calls:

| Process order | Event p50 (µs/call) |
| --- | --- |
| Baseline | 116.814 |
| Candidate | 111.299 |
| Candidate | 112.741 |
| Baseline | 114.584 |

The mean of process medians improves by 3.679 µs, or 3.18%; AB/BA gains
are 5.515/1.843 µs. Every process has six long batches, with roughly
3.6 ms extra per ten-call batch; their cause is unresolved. Per-batch
host wall time was not recorded. This is a shared-die event screen.

Changing row/scale inputs, poisoned outputs, dirty reuse and recapture
compare 881,504 elements exactly. Actual ELF opens select the control and
candidate binaries in those functional runs; timing processes retain the
same bindings but do not capture opens. Raw artifacts stay under
`/home/shanshan/trt-asc/offline/cann-gmm1-singlechunk-20261018-01/` and
`/home/shanshan/trt-asc/offline/cann-gmm1-singlechunk-changing-20261018-01/`.

The [kernel-only builder](../../../../tests/glm52/build_gmm1_barrier.sh)
applies the SwiGLU and single-chunk patches in an isolated source copy,
then calls the installed dynamic wrapper through `opc`. The
[actual build](build-audit.json) completes in 19.82 s. Its selected ELF and
descriptor are byte-identical to the measured candidate; installed
host/API/tiling libraries retain their bytes. Relative to the original
vendor, four non-cache files change: two headers, ELF and descriptor.
Python cache differences are recorded separately. No package-wide host or
protobuf rebuild is required.

The [complete-MoE control](moe-baseline-audit.json) executes the genuine
TP16 layer-3 router→dispatch→GMM1→GMM2→combine path. M1 uses two actual
input rows separately; M4 compares the actual four rows with a synthetic
repeated-row case. Routing changes from 32 experts with one row each to
8 experts with four rows each. Own eager/Graph/dirty/reset/recapture and
decoder-prehook comparisons cover 13,249,536 elements exactly, with
1,539,309,568 bytes peak reserved/die. The launch then stops on an ELF-open
receipt that also counted its verifier's hash reads. The corrected receipt
separates the exact ordered verifier prefix from one runtime control open
on every rank. Candidate execution and complete-MoE cross-configuration
comparison remain unrun. Raw inputs and candidate-only continuation stay
under `/home/shanshan/trt-asc/offline/cann-gmm1-singlechunk-moe-20261018-01/`.

## Metadata candidates

The [source and installed-host audit](metadata-contract-audit.json) finds
that ordinary views of the existing five-dimensional NZ bank retain its
storage descriptor, whereas the multi-tensor API checks for four-dimensional
storage per weight. Passing active-expert views therefore does not satisfy
that API contract; no device rejection or speedup was measured. Dense
counts→prefix metadata is a smaller available experiment, but still leaves
the matmul's group traversal. A production comparison must charge the
dynamic prefix construction. Neither candidate has a benchmark driver yet.
