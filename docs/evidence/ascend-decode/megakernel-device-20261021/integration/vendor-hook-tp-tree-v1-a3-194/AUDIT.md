# Three-rank whole layer with both reductions inside the launch

The candidate runs one layer launch containing both FP32 tree reductions. The
reference keeps the original input norm and quantization, native attention,
complete FFN and explicit `BF16.float -> HCCL SUM -> BF16` reductions. Both arms
retain the explicit Top-K copy and the next layer's output and residual
dependencies. This is the first arm in which the tensor-parallel reductions are
not outside the fused layer.

Both the eager arm and the official ACL graph arm pass on a3-194 physical dies
4, 6 and 7, with Torch 2.12, Torch-NPU 2.12 and CANN 9.2. Thirteen CPU peers
supply constructor geometry without initializing an NPU.

| Arm | Raw byte checks | Tree checks | Captures | Explicit replays | Unchanged capture boundaries | Python calls during replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 5,424 | 192 | 0 | 0 | — | — |
| graph | 5,424 | 192 | 6 | 48 | 678 | 0 |

Each arm covers M1 and M12 with eight changing inputs per width, and each case
checks 113 boundaries exactly. The three-rank oracle is
`(partial[0].float() + partial[2].float()) + partial[1].float()` with one final
BF16 conversion; all 192 tree checks compare against it with three distinct
partials. Every rank's parameter and output witness is checked, and the three
ranks agree on all common fields while differing on all partial fields.

Capture leaves all 678 observed boundaries unchanged and replay returns to
Python zero times. The communication window is 2,359,744 bytes per rank across
both shapes and all changing inputs, advancing four device epochs per two-layer
execution; host cache reset and poison do not touch it. The NZ storage
correction adds no HBM: all four banks keep their data pointer and raw digest.

## Capture does not change the answer

[`cross_arm_check.py`](audit/cross_arm_check.py) compares the two arms case by
case. All 96 retained output digests are identical across three ranks, sixteen
cases per rank and two boundaries per case, so graph capture and replay
reproduce the eager result exactly rather than merely passing their own oracle.

## Scope

No TP16 device collective executes here; `tp16_collectives_executed` is zero in
all six rank reports. Parameters are synthetic TP16-shaped shards, the
checkpoint is not loaded, and the arms are not timed. Whole-layer fusion of the
four native stages into a single launch, TP16 execution and model TPOT each
require their own measurement. The paired timing for folding the reductions is
separate and records 23.049 and 23.264 microseconds saved per two-layer pair,
confounded with the SHMEM-versus-HCCL choice; see
[`vendor-hook-tp-timing-m12-v1-a3-194`](../vendor-hook-tp-timing-m12-v1-a3-194/).

## Reproduction

    python3 run/run-tp-tree-device-v2.py --kind graph --attempt 1 \
      --manifest-sha256 071bc4e035d559fde4641bee95665f01179eedf7345e78f40adced5b377d908c \
      --candidate-sha256 7ec3e8fdd6155d99d98a9b69d2d4b02dc43c9417695f7632ad6559f42ecb96b3 \
      --cpu-report-sha256 fe7e4391349a6912a05162e2b89f502733e0a52c22036e97411f56a7ef96c0be
    python3 audit/audit.py --raw <archived raw> --fixture <staging> --kind graph --output <json>
    python3 audit/cross_arm_check.py --raw raw

The runner verifies the frozen fixture manifest, every pinned library digest,
the CPU report, 899 device-side input digests, host memory and two idle-device
samples before launching. The fixture contract is
[`tests/glm52/whole_layer/vendor_hook_tp_tree/README.md`](../../../../../../tests/glm52/whole_layer/vendor_hook_tp_tree/README.md).
