# Frozen 8K attention differential

Layer 0 at position 8192 consumes token 6500 from the complete captured
prefix. The calibrated [DynamicQuant reference](dynamic-quant-div.md)
matches its input codes, Prolog query, new cache rows and all 2,048 indices.
The [evidence bundle](mla-8k-differential-20261017.json) binds the following
CPU and single-die experiments. Full-model precision remains open.

## Attention boundary

Die 15 executes the production Prolog, sparse flash attention and value
projection with logical rank 0's four heads. The frozen prefix has 8,192
rows; the call appends one row and selects 2,048 keys. Peak Torch reservation
is 130 MiB under a 1.5 GiB limit. No HCCL group or complete model loads.

| Same-input comparison | Different values | NRMSE |
| --- | --- | --- |
| Absorbed query and RoPE vs CPU Prolog | 0/2,304 | 0 |
| SFA context vs FP32 compressed oracle | 23/2,048 | 0.000468 |
| Value projection from the actual SFA context vs CPU | 0/1,024 | 0 |
| Complete local output vs FP32 compressed oracle | 170/1,024 | 0.001642 |
| Frozen full-HF pre-O slice vs local CANN output | 676/1,024 | 0.005195 |

The separate four-head probe expands only the selected keys and gives
0.004494 NRMSE against its HF endpoint. It does not reproduce the full
64-head HF call's matrix shapes. The full-HF pre-O slice above comes from
an independently replayed HF layer whose raw output is bitwise equal to
the saved complete-model HF trace. Moving SFA binding before Prolog and NZ
conversion leaves all eleven saved device fields bitwise equal.

## CPU provider arithmetic

For identical frozen BF16 probabilities and values, integer/rational dot
products establish the correctly rounded BF16 result for all 1,024 outputs.
FP32 and FP64 accumulation followed by BF16 both match it exactly. Default
CPU BF16 matmul differs in 417 values. Disabling oneDNN temporarily removes
all 417 differences; restoring it reproduces the original output bitwise.
The experiment preserves the other backend flags and never initializes
an NPU. It identifies a provider-dependent arithmetic difference without
identifying its instruction sequence.

Repeating the original HF layer with oneDNN disabled preserves the inputs,
indices and new cache rows. Local pre-O NRMSE against CANN falls from
0.005195 to 0.003326; completed hidden-state NRMSE falls from 0.003889 to
0.003595. This intervention leaves the full-model numerical question open.

## Dense branch

Fix the complete HF post-attention input and replay the production norm,
W8 gate/up, SwiGLU and down projection for each of the sixteen original
rank shards, sequentially on die 15. Normalized inputs, all gate/up outputs,
all 12,288 SwiGLU values and quantization pairs are bitwise equal to CPU.
One rank-8 down-projection value differs by 4.77e-7. CPU rank-ordered FP32
accumulation followed by BF16 cast, and the completed BF16 residual, are
bitwise equal to HF. Peak Torch reservation is 118 MiB. The report has
146/148 exact checks; its two failures compare that same partial value twice. No device
collective executes in this experiment.

## Output projection

The original full-HF pre-O tensor also supplies sixteen real W8 O shards,
each with 1,024 input columns and 6,144 outputs. All INT8 codes and FP32
scales match. Rank 7 has one BF16 partial differing by 3.81e-6; the CPU
rank-ordered FP32 sum followed by BF16 cast matches all 6,144 HF output
values. The device report has 49/50 exact checks and 58 MiB peak Torch
reservation. It uses the production loader, NZ postprocessing and linear
method, with no device collective.

The fixed-input norm, dense and O-projection branches close at their tested
BF16 merged outputs. Independent attention trajectories and the complete
TP16 execution remain open.
The complete 8K continuation must also execute the
[SwiGLU barrier overlay](swiglu-barrier.md) before reset and full-model
numerical acceptance can pass.

## Raw artifacts

Under `/workspace/offline` on `a3-gs:trt-asc`:

- `cann-mla8k-20261017`: `probe.py`, `result.{json,pt}`, the `early-bind`
  replay, and `pv-rounding-01.py` / `pv-provider-01.py` with their reports.
- `cann-layer0-dense-20261017`: the exact HF `pre_o_replay.py`,
  `pre-o.{json,pt}`, `device_replay-01.py` and `device-01.{json,pt}`.
- `cann-layer0-dense-20261017/no-onednn`: the isolated CPU replay and
  `pre-o.{json,pt}`. Original frozen artifacts remain unchanged.
- `cann-layer0-dense-20261017/o-proj`: `device_replay.py`, CPU
  `prepared.{json,pt}` and the actual `device-15.{json,pt}`.

The scripts and reports retain commands, checkpoint/input hashes and source
hashes. These experiments measure numerical behavior; they contain no
decode latency or profiling result.
