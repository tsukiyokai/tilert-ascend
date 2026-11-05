# CANN norm/quant contract audit

CPU audit of captured A3 outputs is complete. A fresh single-die replay is
pending: on 2026-10-20 the six A3 hosts in `lsenv.md` had workloads on every
die. The user requires an idle die; sharing a resident workload is excluded.

## Captured GLM inputs

The three dense layers and two M8 calls contain 294,912 INT8 outputs.
Independent PyTorch FP32 and NumPy FP64 formulas agree with CANN except
seven INT8 values differing by one. Exact rational arithmetic places all
seven at ±63.5 before rounding. Residual outputs match exactly; scale error
is at most 2 FP32 ULP, or 1.37e-7 relative. [Results](result.json).

These results show no contract violation on the captured inputs. Passing
op-plugin's broad test tolerance alone is not the basis for this assessment.
The installed CANN 9.2 package and kernel file hashes are recorded in
[installed.json](installed.json); fresh execution of that binary is pending.

## Why fusion changes the trajectory

The deployed separate custom norm rounds to BF16 after residual addition,
after normalization before gamma, and after gamma/beta. Simulating those
three boundaries reproduces all saved norm and INT8 values for M1 and M8.
Simulating only the final boundary misses 1,295/6,144 norm values for M1.
[Comparison](separate-rounding.json), [CPU reproducer](analyze_separate.py).

CANN's fused formula keeps FP32 intermediates. Replacing the separate path
therefore changes its floating-point computation even when both operators
meet their own contracts. A preservation fix must retain all three boundaries.

## Reproduce

In `a3-194:trt-asc`, activate `/workspace/environment/activate.sh`. Inputs
are below `/workspace/experiments/m32-full-teacher-20261020/norm-fusion`:

```bash
python check_norm_quant_contract.py \
  --snapshots model-replay/on-full-78-02/output/rank-0 \
  --output contract-cpu-new
python analyze_separate.py device-01/outputs.pt separate-rounding-new.json
```

The replay script is `tests/glm52/check_norm_quant_contract.py`. On an idle
die, add `--inputs inputs.pt --device npu:<die>` and use a new output path.
It also covers seeded BF16/FP16 inputs, nonzero beta, tail dimensions,
repeatability and input swapping. No latency is measured.
