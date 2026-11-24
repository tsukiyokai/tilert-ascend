# Layer-0 indexer at 8K

The [single-die replay](../../../tests/glm52/replay_cann_indexer.py)
reproduces the first frozen HF/CANN indexer difference at position 8192,
token 6500. It uses the complete BF16 index-key prefix, actual weights,
production loaders, Prolog and paged LightningIndexer. Two independent
processes retain identical query, key, weights and selected indices;
peak reserved memory is 144 MiB on shared die 15.

| Boundary | Actual comparison |
| --- | --- |
| Input RMSNorm | 6,144/6,144 BF16 values exact |
| Input DynamicQuant | Scale exact; 10/6,144 INT8 codes differ by one |
| Prolog with those actual A8 inputs | 2,048/2,048 query codes exact; scale differs by one FP32 ULP |
| Index key / head weights | 128/128 and 32/32 BF16 values exact |
| Full backend vs HF | 2,047/2,048 choices agree; CANN selects 5950, HF selects 3782 |
| Same selector, fixed HF q/k/weights | 2,048/2,048 choices agree |
| CPU FP32 scores from actual CANN q/k/weights | Reproduce all 2,048 CANN choices |

The [raw verification](cann-indexer-8k-20261017.json) checks tensor hashes,
shape, dtype, index range, uniqueness, repeated output and independent
score reconstruction. Scores are reconstructed on CPU; the paged kernel
returns indices only. Choice comparisons use sets; repeat checks also
preserve order.

All ten input-code differences occur at `abs(x)=amax/2`: CPU arithmetic
reaches `abs(x*Div(127,amax))=63.5` and rounds to 64, while CANN returns 63.
With HF inputs the cutoff score is 19.32767; the scores of 3782 and 5950
are 19.59767 and 19.18323. With actual CANN query values their scores become
19.40633 and 19.76605, reversing selection with a 0.35973 gap.

The [native Div calibration](dynamic-quant-div.md) resolves this local
difference: 31,744 BF16 maxima establish an exponent-independent factor
correction. The updated CPU reference reproduces all input/query codes,
the BF16 query and all 2,048 selected indices. Complete TP16 continuation
with the SwiGLU barrier and updated HF reference is the next acceptance.
Full-model parity still precedes profiling.

The historical replay below uses the original source snapshot with the
IEEE CPU formula, preserving the reported 2,047-choice comparison.

```bash
source /workspace/environment/activate.sh
cd /workspace/offline/cann-routed-stage-20261017/source
PYTHONPATH="$PWD:$PWD/tests/glm52:${PYTHONPATH:-}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  python /workspace/offline/cann-indexer-20261017/replay_cann_indexer.py \
    --checkpoint /workspace/models/GLM-5.2-w4a8c8 \
    --trace /workspace/offline/cann-routed-stage-20261017/continuation8k/rank-0/step-08192.pt \
    --reference /workspace/offline/cann-layer4-default-20261017/hf-8k/step-08192.pt \
    --prefix /workspace/offline/prefix8k-allocation-20261017/capture/case-00/main-prefix/rank-0.pt \
    --output /tmp/indexer-replay --device npu:15
```
