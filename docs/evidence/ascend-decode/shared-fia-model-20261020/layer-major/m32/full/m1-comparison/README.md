# Frozen M1 versus M32, 78 layers

Rank 0 M32/SFA differs numerically from the frozen serial M1 trajectory: 582
of 660 boundary/cache tensor pairs agree bit for bit across 78 layers and
32 positions, 8192–8223. The comparison covers 93,340,160 values. Hidden and
residual outputs agree bit for bit through layer 67. Final normalized BF16
states have NRMSE 0.0239811737, maximum absolute difference 0.234375 and
105,925 differing values out of 196,608. These are diagnostic measurements
without a tolerance gate.

The first saved cache difference is layer 58, query 16 / position 8208:
`index_keys[16,5]` is 65→66. Nine INT8 values differ by 1 in that row; its
scale is identical, 0.0297698974609375. All input/output hidden, residual
and merged tensors, latent/rope rows and selected sets still agree at
that layer.

The first hidden/residual difference is layer 68, query 13 / position 8205.
Its input hidden/residual/merged and latent/rope rows agree bit for bit.
Only query 13 changes in the outputs. The first hidden difference is
column 1, −0.08935546875→−0.0908203125; query 13 hidden NRMSE is 0.00666139406.

## Index order and causal evidence

The preceding producer 62 has identical ordered query 13 indices. Producer 66
exchanges columns 1479/1480: M1 has `[2079, 5196]`, M32 has `[5196, 2079]`.
Consumers 67–69 use `skip_topk=true`; each arm restores the complete
producer 66 row exactly. The runner copies `shared[rows]` into the layer
buffer at `run_cann_layer_major.py:266–269` and preserves `shared` at 326.
The source SHA is recorded in [diagnostic.json](diagnostic.json).

Layers 66 and 67 still have identical hidden/residual outputs. At layer 68,
the inherited order difference and M1/M32 SFA arithmetic coexist. Their
causal contributions remain unresolved without a controlled intervention.
Layer 58 query 16
is later than query 13 and cannot cause that earlier row under causal
attention. The experiment also passes only hidden, residual and shared
indices between resident layers.

Selected sets first differ at layer 70 / query 13: 2045/2048 entries overlap.
Entries present only in M1 are 1450, 2380, 5096; entries present only in M32
are 2323, 5822, 7195.
This follows the layer 68 output difference.

## Hidden/residual NRMSE

NRMSE is `||M32−M1||₂ / ||M1||₂`, over all 32×6144 values. Full precision
values for every layer are in [diagnostic.json](diagnostic.json).

| Layer | Hidden | Residual |
| --- | ---: | ---: |
| 0–67, each layer | 0 | 0 |
| 68 | 0.000796813766 | 0.0000646686877 |
| 69 | 0.0357695472 | 0.00166182690 |
| 70 | 0.0272557894 | 0.00673796730 |
| 71 | 0.0462823301 | 0.00850444347 |
| 72 | 0.0441675999 | 0.0114351494 |
| 73 | 0.0441323888 | 0.0129893535 |
| 74 | 0.0637262054 | 0.0150311125 |
| 75 | 0.0625327680 | 0.0175202006 |
| 76 | 0.0254509483 | 0.0197100799 |
| 77 | 0.0152625602 | 0.0210499717 |

## Identity and reproduction

Prompt bytes, teacher trajectory, captured prefix, checkpoint metadata
and package versions match. The 404 GB checkpoint payload was not rehashed.
The 17 current source hashes match the completed rank 0 report. Source and
runtime differences are recorded in [identity-result.json](identity-result.json),
[identity-preflight.json](identity-preflight.json) and
[source-differences.json](source-differences.json). Source, runtime and
query width change together, leaving their individual causal effects
unresolved. Final logits use
BF16 in M1 and FP32 in M32; final normalized states are compared directly.

The server evidence root is
`a3-194:trt-asc:/workspace/experiments/m32-full-teacher-20261020/m1-comparison`.
Run `bash command.sh` on the a3-194 host. [compare.py](compare.py) extends
the existing comparison of the first eight layers, verifies raw artifact hashes
and uses CPU only, one intra-op and one inter-op thread. Backend autoload
is disabled; `torch_npu` is absent. [result.json](result.json) contains all
per-tensor metrics; [first-difference-receipt.json](first-difference-receipt.json)
and [index-order-origin.json](index-order-origin.json) retain the targeted
raw-tensor checks. Original model reports and tensor files remain intact.
