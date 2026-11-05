# EP16 loses to TP16 at batch one, and the fused operator makes it worse

`a3-84` idle, the composed flag set with `HCCL_OP_EXPANSION_MODE=AIV`,
`--max-tokens 1024 --warmup 3 --requests 3`, profiler off. Only the MoE
parallel layout differs; attention stays TP16 throughout.

| case | TPOT median (ms) | spread | rounds | A | per round (ms) |
| --- | --- | --- | --- | --- | --- |
| TP16 | **14.247** | 0.252 | 298 | 3.4329 | **49.07** |
| EP16, no fusion | 15.150 | 0.521 | 310 | 3.3000 | 50.16 |
| EP16 + `MegaMoe` | 15.375 | 0.756 | 306 | 3.3431 | 51.57 |

EP changes the association of the MoE sum, so rounding differs, the greedy
argmax eventually diverges and the trajectory is not comparable token for
token. The per-round column is therefore the honest comparison, and it says
EP16 costs 1.09 ms more per round than TP16 — and that **adding the fused
operator costs another 1.41 ms** rather than saving the two GMMs' prologues.

The run spread grows monotonically, 0.252 → 0.521 → 0.756 ms. That is the
signature predicted for EP at batch one: with 27.1 activated experts over
sixteen buckets there is no statistical smoothing, the busiest bucket holds
four to five whole experts, and `combine` is a per-layer barrier, so every
round waits for its own worst rank.

## The flag that silently does nothing

`enable_fused_mc2 = 1` does not enable the fused operator. vLLM-Ascend rolls
it back internally (`vllm_ascend/ascend_config.py:663-673`):

```python
# TODO(zzzzwwjj): Currently, there are many problems with the megamoe op.
# We will first roll back the megamoe internally and keep `enable_fused_mc2=2`
# to enable the megamoe for testing capabilities.
if self.enable_fused_mc2 in (0, 1):
    _MEGA_MOE_SUPPORTED = False
elif self.enable_fused_mc2 == 2:
    _MEGA_MOE_SUPPORTED = importlib.util.find_spec("cann_ops_transformer") is not None
    self.enable_fused_mc2 = 1
```

So 1 reaches `dispatch_ffn_combine` and only 2 reaches `MegaMoe`, after which
the value rewrites itself to 1 — the config echo cannot distinguish them. The
first EP measurement here was taken with 1 and is the "no fusion" row; the
engaged run shows `MegaMoe` 17 times in its launch log.

A second trap sat in my own verification. The config check ran under
`python3 -I`, and isolated mode drops the `PYTHONPATH` entry that puts
`/opt/ascend/cann-9.2.0/python/site-packages` on the path, so
`find_spec("cann_ops_transformer")` returned `None` and the probe reported
`mega_moe_supported=False` even with the flag at 2. The benchmark itself runs
through `torch.distributed.run` without `-I`. The habit of adding `-I` to
contain untrusted downloads manufactured a false negative here.

## What this settles about the parallel choice

The earlier reasoning stands and is now measured: EP16 at batch one trades a
fixed cost for a tail, and the tail is worse. It removes 78 reductions per
round, which the audit called a prerequisite for 2 ms, and still loses — so
the number of collectives is necessary to cut but not sufficient, and EP16 is
not the way to cut it.

What survives from the argument is the shape of the fix rather than this
instance of it. The tail scales with the number of whole experts in the
busiest bucket, so the candidates are EP4×TP4, where the busiest bucket holds
about 1.5× the mean and `intermediate` per rank is 512 — exactly `MegaMoe`'s
lower bound — or hot-expert replication, which batch one can afford because a
single request leaves roughly 44 GB per die free: a second copy of the
experts costs 22.6 GB and lets routing pick the less loaded replica without
touching the arithmetic.

Neither is tested. What is tested is that the straightforward EP16 route,
with or without the vendor's fused operator, is slower than the TP16 baseline
it was meant to beat.
