# The super kernel's 33.58 ms is a long tail with one outlier

637 distinct super kernels, each called once a round, 33.58 ms a round.

| kernel | us a call | ms a round |
| --- | --- | --- |
| `sk_1277_full_model_start_aiv_all_reduce_float_` | **3282.3** | **3.28** |
| `sk_681_..._static_kernel_AddRmsNorm` | 215.4 | 0.22 |
| `sk_1195_..._static_kernel_AddRmsNorm` | 212.7 | 0.21 |
| seven more AddRmsNorm | 208 to 211 | 0.21 each |

    the ten heaviest are 15% of super kernel time, 5.18 ms a round
    those whose name carries a collective: 6.87 ms a round, 20%

## The outlier

One kernel costs 3.28 ms a round -- a tenth of all super kernel time in a
single call -- and its name says it wraps an all_reduce on the vector cores.
Against the 858 us a standalone collective costs in the same capture, and the
177 to 187 us a sixteen-rank all_reduce measures at decode sizes under AIV,
3282 us is four to eighteen times too much for the collective it contains.

So either it carries far more than one collective, or the fusion costs more
than the parts. Which of those it is decides whether it is a lever: a fused
kernel whose collective is already AIV-expanded would not gain from the AIV
flag, while 3.28 ms a round is larger than the bank narrowing.

## The tail

Everything else is 637 kernels averaging 52.7 us, with the heaviest at 215 and
no second outlier. Nine of the top ten are `AddRmsNorm`, which is the residual
add fused with the norm -- a merge this project already took, at 1.46 ms.

So the super kernel's cost is not a few hot spots to attack but a flat
distribution across the model, except for the one all_reduce kernel. And 20
percent of it sits in kernels whose names carry a collective, which is where
AIV's 1.5 to 1.85x would land inside the super kernel rather than beside it.

## Reducing the argmax instead of the logits does not help

`compute_logits` all-reduces the vocabulary-sharded logits so `argmax(dim=-1)`
sees a global maximum, and the round only needs the token id -- so each rank
could take its own maximum and the pair (value, global index) be reduced
instead. Sixteen ranks under AIV with determinism:

    all-reduce the logits (7.3 MB) then argmax     356.79 us
    all-gather the (value, index) pairs (1.5 KB)   406.48 us
    ratio 0.88x, 49.68 us worse

Replacing 7.3 MB with 1.5 KB saves nothing, because both pay the same per-call
floor -- the same flatness measured all day -- and the pair version adds a local
max, two writes and a second argmax over the gathered values.

## And it reprices the outlier

A full-vocabulary fp32 all_reduce at decode sizes is **356.79 us**, against the
capture's 3282 us for the kernel that wraps one. That is 9.2 times, so the
3.28 ms is not the reduction it contains.

The capture is `max_tokens: 6, mtp_width: 4`, so its logits reduction spans
prefill widths rather than a decode round's twelve rows -- at 8192 rows the
payload is 683 times larger and the cost would follow the bytes once past the
floor. So the outlier is a prefill figure, like the collectives in the same
capture, and there is no 3.28 ms decode kernel to attack.

Which is the third time today that this capture's six-token scope produced a
figure that does not transfer. The steady-state capture the watcher will take is
what settles the super kernel's real distribution.
