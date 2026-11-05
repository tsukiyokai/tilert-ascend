# Selected conditioned MTP head

Disposition: feasible as a separate conditioned-body/head interface, but close implementation work for the current target. The saved width8 phase floor assigns the entire conditioned pass 2.080 ms out of 52.857 ms per round (3.94%). The old seven-rank profile assigns its entire head a median 219.986 µs of device tasks. A selected M1 head remains necessary. Neither number is a measured removable critical-path saving or a current M4/M32 result.

## Actual dependency

`mtp_round.py:121–132` runs target verify, rolls draft KV back to the main start, runs all conditioned MTP rows, reads candidates/target predictions once, resolves the commit and retains row `committed-1` as the next seed. `cann_model.py:590` performs that read with `.tolist()`. Removing the conditioned head preserves the existing host read, but places the selected head after host acceptance processing, on the dependency path to the next chain. Eager row slicing introduces no necessary second device-to-host read; a head graph would require a stable one-row staging buffer.

`cann_mtp.py:294–356` and `cann_mtp_graph.py:56–70` always compute the full head. `full_logits=False` only discards returned logits. The installed vLLM `deepseek_mtp.py:136–144` returns `(pre_norm, shared_head(pre_norm))`; its `compute_logits:226–236` applies `shared_head` before the logits processor. Thus recycle is already normalized. It must not be passed to `compute_logits` as pre-norm input.

The installed Ascend wrapper `vllm_ascend/models/deepseek_mtp.py:23–34` rotates the parent before the body; the loaded own head is exposed at lines 36–54. Keep both operations and the installed `cann_head.py:7–28` BF16-input/BF16-weight→FP32 projection. A selected row still requires the complete vocabulary shard and TP collective; its cost does not scale with the number of rows removed.

## Minimum interface if reopened

1. A conditioned-body call produces all M rows of pre-norm, recycle, indices and KV before `host_tokens`. Seed/chain forwards retain their heads. Capture the conditioned graph explicitly without a head; `full_logits=False` cannot select this behavior in the existing graph.
2. Resolve acceptance and select `r=commit.committed-1`, after output-budget/EOS clipping. Project `pre_norm[r:r+1]` through the existing `compute_logits`. Alternatively pass the already normalized `recycle[r:r+1]` directly through the same logits processor and loaded own LM head, preserving FP32 projection, scale, crop, collective and argmax. The latter avoids reapplying final norm. Device equality between the selected and full-width projection remains unproved because MatMul tiling changes.
3. Form the next seed from the selected device token, the matching recycle row and that row's own indices. Every TP rank must select the same row and issue collectives in the same order. No model/KV forward is needed after the host read.
4. A head-only graph needs its own stable `[1,6144]` input and lifecycle. The existing M1 chain graph also executes rotation/body/KV writes and cannot serve as a head-only graph. Conditioned outputs are persistent replay buffers (`cann_mtp_graph.py:86–112`); consume/stage the selected row before another conditioned replay. Clone a graph-owned token before graph feedback, as `_seed:75–81` already does; close the new graph before the target worker.

## Counterexamples and evidence

With natural acceptance four and only one new output remaining, the next seed row is 0, not 3 (`mtp_state.py:200–220`). A numerical normalization example is saved in `counterexample.json`: pre-norm `[1,1]`, norm weights `[1,2]`, head rows `[3,0]` and `[0,1]` give top1 0 with one norm and top1 1 when normalized recycle is passed through another final norm. This is a scalar semantic example, not CANN validation.

`budget.json` binds the existing phase/profile summaries and current source hashes. Only the tiny scalar example ran on server CPU. No product code, device call or new performance experiment was created.
