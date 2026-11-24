# MTP parent handoff

The main decode boundary is `(main_KV[0:P], position=P, pending=g0)`.
MTP row P-1 consumes `g0` with the main model's normalized hidden[P-1].
Its output supplies the first draft token, normalized recycle hidden and
top-2048 index set. Transferring the parent costs 12 KiB per rank and
allows this boundary to be established with one MTP row.

## Source contract

The [pinned source audit](mtp-parent-source-20261017.json) records
vLLM `c4df37df` and vLLM-Ascend `f558b58c`, seven source hashes and nine
passing static checks. It applies to the launcher's eager, complete-prompt,
batch-one, PP1 path:

- The main model returns its final-norm hidden. The proposer pairs row p
  with token[p+1], retaining position p; its last token is the sampled g0.
- Ascend MTP applies `rot.weight` before hnorm. The transferred main parent
  is normalized and unrotated. Chained drafts use normalized MTP recycle.
- `bind_connector_metadata` runs inside the target forward context before
  main execution. `start_load_kv` can run after main execution, so parent
  capture is installed at bind time.
- With MTP enabled, connector finalization follows drafting. The producer
  synchronizes, writes KV and the parent, then acknowledges completion.
  All sixteen matching rank completions precede receiver readiness.

The inspected producer computes the first MTP pass even for `max_tokens=1`;
the current export retains P-1 MTP rows. Main parent hidden and a cached
MTP row have different roles: skipping the seed would also require the
prefill drafter's token, recycle hidden and indices.

## Optional capture

Set `TRT_MTP_PARENT_HIDDEN=1` for
[serve_vllm_prefill.sh](../../../tests/glm52/serve_vllm_prefill.sh) and
`--mtp-parent-hidden` for
[capture_pd_prefix.py](../../../tests/glm52/capture_pd_prefix.py).
The connector field is `transfer_mtp_parent_hidden=true`.

Layout v3 appends one BF16 `[1,6144]` region to the 196 KV regions;
the default v2 layout retains its offsets and sizes. A target forward hook
copies the last row into persistent staging storage. The request identity
and complete prompt length must match before that pointer is used.
The same synchronous HIXL WRITE covers KV and parent before `done`.

The frozen MTP bundle stores `parent_hidden` with its owned CPU tensor,
position P-1, `normalized=true`, `rotated=false`, request ID and generation.
It retains P-1 layer-78 cache rows. Resetting the receive allocation cannot
change the saved parent.

## Acceptance

The [CPU checks](check-parent-hidden-20261017.json) exercise real Torch
hooks and the installed V1 connector base, with sixteen local TCP senders
and a HIXL WRITE double. All nine pass. They cover the last row, staging
lifetime, identity, layout flags, serialized ownership and delayed
completion. They initialize
no NPU. The default-path regressions also pass: layout 63, connector 29,
PD lifecycle 33 and prefix capture 8. Their reports and the 15 static/CLI
checks are in [the host evidence directory](mtp-parent-host-checks-20261017).
Device and numerical acceptance follow separately.

The [CANN MTP execution path](cann-mtp.md) now connects the v3 parent and
P/P-1 caches to the existing TP16 rounds and HTTP service. CPU API and
streaming lifecycle checks pass. The next device check is an 8K v3 capture
with sixteen receipts, followed by real CANN MTP seeding and HF comparison.
The full-P [CANN M1 adapter](cann-model.md) runs independently of this
optional field. The Native engine retains its own replay entry.
