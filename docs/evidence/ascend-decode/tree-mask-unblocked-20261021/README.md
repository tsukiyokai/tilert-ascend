# Tree verification is not blocked by the attention kernel

This project recorded that sparse attention takes only a prefix-length scalar
and no per-row mask, which made tree drafting wait on a kernel change. Reading
the operator the engine actually calls -- `sfa_v1.py:216` -- that is wrong.

    torch.ops._C_ascend.npu_sparse_flash_attention(
        query=..., key=cache, value=cache,
        sparse_indices=indices,
        sparse_block_size=1,
        block_table=metadata.block_table,
        actual_seq_lengths_query=metadata.query_start_loc[1:].to(torch.int32),
        actual_seq_lengths_kv=metadata.seq_lens.to(torch.int32),
        layout_query="TND", layout_kv="PA_BSND",
        sparse_mode=3, attention_mode=2)

Three things follow. `layout_query="TND"` has no batch axis, so a token is a
row. `sparse_indices` is a tensor indexed by that row. And
`sparse_block_size=1` makes a block a single token.

So each query row already chooses its own visible set, and the engine uses
exactly that to pick each token's top-k blocks. A tree mask is the same
mechanism with a different index set: for a draft token, the indices of its
own ancestors.

## What the blocker actually is

Constructing the indices, which is host-side. Each round the drafter knows the
tree's parent array, so a token's ancestor set is a walk up it, and the
per-row index rows are that walk padded to the operator's pick count. Nothing
in the kernel needs changing.

Two corrections to record. The claim came from reading `atten_mask`, which the
sparse operator indeed does not have -- `npu_fusion_attention` has one and
`npu_sparse_flash_attention` does not. And a probe written against
`torch_npu.npu_sparse_flash_attention` failed on a required `query_rope`,
which is a different implementation from the `_C_ascend` one the engine calls.
The engine's own call site is the specification.

## What it is worth

The tree accepts 14.510 a round at an edit rate of 0.02 against the
automaton's 10.020, and 6.844 against 5.434 at 0.05. A 40 ms round needs 13.8
accepted for 2.9 ms, which the tree reaches at the low end of the edit-rate
range an agentic coding model spends most of its output in.
