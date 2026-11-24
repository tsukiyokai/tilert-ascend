# The width-six gate, and one suspect excluded without a machine

`install_gather_fia` raises above width six because "widths above six failed
serial-equivalence checks". That gate bounds the tree -- which wants 16 to 24 --
and `gather_fia`'s own 21.3 us a layer. `fia-mask-20261020` recorded the failure
as unisolated and it has stayed that way.

## Excluded: the sentinel index

`selection_mask` scatters zeros at `topk_indices`, and the engine's indices
carry `-1` for unused slots -- `sfa_v1.py:175` filters them explicitly where
this does not. A `-1` scatters to column `capacity - 1`, which would make an
arbitrary position visible.

It cannot: the causal term is `columns > slot`, so column 9343 is masked for
every query whose own slot is below 9343. At an 8K context in a 9344 cache no
query reaches it. The sentinel is harmless here and is not the divergence.

## The remaining suspect, and why it fits the width

`attend` shares one mask across the backbone: the first index producer
refreshes `self.mask` and every consumer reuses it with its own Q and KV. Which
layers produce is decided by `impl.skip_topk`, and the checkpoint distributes
that by `(i - 2) % 4` -- `index_skip_topk_offset` in its config.

A shared selection is the design, declared by `index_share_for_mtp_iteration:
True`. But the mask also carries the causal term, which depends on each query
row's own slot: at width w the rows occupy slots p through p+w-1, so the
boundary differs per row. A mask built at one width and reused by consumers
whose rows sit at the same slots is fine; what is not obviously fine is a width
that spans more than one producer's interval.

## The experiment, ready for the machine

Serial against speculative at widths 4, 6, 8 and 12 with `gather_fia` and the
gate lifted, comparing delivered tokens. The engine's `--attention-kernel`
selects it and the record already carries `precision_verified`. If the
divergence tracks the producer interval rather than the width, the fix is to
refresh per consumer rather than per interval -- and the gate opens, which is
worth 21.3 us a layer and the tree.
