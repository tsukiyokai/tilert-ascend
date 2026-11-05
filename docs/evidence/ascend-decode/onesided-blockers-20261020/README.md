# Five blockers between the one-sided reduce and a number

2026-10-20, a3-84, sixteen ranks. Eleven launches. The lever had never
produced an end-to-end figure across four earlier sessions, and the reason was
never the machine.

## The baseline it would be measured against

    base: tpot [12.2866, 11.2817, 11.2522, 11.2312, 11.2599, 11.2689]
          rounds 194  accepted/round 5.273  round 63.869 ms  first [382]

Acceptance is **5.273**, identical to the figure measured on 194 -- so it is
deterministic and not a property of the machine. The round is 58.7 to 63.9 ms
against 194's 56.9, so a3-84 runs about 3% slower and a pairing has to be
taken within one machine.

## What was actually wrong

`--onesided-reduce` **takes a value**, a config-store endpoint, and is not a
bare switch. Passed bare, argparse exits 2 on all sixteen ranks in nine
seconds, writes no traceback, and puts nothing in the arm's own log -- the
failure lands in `launch.log` instead. This is what the earlier "rank-15
silent exit" was.

`onesided_reduce.py` was **missing from this tree**, leaving only a stale
`.pyc` in `__pycache__`. The module is in the repository and in five other
trees.

`aclshmem_torch.so` was **stale against its own kernel**: built 2026-10-18
22:32 against a `libtp_push_kernel.so` rebuilt 2026-10-19 07:53, which had
gained an `int32_t* errors` parameter -- the per-core probe buffer. The binding
failed at `dlopen` with `undefined symbol: _Z12tp_push_demojPvmPfS0_Phiiiii`,
and `torch.ops.load_library` reports only "Could not load this library",
naming no symbol. Fixed by declaring the parameter, allocating 1024 int32
zeroed as `main.cpp` does, and rebuilding.

The binding's **sibling kernels are in DT_NEEDED but their directory is not
on any search path**, and `LD_LIBRARY_PATH` cannot carry it because the arm's
`activate.sh` resets the environment -- the same trap the launch script
already documents for `ASCEND_CUSTOM_OPP_PATH`. Resolved by preloading the
chain by full path and globally: `libshmem_utils`, `libshmem`, then the three
kernels.

And `SHMEM_LOG_LEVEL` turns on the SDK's own trace in `/root/shmem/log/`,
which is the only instrument that distinguishes the causes below.

## The gap that remains, stated precisely

`attr_init` returns -4. The SDK's log gives the chain:

    Set device id to be -1 failed: 107001       mem_entity_default.cpp:85
    hybm import failed, result: -4              shmem_init_backend.cpp:588
    exchange slice failed, result: -4           shmem_init_backend.cpp:748
    return shmem error: -4 - setup_heap() failed  shmem_init.cpp:1234

`inited_device_id` starts at -1 (`mem_entity_entry.cpp:38`), is assigned only
at the end of `hybm_init` (:139), and is read by `SetThreadAclDevice` through
`HybmGetInitDeviceId()`. With INFO on, rank 0 shows `hybm_init` **succeeding**:

    success to get logic device user id=0, logic deviceId = 0   20:21:17.803
    hybm_init end device id 0, logic device id 0                20:21:17.804
    init_device_state success. world_size: 16                   20:21:19.028
    RankId: 0..15, vaddr 0x12c8c0000000, size 1080033280        20:21:19.083
    Set device id to be -1 failed: 107001                       20:21:19.149

So in one process the global is 0 at 17.804 and reads -1 at 19.149, with only
the peer-slice VMM mapping between. Both the writer's getter and the reader
are in `libshmem.so` and nothing else on the box defines either, so the
obvious explanations do not hold: there is one copy of the global, the
environment preloads no other `libshmem.so` (`activate.sh` does not mention
shmem and sets no path for it), and nothing assigns -1 after :139.

## Three wrong turns, and what they have in common

Blocking signals around the handshake, on the module's own docstring, which
attributes -4 to EINTR misclassified as a link error. `EINTR` is already in
`g_errnoToReconn` (`acc_tcp_link_default.h:29`); that docstring records a
*different* -4.

Binding the thread's device with `aclrtSetDevice` after `_load()`. The log did
not change by one character.

Moving the bind before `_load()`, reasoning that the backend reads the device
in its constructor at dlopen. Also no change.

Each was plausible, each was cheap, and each was made before reading the log
that was already on disk. One error code, several causes; the code is not the
diagnosis.
