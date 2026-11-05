# Two defects in CANN SHMEM 1.7.0, and what they cost

Both were found while building a one-sided TP all-reduce on A3 and both are in
the SDK rather than in our code. `two-defects.patch` holds the diffs against a
clean `gitcode.com/cann/shmem` v1.7.0 checkout.

## An eight-byte kernel argument after a four-byte one is read out of bounds

`aclshmemi_barrier_on_stream_kernel(aclshmem_team_t team, uint64_t ffts_config)`
packs a 4-byte team ahead of an 8-byte address, so the address lands at a
non-8-aligned offset and the device prologue loads it from the next aligned
offset — past the end of the argument buffer. The kernel therefore received
the high half of the FFTS base, and **`aclshmem_barrier_all()` has never
worked on this machine**: one sixteen-rank run logged 964 instances of
`aclError 507015`, every one swallowed because the wrapper returns void.

Reordering the parameters fixes it. The same mistake appears nine more times
in the SDK's own examples.

The rule this gives our own kernels: eight-byte kernel parameters go first.

## EINTR is classed as a link error in the bootstrap handshake

`acc_tcp_link_complex_default.h` special-cases only `EAGAIN` at lines 353, 357,
382, 386 and 456; `EINTR` falls through to `ACC_LINK_ERROR`, the handshake
fails, and `aclshmem_init` returns −4 with
`Set device id to be -1 failed: 107001` as the visible symptom. Its sibling
`acc_tcp_link_default.h:29` lists `EINTR` in `g_errnoToReconn` and retries at
110, 191 and 273, so the complex variant is simply missing it.

This is why the identical call returned 0 in a standalone test and −4 inside a
vLLM worker: **a worker is full of signals** — the NPU watchdog, Python's
handlers, `SIGCHLD`, profiler timers — and a quiet `torchrun` process has
almost none. Five lines make `EINTR` behave like `EAGAIN`, after which sixteen
ranks bring the domain up inside the worker with no SHMEM errors.

Note that the bootstrap store is a separate target: `src/CMakeLists.txt:190`
builds `aclshmem_bootstrap_config_store` as its own shared library, so editing
this header requires building that target and not just `shmem`.

## One thing left open

The `-1` is layout-sensitive. Adding probes or a canary to
`mem_entity_entry.cpp` — anything that moves `inited_device_id` — makes it stop
appearing, while the canary itself stays intact. That is consistent with the
EINTR failure being its only cause, and also with an independent out-of-bounds
write whose target merely moved. One run on the restored tree decides it:
`bash /tmp/run_onesided.sh` then
`grep "Set device id to be" /root/shmem/log/*.log`. Empty closes it on one root
cause; non-empty means there is a stray writer worth finding, since the next
time it may land somewhere that does not announce itself.
