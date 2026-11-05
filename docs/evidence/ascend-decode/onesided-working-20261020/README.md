# The one-sided reduce, working, and measured against HCCL

2026-10-20, 187, sixteen ranks, no engine. Twelve attempts; the first eleven
were inside a sixteen-rank arm and said only `aclshmem init returned -4`.

## Correctness

    two ranks:      rank 0: first 3.0 expected 3.0  exact True
    sixteen ranks:  rank 4: first 136.0 expected 136.0  exact True

Every rank pushes `rank + 1` into every peer's symmetric window and reduces in
rank order; 1 + 2 + ... + 16 is 136, element for element. The domain, the
`hybm_init`, the VMM mapping, the push and the ordered local reduction all
hold.

## Cost, both paths in one process

    96 KiB over 16 ranks, 200 calls
      one-sided  32.488 us   exact True
      hccl      201.995 us
      ratio      6.22x
      over 157 reductions a round: 5.10 ms against 31.71 ms

## Three discounts on that ratio

The HCCL figure is 201.995 µs where the module's docstring records 63.5 µs,
three times higher. The likeliest cause is that another tenant held dies 0 and
4 throughout: a collective synchronises all sixteen ranks, so an occupied die
drags the whole chain, where a one-sided push writes its own windows and is
far less exposed. On a clean machine the ratio will shrink; 6.22x is an upper
bound, not a constant.

32.488 µs is the bare call including the Python side, where the 22.3 µs in the
docstring is a kernel figure.

And 157 reductions a round comes from that docstring, not from counting on
this arm.

## What the eleven failures were

Not the SDK and not the rank count -- sixteen ranks come up in the minimal
harness while a tenant is on the machine. Two things in my own code. `_bind`
ran after `_load`, so the backend's constructor read the device before one was
set and `inited_device_id` stayed -1, which `AclrtSetDevice(-1)` then reported
as 107001 and `setup_heap` returned as -4. And `_bind` itself parsed the
device index as `device.index if hasattr(device, "index") else ...`, which is
true for the string "npu:0" because `str` has an `index` method, so it handed
back the method.

Neither was visible from an arm. The arm costs twenty minutes, buries its
tracebacks in `launch.log`, and reports one error code for several causes.
Two ranks and no engine cost a minute and print the traceback.
