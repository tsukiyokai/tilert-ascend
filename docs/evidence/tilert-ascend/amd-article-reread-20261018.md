# Rereading the AMD write-up: the premise I had skipped

Source: [469 tok/s/user on Million-Token Agent Sessions: TileRT Takes the Top
Spot on AgentX with AMD Instinct MI355X GPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/tilert-agentx-on-amd-instinct-gpus.html),
fetched in full 2026-10-18. I had been working from an excerpt about the
one-shot push and had the architecture wrong.

## What the design actually is

> TileRT's core approach is to statically unfold the entire model at compile
> time into a persistent **Engine Kernel** on the GPU. Compute, communication,
> and asynchronous I/O all progress continuously within this engine. **There
> is no kernel-by-kernel launch sequence**, and optimization is no longer
> constrained by kernel boundaries: the scope expands from individual kernels
> to the entire forward pass.

Not two or three kernels per layer. One persistent kernel for the whole
forward pass. **This retracts the structure I inferred from the ROCm symbol
table** — 459 kernels across 29 base names, which I read as the execution
granularity. Those are building blocks compiled into the engine, not
scheduling units.

## Why, in their words, and it matches what we measured

> as hardware compute and memory bandwidth continue to increase, the time
> actually spent on computation during a decode forward pass has become
> increasingly small. Other overheads — **kernel launches, cross-kernel
> synchronization, and round trips to global memory** — start to enter the
> critical path.

They call this the execution gap. Our measurements today are exactly those
three terms: an eager dispatch floor of 21.5 µs, an in-graph per-operator
floor of 8 µs ([per-call cost](../ascend-decode/fixed-cost-20261018/README.md)),
HCCL at 63.5 µs of pure rendezvous
([rendezvous](../ascend-decode/reduce-floor-20261018/README.md)), and ~637
cube/vector handoffs through GM per round.

## Three mechanisms, of which I had read one

**Streaming synchronisation replaces the global barrier.** I had missed this
entirely:

> A conventional approach is to use a global barrier: wait until every block
> has finished computing and writing its results before allowing the next
> stage to proceed. TileRT avoids this global alignment. As soon as the
> upstream stage produces part of its output, that data can be handed off to
> the downstream stage. The consumer can wait for and process data
> incrementally as it becomes ready.

So the one-shot push is not a separate trick: it is this same
producer-consumer handshake pointed across ranks. Our profile measures
cube/vector overlap at exactly 0.000000 ms, which is this handshake's absence.

**Prefetch above the kernel boundary.**

> TileRT goes further by eliminating the kernel boundary itself... allowing
> data required by later stages to start moving earlier without waiting for a
> "next kernel" to be scheduled... decides *when* to prefetch, *how far ahead*
> to prefetch, and *what data* to move **across the entire forward pass**.

This resolves today's puzzle: why chained in-graph matmuls cost the same per
call and yet fusing four into one still saves 37%. The gain is not fewer
launches; it is moving the next stage's data across what used to be a kernel
boundary.

**Communication is not a kernel.**

> communication in TileRT is not implemented as a standalone kernel. It is
> integrated directly into the execution pipeline, allowing compute and
> communication to continuously overlap.

## Their numbers

GLM-5.3, 8× MI355X, **TP8**, FP8, MTP K=3, no precision traded: 469 tok/s
single-user, i.e. about 2.13 ms per token, first on AgentX and over 100 tok/s
ahead of the second-place GB300 NVL72 running FP4. From 1K to 1M context the
speed retains roughly two-thirds. Figure 2's long-context curve is MI350X,
which clocks lower, so it is the conservative reference.

We are at 14.247 ms and 68.8 tok/s. Note TP8 against our TP16: fewer ranks
per reduction is a cheaper rendezvous, and at batch one the rendezvous is the
whole cost.

## "Why This Approach Goes Further on AMD" is a statement of premises

Three hardware dependencies, each needing its own check on A3.

A larger per-CU register file, so "more in-flight data can remain resident at
the same occupancy, enabling deeper prefetching".

Partitioned caches with software-controlled visibility scope: "This enables
cross-block data exchange within a partition using lightweight mechanisms
rather than frequently relying on device-wide synchronization... Hardware
partitioning becomes a locality hierarchy that software can actively exploit."
**A3 has this locality hierarchy and we measured it today** — 0.31 µs within
a module against 0.84 µs across one — and HCCL buries it
([wire](../ascend-decode/shmem-install-20261018/README.md#sub-microsecond-and-the-topology-appears)).

Direct ISA control through inline assembly with a disassemblable toolchain.
We have this: bisheng disassembles, and `SET_CROSS_CORE` was visible in
`objdump` of our own probe
([stack selection](../ascend-decode/stack-selection-20261018/README.md)).

> the hardware provides the capabilities; software decides how to schedule and
> compose them.

## What changes for us

PTO-ISA's `static_assert` — A3 has no `Acc→Vec` move, so a cube result
reaches vector only through GM — carries different weight under this reading.
In a model where one persistent kernel runs the whole forward pass, a
mandatory GM round trip between cube and vector is a structural tax on the
central mechanism, not a blemish on one operator. It is the single largest
architectural difference between this machine and the one the article
describes, and it needs its own measurement rather than a verdict.
