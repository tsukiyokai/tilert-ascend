# What a profiled trace can and cannot price on a host-fed device

`profiles/width8-20261019/rank-0/trace_view.json` against the hooked phases of
`phase-synced-20261019`. The round reads 65.64 ms profiled and 53.05 ms hooked,
so 12.59 ms is the instrument. The question is where it lands.

## Only the per-stream compute union survives

Take each phase's window from its `tilert.*` host marker, drop every task whose
name is a synchronisation primitive (`MEM_WAIT_VALUE`, `MEM_WRITE_VALUE`,
`EVENT RECORD/WAIT/_RESET`, `MODEL_EXECUTE`, `MODEL_WAIT_COMPLETE`,
`PROFILER_TRACE_EX`), and union the rest per stream:

| phase | span | compute union | hooked, no profiler | union/hooked |
| --- | --- | --- | --- | --- |
| chain, 6 steps | 12.50 | 6.72 | 10.27 | 0.65 |
| verify glue | 2.96 | 0.10 | 2.08 | 0.05 |
| target | 47.84 | 44.26 | 39.68 | 1.12 |
| commit | 2.34 | 1.15 | 1.02 | 1.13 |
| **round** | **65.64** | **52.56** | **53.05** | **0.99** |

The totals agree to 1%. The phases do not, and the two directions are
different facts: the target and commit run 12% long under the instrument,
while the chain and the glue have wall the device is not computing in.

## Every gap, wait and starvation figure here is the instrument

Three independent checks, because this trace invites the opposite reading:

- `Wait Time` per task: in-graph tasks starve 0.9–1.3 µs each (chain 1.06 ms
  over 1206 tasks, target 9.63 over 7467) while eager bookkeeping starves
  **43.6 µs each** — 33.65 ms over 772 tasks, 75.3% of the capture's 44.72 ms.
  That is 11.2 ms a round against a 12.59 ms instrument.
- The 975 µs gaps between chain replays hold 19 eager tasks worth 50 µs of
  device work and 892 µs of `Wait Time`. Read as host starvation they price at
  over 1 ms of TPOT. `hoist-20261019` had already run that A/B: launches
  313 → 91, priced host 3.30 → 0.96 ms, **TPOT −0.0225 ms, −0.14%**.
- `MEM_WAIT_VALUE` on the target's compute stream: 2393 tasks, 5.93 ms union,
  **zero overlap with compute** — strictly serial, and therefore 15% of the
  phase if believed. But stream 81's compute union is 39.24 ms against the
  hooked 39.68, which leaves no room for it.

So the rule for this trace: price the compute union, discard the gaps. The
profiler's cost is per-task host dispatch, which is exactly what a host-fed
device shows as device idle.

## What is left, and it is not launch count

The chain's 3.55 ms and the glue's 1.98 ms of non-compute wall are measured
against the hooked phases, not the profile, so they are real: **5.53 ms a
round, 1.23 ms of TPOT at acceptance 4.507.** Neither is launch count — the
hoist A/B settles that.

Two candidates, both host wall on the critical path:

- The glue is `target.host_tokens`, whose `torch.cat(tensors).tolist()` is the
  round's only synchronisation, plus `MtpState.resolve`. `resolve` is pure
  Python over eight elements and cannot be milliseconds, so the 1.98 ms is the
  blocking read and whatever the host does before it can issue it.
- The chain is 0.59 ms a step of host wall. `CannMtp.forward` runs six
  validation blocks, three context managers and `CannModel._metadata` per
  step. `_metadata` creates three CPU tensors and does three H2D copies
  **before** its `hoisted is not None` early return, so the hoist removes the
  metadata builder and keeps the copies — which is a mechanism for the hoist
  A/B reading null.

`hoist_metadata` is `false` in the licensed command, so the 58 `aclnnIndex`
calls of the vendor rope gather that `probe.md` left open are not forwards the
hoist misses. The hoist is simply off.

Open: the hoist A/B ran at width 4, where the chain is two steps and the round
is 15.93 ms. At width 8 the same host wall is three times the steps against a
53.05 ms round. One paired arm with `--hoist-metadata` on today's
configuration decides it.
