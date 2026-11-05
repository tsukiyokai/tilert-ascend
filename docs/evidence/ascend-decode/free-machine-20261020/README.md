# First whole-machine measurements since the fleet freed

`a3-84`, 16 dies, `free2-20261020`. Four paired arms at width 32 with the
lookup drafter, three live requests each.

| arm | request TPOT, ms |
| --- | --- |
| base-a, `--lookup` | 11.0052, 10.9691, 10.9937 |
| combine-a, `--plain-moe-combine` | 10.9420, 10.9553, 10.9285 |
| base-b, `--lookup` | 10.9848, 11.0309, 10.9811 |
| combine-b, `--plain-moe-combine` | 11.0452, 10.9897, 10.9737 |

## A new best, from the tree rather than from the flag

10.9285 ms is the lowest request TPOT this project has recorded; the previous
figure was 11.6198. Every arm here is under 11.05, so the gain belongs to the
current tree — pinned staging and the lookup drafter at width 32 — and not to
anything measured today.

## The combine flag's measurement is void, and the cause is on record

Pair A reads 10.989 against 10.942, pair B 10.999 against 11.003: a mean
effect of about 0.02 ms where the kernel-level figure predicts 0.21 ms of
TPOT, which is 1.46 ms a round over 75 layers at acceptance 7.

The flag reached the process — the arm's own log line carries
`--lookup --plain-moe-combine` — and the override is three lines in the tree's
bench. What the arms shared is
`lookup-20261019/tree/static_kernel_compile_outputs/static_kernel_cache`: one
static-kernel cache below the tree, outside the per-arm `cache-*` directories
the launcher makes. `base-a` ran first, compiled, and cached; the combine arms
restored that package, including its `ForeachAddList`.

This is the trap
[w4a8-race-closed-20261019](../w4a8-race-closed-20261019/README.md) already
documents — "the key, built at `:1071-1092`, covers the dumped operator
parameters, the super-kernel option string and
`torch.are_deterministic_algorithms_enabled()` — and nothing else", not the
vendor source and not a monkeypatch.

**And the fix was already written.** `launch_arm.sh` in the repository says
"Static kernel packages are cached below cwd, outside `VLLM_CACHE_ROOT`" and
then `cd "$out"`, which puts the cache in the arm's own output directory. The
10-19 experiment tree's copy still does `cd "$tree"`, so every arm compiles
into the tree and the first one to run decides what the rest replay. Two
launchers, one line apart, md5 `2833c690` against `da8f45f7`.

So the remedy is to ship the current launcher into the tree and redo the
pairing. Not mid-queue: the arms still running change the drafting policy
rather than the compiled graph, so a shared package is harmless to them, and
swapping the launcher underneath them would break the one thing this queue
still has — a consistent basis across its arms.

So nothing is concluded about `--plain-moe-combine` either way. The kernel
measurement that motivated it stands on its own: at matched shapes in one
capture, `Add` is 1.820 µs against `ForeachAddList`'s 21.282, flat in rows.

## What else the queue is for

`bigram-c` onward: the unigram pairing, which measures acceptance and the
expert union at width 32, and the width 8 and 16 arms. Those do not touch the
static-kernel cache question, because they change the drafting policy rather
than the compiled graph.

The one-sided reduce arm failed on the first attempt for a dull reason —
`onesided_reduce` is not in the 10-19 tree, which predates it — and has since
been shipped there, md5-verified, with the engine and bench contract checked
against that tree and the module imported without touching a device. It goes
last in the next queue, because `chain.sh` exits on a failing arm.


## The re-run, with the launcher current

Same tree, same flags, same machine; the one change is the launcher putting
the static-kernel cache in each arm's own output directory. Six requests an
arm, of which the first is plainly a warm-up -- 11.92, 12.07, 11.89, 11.74
against steady states near 11.0 -- so the mean is over the other five.

| arm | steady-state requests, ms | mean |
| --- | --- | --- |
| base-a | 11.0403 11.0360 11.0435 11.0369 11.0638 | 11.0441 |
| combine-a | 11.0108 10.9456 10.9132 10.9261 10.9054 | 10.9402 |
| base-b | 11.1702 11.1130 11.1802 11.1700 11.0966 | 11.1460 |
| combine-b | 10.8538 10.8528 10.8813 10.9050 10.8917 | 10.8769 |

Pair A gives −0.104 ms and pair B −0.269, a mean of −0.19 against the
−0.21 the kernel measurement predicted. The ordering is what makes it: inside
each pair every combine request beats every base request, ten of ten --
combine-a's worst, 11.0108, is below base-a's best, 11.0360, and combine-b's
worst, 10.9050, below base-b's best, 11.0966. 10.8528 ms is the lowest single
request this project has recorded.

What to hold back: base-a and base-b differ by 0.102 ms, so the spread between
nominally identical arms is about the size of pair A's effect, and pair B's is
2.6 times it. The honest figure is 0.1 to 0.3 ms, consistent in sign, with no
overlap inside either pair.

This also closes the diagnosis above by construction rather than by argument.
Nothing about the override changed between the void measurement and this one.
