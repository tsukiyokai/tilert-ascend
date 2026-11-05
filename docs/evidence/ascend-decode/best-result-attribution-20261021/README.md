# The best result is the lookup drafter's, not the automaton's

`lever84-20261020/auto12/rank-0.json` records

    drafter: "lookup"   lookup_prompt_tokens: 8192   width: 12

and the bench writes that field as `"automaton" if args.automaton else
"lookup"`. So the arm did not pass `--automaton`: the 9.3794 ms TPOT belongs to
`LookupDrafter`, the hash-tuple drafter, and not to the suffix automaton.

The arm directories are named `auto8`, `auto12`, `auto16`, `auto32`, which is
where the misattribution came from -- the names were read as the drafter rather
than as the sweep they belonged to.

## What this changes

The automaton has never run in the engine. Its advantage over the chain is a
CPU measurement only: 6.053 against 5.273 accepted at width 32 on this
repository's own sources, and at equal budget it matches the tree to within
half a percent.

So the engine's best configuration is the one the plan has been calling a
baseline, and the drafter the plan credits has never been measured end to end.
Every projection that scaled from 9.3794 while assuming the automaton was
already in it is off by whatever the automaton is worth -- which may be
positive, and is unmeasured.

## And a bug the same reading exposed

`--automaton` and `--tree` select which host drafter to use, but the
host-drafted path was gated on `--lookup` alone. An arm passing only
`--automaton` therefore selected a drafter and left the MTP engine running.
Every one of the thirty-five arms queued today passes only `--automaton`, so
the queue would have measured the engine thirty-five times under names that say
drafter. Fixed: a drafter choice now implies the path.
