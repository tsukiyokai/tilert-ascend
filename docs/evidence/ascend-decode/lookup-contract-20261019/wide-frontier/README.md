# Copy width saturation

Server CPU replay of the actual `LookupDrafter` and `bench_lookup.replay`,
shortest1/longest8. Each width walks its own committed boundaries, includes
pending in context, pads with pending and clips the final bonus. Indexed and
reverse-scan proposals have identical signatures at every width.

| Width | Rounds | Committed/round | Verified rows | Mean round cost allowed relative to M32 |
| --- | --- | --- | --- | --- |
| 32 | 147 | 6.959184 | 4704 | 1 |
| 64 | 144 | 7.104167 | 9216 | <1.020833 |
| 128 | 144 | 7.104167 | 18432 | <1.020833 |
| 256 | 144 | 7.104167 | 36864 | <1.020833 |

Each replay commits1023 tokens after the common first output. M64/128/256 have
identical commit boundaries and maximum single commit61. A wider policy wins
only if its mean round time is below `147/144` times M32's. These data provide
no case for extending the runtime beyond M32.

The three recorded requests are identical copies of one old, teacher-unaccepted
8K+1024 trajectory. They are one policy-screening input, not three independent
prompts. No NPU work, expert-union estimate or production width change.
[Summary and source/input hashes](summary.json), [raw replay](raw.json).
