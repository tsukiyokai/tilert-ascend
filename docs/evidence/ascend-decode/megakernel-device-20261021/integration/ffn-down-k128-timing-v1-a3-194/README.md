# K128 versus N1024/K64 paired FFN timing

On a3-194 physical die 4, K128 saves 7.339 and 5.872 microseconds in two independent owner arenas at M12/24 with 65 active experts. All 60 alternating pairs favor K128. Candidate/control medians are 203.950/211.309 and 203.187/209.089 microseconds. Each graph contains 16 calls using 16 weight banks; each event contains 30 replays.

All 19,200 numerical, 360 event-tail, 696 active-table and 20 complete 200-byte Down-tiling checks pass. Both arms retain 32 common scratch addresses and the same weights; only body slot 7 retains its own immutable prepared tiling. The original Decoder random-width and hot-expert gates passed before this run.

`summarize.py` verifies the retained result, recomputes paired differences, medians and win counts, and checks every complete tiling against its frozen oracle. `handoff.json` pins the benchmark and CPU evidence. This is a synthetic standalone FFN comparison. Whole-layer and TPOT gains require their own matched measurements.
