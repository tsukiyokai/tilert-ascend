# K128 original Decoder FFN gate

On one idle A3 die, the complete K128 FFN reproduces the original Decoder post-attention norm and MLP at M1/2/12/32 for two synthetic weight sets and eight changing inputs. All 704 byte comparisons pass, including 256 immediate graph-consumer checks; all 72 complete 200-byte Down records match the candidate. Four original NZ weight allocations retain their addresses and bytes.

The candidate runs through native graph replay. Sixteen constructor processes establish TP16 shapes; only rank 0 initializes NPU. Attention, TP collectives and full-model generation are outside this gate. `summary.json` is independently reconstructed by `audit.py` from the retained reports. `fixture-inputs.sha256` pins all 789 staged files.

CPU preflight v1 did not start because the requested image was absent on a3-194. V2 uses the pinned production image, mounts no device, passes four scale contracts and co-loads the actual candidate in Torch 2.12 with NPU uninitialized. Both command/log pairs are retained.
