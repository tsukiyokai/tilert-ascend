# M12-only indexed WK integration

Library `5f8868bf96efcfd5a53f17890ce8b619287d4463386a2f020c80546332edfabc`
adds the measured M12 K traversal to the frozen Torch212 indexed-attention
parent `e957650c`. The new branch uses the existing five KN32 weight banks,
twelve K512 segments, four K128 updates per segment and one L0C accumulator.
Logical M12 is distinct from the physical M16 tile; output padding is untouched.
The original M32 branch and all other front-end operations remain unchanged.
No additional GM workspace is allocated.

[manifest.json](manifest.json) records the real CANN build, unchanged host and
tiling objects, 59 source identities and successful Torch212 schema co-load
without NPU initialization. [front.patch](front.patch) is the complete front-end
change; [native.h](native.h) contains the new branch. CPU checks cover all
operand coordinates, unique valid output writes, capacity padding and exact
recovery of the frozen parent source after removing the branch.

The isolated natural/shifted arithmetic gates are recorded
[here](../captured-wk-m12-kshift-v1-a3-194/summary.json). The integrated library
then passed [the original Decoder layers 6/7](../vendor-hook-decoder212-wk-m12-int8-wide-v1-a3-194/summary.json):
M1/2/12/32, eight changing inputs per width and 816 raw-byte boundaries.
Both arms use the official FFN and unchanged physical weight bytes. This gate
uses explicit local TP cut inputs and executes no TP16 numerical reduction.
