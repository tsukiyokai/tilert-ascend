# FIA invalid-row correction

A3-194 die15, CANN9.2/B114,512MiB Torch quota, functional replay only.
Eight frozen local-attention inputs from layers66–73 restore the same
Q, RoPE Q,8K prefix,32 new KV rows, selection mask and capacity9344.
Only `inner_precise` changes from0 to2; no weights or process group load.

All524,288 BF16 output values are bit identical between0 and2. All16
saved SFA/FIA0 controls reproduce bitwise and input hashes remain unchanged.
FIA/SFA layer NRMSE remains0.000973759–0.001595521; max absolute difference
is0.0078125. Peak allocated/reserved is116,614,656/130,023,424 bytes.
[Result](device-01/result.json), [CPU binding](cpu-ready.json),
[preflight](preflight.json), [exit](launch-exit.json), [replay](replay.py).

Close this flag-only route for these inputs. The fixtures precede the failed
full shared-FIA run, so this does not prove every layer is unaffected. No
full-model rerun or performance claim follows from this null result.
