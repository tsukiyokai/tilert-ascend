# The cube core cannot read unified buffer

The stage-major restructuring staged the group counts for the cube half in a
`TBuf<TPosition::VECCALC>` and read them back with `GetValue`. On dav-2201 the
cube core's local memory is L1 and the L0s; unified buffer belongs to the
vector cores. Every read returned zero, every expert hit the `continue`, and
no matmul ran.

    before   vendor 0.45117  kernel 0.00000  max difference 0.451172
    after    vendor 0.45117  kernel 0.47461  max difference 0.126953

Reading the counts straight from global memory with a scalar load fixes it:
256 reads a launch against a matmul each, which is why the skip was worth
having -- 256 experts cost 583 us without it and 79.206 with.

## Why it hid

The vector half reads the same counts through unified buffer correctly, so
`Split` produced a correct `stacked` -- 332678 nonzero bytes, plausible INT4
pairs -- and only the cube half silently did nothing. The symptom was
therefore an absent product with healthy inputs, which reads as an arithmetic
fault, and ten hypotheses looked for one. The restructuring itself was sound;
what broke was one buffer's residence.
