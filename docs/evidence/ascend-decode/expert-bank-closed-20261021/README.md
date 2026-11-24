# The expert bank is read whole, and compacting it costs 115x what it saves

2026-10-21, a3-84, one die, 256 declared experts, 96 rows.

Cost does not track the number of experts holding rows:

     8 experts busy    6.0 MB touched   112.850 us
    24 experts busy   18.0 MB touched   100.290 us
    70 experts busy   52.5 MB touched   132.926 us
   128 experts busy   96.0 MB touched    99.995 us
   256 experts busy  192.0 MB touched   111.524 us

A 32-fold range in bytes gives a 33 percent spread with no ordering. So the
operator does not read by the group list, and a narrower draft -- which touches
fewer distinct experts -- saves nothing in the MoE. That removes a saving the
adaptive width was assumed to carry beyond its verification rows.

Compacting the touched experts into a contiguous bank, 70 of 256:

    whole bank        221.089 us
    compacted bank    188.956 us
    the block copy   3688.184 us

The operator is 32 us faster on a bank a quarter the size -- 15 percent for a
3.7-fold reduction in bytes, which again says it is not reading by bytes -- and
building that bank costs 3688 us, 115 times the saving. A 192 MB fractal-NZ
copy is a bandwidth operation in its own right.

Closed.

## The pattern across three closures today

Attention reads its whole cache and masks afterwards; pre-gathering is 35
percent slower because scattered reads run at 115 GB/s against 380 contiguous.
The grouped matmul reads its whole expert bank; compacting costs 115 times the
saving. In both cases the operator's apparent waste is the cheaper choice, and
in both cases this project's own evidence had recorded the waste as an
opportunity.
