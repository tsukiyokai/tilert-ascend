# The MoE matmul is fixed cost, not arithmetic

2026-10-21, a3-84, one die, sixteen experts declared, rows varied inside the
vendor call that already works.

    rows   16 (1 an expert)   146.431 us
    rows   32 (2 an expert)   140.969 us
    rows   96 (6 an expert)   144.651 us
    rows  256 (16 an expert)  150.565 us

Sixteen times the rows for 2.8 percent more time. The operator's cost is
almost entirely independent of the work it does.

## What this changes

The round's accounting prices the MoE matmuls at 12.3 ms, and that figure
stands -- 144.651 us a layer over 78 layers is 11.3 ms. But it is not
arithmetic. Under three percent of it tracks the rows, so it belongs with the
host issue at 6.6 ms and the routing at 12.1 ms: a per-layer fixed tax.

The premise it replaces is one this project has held all along -- that the MoE
line item is computation, reducible only by a faster matmul. Measured, a
faster matmul buys at most three percent of it, which is why the hand-written
kernel's 2.33x disadvantage is beside the point.

## Where the round's 45.88 ms fixed term comes from

It is the sum of these taxes, and they share one mechanism: each is paid once
a layer because each is a separate launch. That is the only account of the
fixed term that the measurements support, and it has one exit -- fewer
launches covering more layers.

So the kernel's value is as a vehicle: not to beat the vendor's matmul, which
it loses to by 2.33x, but to let one launch cover several layers and pay the
tax a handful of times instead of 78.
