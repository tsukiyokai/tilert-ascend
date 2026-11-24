# M1 GEMV layout and K traversal

Library `8493189945cd922ac2cff9a7653bccd1b5ad03bcf7574ea1278ef91fd44594a8` passes the captured M1/N160/K6144 control and candidate checks on A3 with CANN 9.2 and TorchNPU 2.12. Three natural-order calls reproduce all 160 captured native FP32 accumulators and BF16 values; three shifted-order calls reproduce all 160 original BF16 values. Every call preserves the fifteen output guard rows. The [device report and independent raw audit](../captured-wk-m1-kshift-v3-a3-194/) retain the numerical evidence.

The [kernel patch](source/gemv-layout.patch) changes only A's load layout relative to the failed v1/v2 library. A3's M1 GEMV path consumes a contiguous K vector: `dstNzC0Stride=1`, with one LoadData fractal for the K128 operand. B retains its matrix layout. Launch configuration, Mmad/Fixpipe dimensions, K order, output addresses and synchronization are unchanged. The [earlier source audit](../captured-wk-m1-kshift-build-v1/m1-gemv-source-audit/) records the official example, actual installed headers and matching M1/M12 launch instructions.

The shifted traversal implements the actual key-98513 schedule, `q(i,j)=((j%2 ? 11-i : i)+2*floor(j/2))%12`, followed by four K128 MMADs per segment into one FP32 L0C tile. At column 25 it produces FP32 `0xbeab8000`, exactly the BF16 midpoint; round-to-nearest-even returns `0xbeac`, matching the original operator. This result applies to the captured M1 operands and recorded dispatch. M2 remains outside this evidence.

The [build manifest](source/build/manifest.json) pins compiler, headers, commands, source and objects. The [frozen harness manifest](source/manifest.json) pins the observation-only checker and geometry tests. The actual Torch 2.12 [CPU run](wk-m1-kshift-cpu212-v3/run.log) loads the C ABI and passes six address/output-preservation checks without device mappings or NPU initialization. [Binary locations and hashes](binary-artifacts.json) identify the files retained outside the repository.

The two failed libraries and their reports remain unchanged. Their NaN values alone did not determine whether an output address had been written.
