# M1 WK K-order probe

The isolated library `601c05b229019cb2672c6715babaaf684a49b64c9a412cec5b47c16763b8b59c` compiled with CANN 9.2 but failed its first natural-order device control. The original `F.linear` reproduction passed. The observation-only v2 retained the unchanged library's complete output: columns 0–79 are finite but incorrect; columns 80–159 equal the prefilled NaN bits. All fifteen guard rows are intact. Equal NaN bits do not establish whether a location was written. No K-order numerical conclusion follows from this failed probe.

The [build manifest](source/build/manifest.json) records compiler, SDK headers, commands, source and object hashes. [Binary locations](binary-artifacts.json) retain the object and library outside the repository. The [v2 checker patch](v2-observation/checker.patch) saves all output rows and masks before the unchanged finite and guard assertions.

The [actual M1 dispatch](../captured-wk-m1-source-audit/dispatch.json) and [selected key-98513 instructions](../captured-wk-m1-source-audit/source-anchors.json) define the candidate K512 traversal. Both arms update one FP32 L0C tile with 48 K128 MMADs. They differ only in segment order. The input has one valid row in a physical M16 tile; Mmad and Fixpipe expose only that row. Fifteen following output rows remain guards. This probe does not select a schedule for M2 or any other unmeasured row count.

The frozen checker first requires the exact captured ND `F.linear` call to reproduce all 160 original BF16 outputs. Three interleaved repetitions then require the natural-order arm to reproduce all 160 retained native FP32 accumulators and BF16 values before accepting the shifted arm against all 160 original BF16 values. Both arms also check guard bytes and direct BF16 output against their own FP32 RNE cast. It records no timing.

The [CPU report](cpu-wk-kshift.json) and [four geometry checks](cpu-run.log) passed in the actual Torch 2.12 image, without device mappings or NPU initialization. These checks modeled the matrix layout and did not cover the hardware's automatic M1 GEMV interpretation. The [v2 CPU checks](v2-observation/cpu-run.log) additionally verify preservation of NaN payloads and damaged guard bytes.

The [actual installed header](m1-gemv-source-audit/installed-headers.json) exposes `disableGemv` only for architecture 3510, not A3/dav-2201. The [official GEMV example](m1-gemv-source-audit/mmad_gemv.asc.excerpt.txt) specifies a contiguous A vector for M1: `dstNzC0Stride=1` and one LoadData fractal per 256 BF16 values. The failed probe instead used the M16 NZ gaps. Its first 128 GEMV elements contain only sixteen initialized A values. The [actual M1/M12 instruction comparison](m1-gemv-source-audit/m1-m12-actual-aic-instruction.diff) changes only the four M immediates; block indexing, output address calculations and launch metadata agree. A subsequent candidate therefore changes the A load layout while preserving launch, B layout, Mmad dimensions, K traversal and Fixpipe.

Frozen command on a3-194, after the device admission gate:

```sh
bash /home/shanshan/tilert-megakernel-20261021/wk-m1-kshift-v1/run_kshift.sh 4 captured-wk-m1-kshift-v1
```
