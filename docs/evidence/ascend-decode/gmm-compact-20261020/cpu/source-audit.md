# GMM A8W4 one-chunk active-table audit

Scope: a3-194 production overlay below; source reads only, no implementation or NPU execution.
Device source root: /home/shanshan/trt-asc-194/experiments/gmm-fastpath-20261018/prod-kernel/overlay/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/grouped_matmul_swiglu_quant_v2/
Abbreviations below: pipeline/mid/pre/post = grouped_matmul_swiglu_quant_v2_a8w4_msd_{pipeline,mid,pre,post}.h.

1. Workspace is available under the proposed guard: loopCount==1, groupListType==1, E=256, K=6144, N=256, 0<M<=256, valid nonnegative counts summing to M. Host base_tiling.cpp:379–402 uses mLimit=floor((64MiB/2)/(K+4N))=4681. pipeline:84–87 makes the second pre bank begin at B=28,760,064 bytes and the first MM bank begin at 2B=57,520,128. pre:132, mid:174–175 and post:373 use only bank 0 for this chunk. Metadata [B,B+2080) is therefore disjoint from live input [0,MK) and MM output. Use the tiling-derived bank address and size, not hard-coded offsets. Do not enable this reuse for loopCount>1: pipeline:138–145 overlaps pre(n+1) and post(n-1), whose bank parity is equal.

2. Publish a stable list e_j in increasing original expert-ID order, c_e>0, with C_j=sum_{e<=e_j} c_e; C_-1=0, A<=min(256,M), C_{A-1}=M. Layout: int64 A plus 256 packed uint64 records ((uint64 C_j)<<32)|e_j; 8+256*8=2056 bytes, padded to 2080 (65x32B). Counts fit the 32-bit fields only under this guard. AIV0 needs separate 2048B input and 2080B output UB buffers: in-place forward packing overwrites an unread count (counts[0]=counts[1]=1, first record stored at input slot 1). pre:95–102 uses 43,264B at K6144; adding both buffers gives 47,392B. No AIC UB operations.

3. Publication must be MTE2->S wait, scalar UB compaction, S->MTE3 wait, MTE3 GM copy, MTE3 completion before buffer reset, then the existing mixed barrier (pipeline:113). Actual CANN 9.2.0 / V100R001C25B114 dav_c220 SyncAllImpl:384–400 performs PIPE_ALL and MTE3/FIX cross-core flags; it does not invalidate this table's scalar DCache. kernel_tensor_impl.h:1307–1311 normally implements GetValue as a plain GM scalar load. Invalidate each consumer's metadata cache lines after the barrier and before its first GetValue (both AIC mid and AIV post); the 64B alignment helper is at 1200–1209. Invalidate header line first, read A, then invalidate the remaining lines covering [0,8+8A); the maximum is offsets 0,64,...,2048 (33 lines), but do not force all 33 for a short active list. Public DataCacheCleanAndInvalid is declared in include/basic_api/kernel_operator_cache_intf.h:46–53; do not substitute an L2 hint. A/B graph replay with equal A but different expert IDs is the minimal stale-cache counterexample. All cores must continue to execute both existing barriers.

4. Keep MMCompute unchanged: production mid:258–280 already uses the true groupIdx for single-tensor weights/scales, and 248–256/274–276 handles tensor lists. For record j set groupIdx=e_j, m=2(C_j-C_{j-1}), xBaseOffset=2*C_{j-1}*K (int4 element units), yBaseOffset=2*C_{j-1}*N (half units); retain blockDimM=ceil(m/baseM), blockDimN and preCount updates at 212–222. This reproduces every nonempty original loop iteration; empty groups never changed preCount. Do not use compact index j as expert ID, or e_j as row offset: IDs [0,255], counts [1,1] require the second weight from expert 255 but row prefix 1.

5. Post must consume the same list throughout, replacing both scans at post:271–293 and 339–355. For a used core starting at row r, select first C_j>r, set true curGroupIdx=e_j and nextUpdateInterVal=C_j-r. Maintain a separate compact cursor j; when the interval reaches zero before another actual row, advance cursor and use the next expert's count. Keep auxiliary-matrix queue/merge order unchanged. Never search/read a record for an unused core or after the last row; no sentinel is needed. For verifier width 1/8/32 with top-8 routing, operator row counts are 8/64/256 and A may reach 8/64/256; a single expert holding all 256 rows still requires m=512 and four baseM=128 blocks. This rules out the one-row-per-active-expert shortcut.

Source identities (SHA256):
pipeline d6aa345d0f1f4bbfdd659d131a7da52b246af36fbd9f2d310a3b6b2d7620eee6
pre b87eda2542700e1800412f9775360552b5c6ae6e38e2afcbd77da677d60f8332
mid 1881d03e65258e846fe9bbfb736f4cbbcd770b84a6730319dd0c9d00abdc4087
post 730e0bac7ce6ea3e085a48225d27093431debcd6f05c542378beefc7e96d3689
host base_tiling.cpp 8e2d570efd12ddda8360fcd515c81fd50444571d47e2ed3200c10f72f7aaef97
CANN sync header 695bb46047f853dd4219c731ee25cf831ea964fde778e03e07df298d3ae72c1e

