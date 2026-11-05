# Shared-attention terminal completion

The source contract supports local completion at the final exit of the independent shared-attention kernel. It does not support replacing the callable completion contract with local completion. No kernel was changed or executed for this audit.

The parent is runtime212 library `64108f6f1eefe286c999dd11fa52bd1cef90535c0888f89915c209cbcc5c8a7c`, kernel object `c9875a9ce03ff30fbe362cc05a0b9d951dd33a34651ce0da5c9d7bcc527071ec`. All 26 source hashes in the runtime212 manifest were verified. `audit.json` records installed CANN 9.2 excerpts and hashes separately from the local documentation checkouts.

## Completion obligations

1. Every earlier producer/consumer join remains. The final tail join before dequantization remains, so all output-projection INT32 inputs are available.
2. Each final BF16 output element has one AIV owner. For task `t`, columns are `256*(t%24)..256*(t%24+1)`, rows are `8*(t/24)..min(rows,8*(t/24+1))`; worker `v` owns `t=v+2*blocks*k`. There is no inter-worker reduction or reader after these writes.
3. All local reads and writes complete before the core exits. Keep explicit `PipeBarrier<PIPE_ALL>(); DataSyncBarrier<MemDsbT::DDR>();` and the ordinary `TPipe` destructor. Do not substitute DDR alone for pipeline completion.
4. No GM, UB, flag or pipe resource is reused by subsequent code in the same kernel. Every block follows the terminal path. The normal, non-superkernel destructor remains enabled.
5. A consumer is a later task on the same actual stream, or waits for an event recorded after this kernel. Host reads require synchronization. Cross-stream submission without an event and a device callable return provide no such completion edge.

`mla_tail_native.h:409-419` performs the final MTE3 output copy and waits on MTE3_MTE2 before another task can reuse its UB. Lines 470-476 retain the prior mixed join, execute `DequantizeOutputRows`, and publish. `attention_full_native.h:32` is the only following operation; the global entry returns immediately after `FullForward` at `attention_full_kernel.cpp:25-28`.

## Destructor and cache boundary

Installed `kernel_tpipe_impl.h:80-85` calls `Destroy`. Its `DestroyWithoutPipeAll` at 541-576 waits for outstanding local queue events and AIC M_MTE1 events 0, 1 and 2. `Destroy` at 579-588 adds PIPE_ALL outside superkernel builds. It does not join other cores. The free-buffer loop is the same loop executed by `Reset` at 591-600; deleting the final Reset defers those waits to destruction. Reset then initializes local state and clears resource tables. There is no later allocation requiring those tables at a terminal return.

Installed `SyncAllImpl` at `dav_c220/kernel_operator_sync_impl.h:363-403` adds PIPE_ALL and mixed-core flag handshakes, not a DCache flush. Current final outputs use DataCopy/MTE3, not scalar GM stores. The local DataCache API documentation distinguishes coherent DMA accesses from scalar DCache accesses. This argument must be rechecked if scalar GM producers, cached polling, superkernel modes or a post-return device consumer are introduced.

The runtime programming-model contract says that a later kernel on the same stream starts after the preceding kernel completes. Installed `acl_rt.h:2807-2817` documents waiting for all stream tasks, and 2893-2905 documents cross-stream event waits. This does not mean that an individual AIC return is completion of the mixed task. The local official mixed-kernel example `cross_core_set_wait_flag.asc:35-47` permits AICs to return while AIV work remains and ends with local PIPE_ALL; this is supporting source evidence, not an additional device test.

## Callable counterexample

At M12 with 24 mixed groups, the 48 AIVs own 48 independent output tiles. After the retained projection join, an AIC can return while an AIV is still producing one row's columns. If the callable's next phase starts a row norm, TP publication or scratch reuse immediately, that AIC or another AIV can read old output or overwrite inputs before the delayed producer finishes. Per-core PIPE_ALL cannot establish the missing cross-core edge. Keep the default callable `EndPhase` or provide a separate proved consumer-specific protocol.

## Narrow candidate and falsifier

In an independent frozen-source variant, make `FullForward` a `template<bool Terminal=false>`; use local PIPE_ALL then DDR only in the Terminal branch, and keep the original EndPhase otherwise. Only the global `whole_layer_full_attention` entry calls `FullForward<true>`. Earlier joins, arithmetic, packing, metadata and host validation remain unchanged.

The first device gate should queue the attention kernel and a consuming clone or norm in one graph, without a host synchronization between them. Poison output/latent before replay, vary rows and cache updates, and check complete original caches and every retained boundary. Include idle AIVs at M1 and multi-task owners at M32, and preserve buffer owners through the downstream consumer. A mismatch or hang falsifies this candidate; a passed numerical gate alone does not establish latency benefit. Measure the full kernel against the parent on both common-address arenas.
