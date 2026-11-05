# TileRT's own ROCm GLM-5.2 build — measured fusion structure

Static reading of a prior extraction of `libtilert_glm52_rocm.so` (TileRT 0.1.6.post3, gfx950 /
MI350-MI355X), at
`/home/shanshan/note/proj/ai-infra/llm/infer/ultra-low-latency/codes/tile-ai/TileRT/bin/tilert-glm52-rocm-20261015/`.
That directory's `README.md` carries the provenance: 22 uncompressed Clang offload bundles out of
`.hip_fatbin`, single target `hipv4-amdgcn-amd-amdhsa--gfx950`, 459 kernels in 30 families, code
object v6, nothing stripped. Library sha256
`4f9b74ef16e479042616cb43c0389078bd5bd7a3069f4ef63b7d186d8e1f98de`. All line numbers below are
into the files of that directory as produced by its `extract.sh`. Nothing here was run on a GPU.

## Summary

One kernel per MoE layer. `pure_mla_moe_layer_kernel` is the whole layer, norm through MLA, O
projection, TP all-reduce, router, routed experts, shared expert, SwiGLU, second all-reduce and
bf16 epilogue, in one launch: 6087-7864 instructions, 30-34 workgroup barriers, grid 256 x 512 on
a 256-CU part, 109-150 KB LDS, so one workgroup per CU. It is `pure_mla_allreduce_kernel`
concatenated with `fused_moe_allreduce_v4_kernel` (barrier-segment alignment is monotone, 9 of its
32 segments have an identical opcode multiset), and its LDS equals the attention kernel's exactly,
so the MoE half costs no extra on-chip state. Both halves also ship separately, giving a 2-kernel
fallback. No megakernel: the grid is device-sized but the kernel exits per layer, with no outer
loop, no work queue and no grid-wide barrier. Cross-GPU waits are in-kernel spins on a 16-bit
epoch tag packed into every transferred word. No W4 anywhere; weights are uint8 FP8 with f32
scales, dequantised to bf16 one instruction before the MFMA.

## 1. Fusion granularity

Reachability closure over direct calls in the stripped host code, from the single-step function
`0x772c80` that the graph capture wraps, gives 20 kernel families that a decode round can launch.
The number after each is the count of static launch sites, which is the number of template
specialisations folded into one source-level launch, not the number of launches:

```
# reachable from 772c80 : 287 functions
   48 launch sites  tilert::ops::glm_5_2::pure_mla_moe_layer::pure_mla_moe_layer_kernel
   24 launch sites  tilert::ops::glm_5_2::pure_mla_allreduce::pure_mla_allreduce_kernel
   12 launch sites  tilert::ops::glm_5_2::rmsnorm_head_proj::rmsnorm_head_proj_kernel
   10 launch sites  tilert::ops::glm_5_2::eh_proj_allreduce::eh_proj_allreduce_kernel
    6 launch sites  tilert::ops::glm_5_2::fused_moe_allreduce_v4::fused_moe_allreduce_v4_kernel
    6 launch sites  tilert::ops::glm_5_2::fused_moe_allreduce_w8a8_v4::fused_moe_allreduce_w8a8_v4_kernel
    6 launch sites  tilert::ops::glm_5_2::padded_allreduce_add::padded_allreduce_add_kernel
    6 launch sites  tilert::ops::glm_5_2::sparse_select_s0123::sparse_select_s0123_kernel
    4 launch sites  tilert::ops::glm_5_2::sparse_index::topk_select_kernel
    3 launch sites  tilert::ops::glm_5_2::sparse_index::sparse_index_ts16_kernel
    3 launch sites  tilert::ops::glm_5_2::sparse_index::sparse_index_fp8_kernel
    3 launch sites  tilert::ops::glm_5_2::upgate_silu::rmsnorm_upgate_silu_kernel
    2 launch sites  tilert::ops::glm_5_2::top_p::top_p_sample_kernel
    1 launch sites  tilert::lib::glm_5_2::(anonymous namespace)::glm52_prefill_next_kernel(int*, int const*)
    1 launch sites  tilert::ops::glm_5_2::broadcast_selected_token_ids::broadcast_selected_token_ids_kernel(...)
    1 launch sites  tilert::ops::glm_5_2::llm_preprocess::llm_preprocess_kernel(...)
    1 launch sites  tilert::lib::glm_5_2::(anonymous namespace)::simulate_accept_len_kernel(...)
    1 launch sites  tilert::ops::glm_5_2::mtp_verify_gather::mtp_step_tail_kernel(...)
    1 launch sites  tilert::ops::glm_5_2::mtp_ar_tail::ar_tail_no_mtp_kernel(...)
    1 launch sites  tilert::ops::glm_5_2::top_p::top_p_local_scores_kernel(...)

## kernel families with use sites but UNREACHABLE
   72 launch sites  tilert::ops::cell::gemv_w8a16_cdna4::gemv_w8a16_mfma_cdna4_kernel
   24 launch sites  tilert::ops::glm_5_2::unprojo_allreduce::unprojo_allreduce_kernel
    3 launch sites  tilert::ops::cell::gemv_w8a16::gemv_w8a16_mfma_kernel
    3 launch sites  tilert::ops::cell::rmsnorm_projx_wqkv_w8a16::rmsnorm_projx_wqkv_w8a16_mfma_kernel
    3 launch sites  tilert::lib::misc::rmsnorm_raw::rmsnorm_raw_reg_kernel
    3 launch sites  tilert::ops::cell::gemv_w8a16_m4::gemv_w8a16_m4_cdna4_kernel
    1 launch sites  tilert::ops::glm_5_2::flash_sparse_mla::xfer_send_indices_kernel(...)
    1 launch sites  tilert::ops::glm_5_2::top1_allreduce::top1_allreduce_kernel(...)
```

Reproduced from `host/kernel-uses.tsv` and `host/disasm.s`; the unreachable set are wrapper
functions with no caller. The whole per-primitive `tilert::ops::cell::gemv_w8a16*` GEMV library is
dead code in this build. TileRT ships the one-kernel-per-primitive path and does not use it.

The per-layer sequence on the host is `graph-call-trees.txt` lines 6-32. Three call sites matter:

```
# host/graph-call-trees.txt:6-32 (elided to the layer stack)
  0x77330b call 0x778c40
      0x77a127 call 0x7a5480
        0x7a6f23 <<<>>> tilert::ops::glm_5_2::sparse_select_s0123::sparse_select_s0123_kernel x6
      0x77aa68 call 0x79a280
        0x79a925 hipLaunchKernel tilert::ops::glm_5_2::sparse_index::sparse_index_fp8_kernel x3
      0x77ab10 call 0x798b30
        0x798e67 hipLaunchKernel tilert::ops::glm_5_2::sparse_index::sparse_index_ts16_kernel x3
      0x77ab4f call 0x79b1b0
        0x79b88f hipLaunchKernel tilert::ops::glm_5_2::sparse_index::topk_select_kernel x4
      0x77ab8b call 0x785080
        0x785489 hipLaunchKernel ...broadcast_selected_token_ids_kernel x1
      0x77ac51 call 0x785890
        0x785f61 hipLaunchKernel ...padded_allreduce_add_kernel x6
      0x77c109 call 0x78ca00
        0x78da26 call 0x78fb90
          0x78fc4d hipLaunchKernel ...pure_mla_moe_layer::pure_mla_moe_layer_kernel x24
        0x78da6a call 0x78bdb0
          0x78be6d hipLaunchKernel ...pure_mla_moe_layer::pure_mla_moe_layer_kernel x24
      0x77c589 call 0x75d460
        0x75d751 hipLaunchKernel ...pure_mla_allreduce::pure_mla_allreduce_kernel x24
      0x77cdb3 call 0x766d20
        0x76839a <<<>>> ...fused_moe_allreduce_w8a8_v4::fused_moe_allreduce_w8a8_v4_kernel x6
      0x77ce3e call 0x75ffb0
        0x76162a <<<>>> ...fused_moe_allreduce_v4::fused_moe_allreduce_v4_kernel x6
      0x77cf2b call 0x7ad170
        0x7ad764 hipLaunchKernel ...upgate_silu::rmsnorm_upgate_silu_kernel x3
```

The `xN` counts are folded template specialisations, not launches. For `rmsnorm_upgate_silu` the
three are the three speculation widths at one call site, so nothing here corroborates the
3-dense/75-MoE layer split:

```
# host/kernel-uses.tsv, all three reachable rmsnorm_upgate_silu sites
0x7ad764	0x7ad170	hipLaunchKernel	void ...rmsnorm_upgate_silu_kernel<1, 1, 8>(__hip_bfloat16 const*, float const*, unsigned char const*, float const*, __hip_bfloat16*)
0x7ad827	0x7ad170	hipLaunchKernel	void ...rmsnorm_upgate_silu_kernel<2, 1, 8>(__hip_bfloat16 const*, float const*, unsigned char const*, float const*, __hip_bfloat16*)
0x7ad8ea	0x7ad170	hipLaunchKernel	void ...rmsnorm_upgate_silu_kernel<4, 1, 8>(__hip_bfloat16 const*, float const*, unsigned char const*, float const*, __hip_bfloat16*)
```

Kernel count per layer, measured:

- MoE layer, fused path: 1 layer-body kernel (`pure_mla_moe_layer_kernel`) plus the KV index
  stage. The index stage is 1 kernel if `sparse_select_s0123_kernel` is taken (the name states
  four sub-stages fused), otherwise 3 to 5 (`sparse_index_fp8` or `sparse_index_ts16`, then
  `topk_select`, then `broadcast_selected_token_ids` and `padded_allreduce_add`).
- MoE layer, split path: 2 layer-body kernels (`pure_mla_allreduce_kernel` then
  `fused_moe_allreduce_v4_kernel` or `fused_moe_allreduce_w8a8_v4_kernel`) plus the same index
  stage.
- Dense MLP: the only reachable kernel whose name and signature describe a non-expert MLP front
  half is `rmsnorm_upgate_silu_kernel<tokens, 1, 8>` — one bf16 input, one f32 vector, one uint8
  weight, one f32 scale, one bf16 output, grid 192 x 512. Whether it serves the dense layers is
  not established by the binary.

The two `pure_mla_moe_layer` wrapper calls at `0x78da26` and `0x78da6a` are a runtime binary
choice, not two launches:

```
# host/disasm.s 0x78da01-0x78da7a
  78da01:	test   r13,r13
  78da04:	je     78da4a <...>
  78da26:	call   78fb90 <...>          ; W8A16 MoE path, code object 13
  78da4a:	sub    rsp,0x8
  78da6a:	call   78bdb0 <...>          ; W8A8  MoE path, code object 14
```

The decisive fusion evidence is that the fused kernel is the concatenation of the two split
kernels. Segmenting each kernel on `s_barrier` and matching segments by opcode-multiset Jaccard:

```
pure_mla_moe_layer: 32 segments   pure_mla_allreduce: 17 segments
seg(A)  n     best match in B (jaccard on opcode multiset)
  0    400    -> B seg 0     0.993 ~
  1    140    -> B seg 1     1.000 EXACT
  2    165    -> B seg 2     1.000 EXACT
  3    484    -> B seg 3     1.000 EXACT
  4    536    -> B seg 4     0.996 ~
  5    141    -> B seg 5     1.000 EXACT
  6    108    -> B seg 6     1.000 EXACT
  7    192    -> B seg 7     1.000 EXACT
  8    100    -> B seg 8     0.941 ~
  9    32     -> B seg 9     1.000 EXACT
  10   78     -> B seg 10    1.000 EXACT
  11   348    -> B seg 11    0.966 ~
  12   370    -> B seg 12    0.981 ~
  13   48     -> B seg 13    0.958 ~
  14   42     -> B seg 14    0.977 ~
  15   41     -> B seg 15    0.976 ~
  16   191    -> B seg 16    0.668 .
```

and against the MoE kernel, continuing from the seam:

```
pure_mla_moe_layer: 32 segments   fused_moe_allreduce_v4: 14 segments
  18   68     -> B seg 0     0.622 .
  19   188    -> B seg 1     0.829 .
  20   317    -> B seg 2     0.792 .
  21   269    -> B seg 3     0.826 .
  22   150    -> B seg 4     0.831 .
  23   245    -> B seg 5     0.909 ~
  24   507    -> B seg 6     0.861 ~
  25   196    -> B seg 7     0.802 .
  26   204    -> B seg 8     0.962 ~
  27   217    -> B seg 9     0.941 ~
  28   49     -> B seg 10    0.833 .
  29   43     -> B seg 11    0.826 .
  30   38     -> B seg 12    1.000 EXACT
  31   111    -> B seg 13    0.494
```

Both alignments are strictly monotone. Barrier arithmetic corroborates: 16 barriers in
`pure_mla_allreduce` plus 13 in `fused_moe_allreduce_v4` is 29, and the fused kernel has 31, so
fusing added exactly two workgroup barriers at the seam where two separate launches would have
supplied an implicit device-wide one.

```
# barriers / static instruction counts, kernels.tsv + disasm.gfx950.s
pure_mla_moe_layer_kernelILi1ELi8ELNS_4cell4xfer5ProtoE0ELi0ELb0ELb0EL   barriers=31 instrs=6090
pure_mla_allreduce_kernelILi1ELi8ELNS_4cell4xfer5ProtoE0ELb0ELb0EEE      barriers=16 instrs=3342
fused_moe_allreduce_v4_kernelILNS_4cell4xfer5ProtoE0ELi1ELi12ELb0EEE     barriers=13 instrs=2837
```

## 2. Persistent or megakernel pattern

The grid is sized to the device, not the problem, and occupancy is forced to one workgroup per
CU, but the kernel is not persistent across layers.

Launch configuration, read off the `__hipPushCallConfiguration` immediately preceding the layer
launch. `rdi` carries `gridDim.x` in its low dword and `gridDim.y` in its high dword, `rdx`
carries `blockDim` the same way:

```
# host/disasm.s 0x78fbdd-0x78fc4d
  78fbdd:	movabs rdi,0x100000100        ; gridDim  = (256, 1, 1)
  78fbe7:	lea    rdx,[rdi+0x100]        ; blockDim = (512, 1, 1)
  78fbee:	mov    esi,0x1
  78fbf3:	mov    ecx,0x1
  78fbf8:	xor    r8d,r8d                ; dynamic shared = 0
  78fbfb:	call   7b40e0 <__hipPushCallConfiguration@plt>
  78fc0d:	mov    edx,0x230              ; 560-byte Args blob memcpy'd to the kernarg buffer
  78fc12:	mov    rdi,r14
  78fc15:	mov    rsi,rbx
  78fc18:	call   7b3d80 <memcpy@plt>
  78fc4d:	lea    rdi,[rip+0x286e4]        # 7b8338 <...>
```

`0x230` is 560, matching `.kernarg_segment_size: 560`, and `0x7b8338` is the kernel handle:

```
# host/kernel-uses.tsv:169
0x78fc4d	0x78fb90	hipLaunchKernel	void tilert::ops::glm_5_2::pure_mla_moe_layer::pure_mla_moe_layer_kernel<1, 8, (tilert::cell::xfer::Proto)1, 1, false, false, false>(tilert::ops::glm_5_2::pure_mla_moe_layer::Args)
```

256 blocks of 512 threads is one workgroup per CU on a 256-CU gfx950, and the LDS request makes
that mandatory rather than incidental: 109072 B of a 160 KiB per-CU budget admits one workgroup,
150592 B at the widest speculation setting admits one with 9 KiB to spare. Other kernels on the
same convention: `pure_mla_allreduce`, `rmsnorm_head_proj`, `sparse_select_s0123`,
`padded_allreduce_add` and `unprojo_allreduce` all launch with `movabs rdi,0x100000100`;
`rmsnorm_upgate_silu` uses `movabs rdi,0x1000000c0`, grid 192; `eh_proj_allreduce` grid 128. The
two `fused_moe_allreduce` wrappers receive the packed dim3 in a register
(`mov rbp,rsi` at `0x76006b`, then `mov rdi,rbp` / `lea rdx,[rbp+0x100]`), so their grid is
runtime-determined, with `blockDim.x` tied to `gridDim.x + 256` by construction.

Workgroups are then split by role on `blockIdx.x % 8` and a linear work id is range-checked. This
prologue is the clearest statement of the design:

```
# disasm.gfx950.s:323064-323111, kernel +0x000 .. +0x0e4
0000000000007e00 <_ZN6tilert3ops7glm_5_218pure_mla_moe_layer25pure_mla_moe_layer_kernelILi1ELi8ELNS_4cell4xfer5ProtoE0ELi0ELb0ELb0ELb0EEEvNS2_4ArgsE>:
	s_ashr_i32 s3, s2, 31                                      // 000000007E00: 90039F02
	s_load_dwordx2 s[76:77], s[0:1], 0x150                     // 000000007E04: C0061300 00000150
	s_load_dwordx4 s[28:31], s[0:1], 0x140                     // 000000007E0C: C00A0700 00000140
	s_lshr_b32 s4, s3, 29                                      // 000000007E14: 8F049D03
	s_add_i32 s4, s2, s4                                       // 000000007E18: 81040402
	s_ashr_i32 s80, s4, 3                                      // 000000007E1C: 90508304   ; s80 = blockIdx.x / 8
	s_and_b32 s4, s4, -8                                       // 000000007E20: 8604C804
	s_sub_i32 s34, s2, s4                                      // 000000007E24: 81A20402   ; s34 = blockIdx.x % 8
	s_cmp_lt_i32 s34, 2                                        // 000000007E28: BF048222   ; role A: 2 of every 8
	s_cbranch_scc0 6                                           // 000000007E2C: BF840006 <SYM+0x48>
	s_cmpk_gt_i32 s2, 0x7f                                     // 000000007E30: B202007F   ; grid-half test
	s_cselect_b32 s4, -16, 16                                  // 000000007E34: 850490D0   ; swap the two halves
	s_add_i32 s4, s4, s80                                      // 000000007E38: 81045004
	s_lshl_b32 s6, s4, 1                                       // 000000007E3C: 8E068104
	s_mov_b64 s[4:5], 0                                        // 000000007E40: BE840180
	s_branch 1                                                 // 000000007E44: BF820001 <SYM+0x4c>
	s_mov_b64 s[4:5], -1                                       // 000000007E48: BE8401C1
	s_load_dwordx8 s[52:59], s[0:1], 0x48                      // 000000007E4C: C00E0D00 00000048
	s_load_dwordx4 s[68:71], s[0:1], 0x68                      // 000000007E54: C00A1100 00000068
	s_load_dwordx16 s[36:51], s[0:1], 0xd8                     // 000000007E5C: C0120900 000000D8
	s_load_dwordx8 s[60:67], s[0:1], 0x118                     // 000000007E64: C00E0F00 00000118
	s_load_dword s35, s[0:1], 0x188                            // 000000007E6C: C00208C0 00000188
	s_load_dwordx16 s[12:27], s[0:1], 0x190                    // 000000007E74: C0120300 00000190
	s_andn2_b64 vcc, exec, s[4:5]                              // 000000007E7C: 89EA047E
	s_movk_i32 s4, 0xc0                                        // 000000007E80: B00400C0   ; role A base = 192
	s_cbranch_vccnz 2                                          // 000000007E84: BF870002 <SYM+0x90>
	s_mul_i32 s6, s80, 6                                       // 000000007E88: 92068650   ; role B: 6 of every 8
	s_mov_b32 s4, -2                                           // 000000007E8C: BE8400C2
	s_waitcnt lgkmcnt(0)                                       // 000000007E90: BF8CC07F
	s_mul_hi_u32 s5, s35, 0x80008001                           // 000000007E94: 9605FF23 80008001
	s_lshr_b32 s5, s5, 15                                      // 000000007E9C: 8F058F05
	s_mul_i32 s5, s5, 0xffff                                   // 000000007EA0: 9205FF05 0000FFFF
	s_sub_i32 s33, s35, s5                                     // 000000007EA8: 81A10523   ; s33 = step mod 65535
	s_add_i32 s4, s4, s34                                      // 000000007EAC: 81042204
	s_add_i32 s33, s33, 1                                      // 000000007EB0: 81218121   ; epoch = that + 1
	s_add_i32 s4, s4, s6                                       // 000000007EB4: 81040604
	s_cmpk_lt_i32 s4, 0xa5                                     // 000000007EB8: B30400A5   ; work id < 165 ?
	s_cselect_b32 s81, s4, -1                                  // 000000007EBC: 8551C104
	v_and_b32_e32 v66, 63, v0                                  // 000000007EC0: 268400BF   ; lane
	v_lshrrev_b32_e32 v67, 6, v0                               // 000000007EC4: 20860086   ; wave 0..7
	v_bfe_u32 v92, v0, 4, 2                                    // 000000007EC8: D1C8005C 02090900
	v_and_b32_e32 v93, 15, v0                                  // 000000007ED0: 26BA008F
	s_cmpk_lt_i32 s81, 0xa4                                    // 000000007ED4: B35100A4   ; stage guard < 164
	s_mov_b64 s[4:5], -1                                       // 000000007ED8: BE8401C1
	s_cbranch_scc0 1417                                        // 000000007EDC: BF840589 <SYM+0x1704>
	s_cmp_lt_i32 s81, 0                                        // 000000007EE0: BF048051
	s_mov_b32 s5, 0                                            // 000000007EE4: BE850080
```

Role B (`blockIdx.x % 8` in 2..7, so 192 of 256 workgroups) gets work id
`6*(blockIdx.x/8) + (blockIdx.x%8 - 2)`, range 0..191, clipped to -1 above 164. Role A
(`blockIdx.x % 8` in 0..1, 64 workgroups) gets `192 + (blockIdx.x%8) + 2*(blockIdx.x/8 ± 16)`,
range 192..255, which always clips to -1, so role A sits out this stage; the `± 16` swap of the
two 128-block grid halves only matters to a later stage. The bound 164/165 is not explained by
any GLM-5.2 dimension I can identify (see Gaps).

Nothing in the layer kernel is a grid-wide barrier. Per-instance sync census across all 48
specialisations: `s_barrier` 30-34, `global_atomic*` 0, `s_sleep` 0, `buffer_wbl2` exactly 1.
What does exist is a bounded consumer spin on data arrival. Inside the KV-staging segment there
are nine such loops, all the same shape:

```
# disasm.gfx950.s:324374-324403, kernel +0x1574 .. +0x15e0
	ds_read_b32 v2, v15                                        ; owner tag out of LDS
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s8, v2
	s_cmp_gt_i32 s8, -1
	s_cselect_b64 s[6:7], -1, 0
	s_cmp_eq_u32 s8, s81                                       ; tag == my work id ?
	s_cselect_b64 s[8:9], -1, 0
	s_and_b64 s[6:7], s[6:7], s[8:9]
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz 27                                         ; not mine, skip
	s_nop 0
	global_load_dwordx2 v[2:3], v[8:9], off sc1
	global_load_dwordx2 v[10:11], v[8:9], off offset:8 sc1
	global_load_dwordx2 v[4:5], v[8:9], off offset:16 sc1
	global_load_dwordx2 v[12:13], v[8:9], off offset:24 sc1
	s_waitcnt vmcnt(3)
	v_cmp_ne_u32_e32 vcc, s35, v3                              ; tag dword vs the step sentinel
	s_waitcnt vmcnt(2)
	v_cmp_ne_u32_e64 s[6:7], s35, v11
	s_waitcnt vmcnt(1)
	v_cmp_ne_u32_e64 s[8:9], s35, v5
	s_or_b64 s[6:7], vcc, s[6:7]
	s_or_b64 s[6:7], s[6:7], s[8:9]
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e32 vcc, s35, v13
	s_or_b64 vcc, s[6:7], vcc
	s_cbranch_vccnz 65514                                      ; -22 dwords: spin until all four tags land
	v_mov_b32_e32 v3, v10
	v_mov_b32_e32 v5, v12
	ds_write_b128 v7, v[2:5]                                   ; 16 payload bytes into LDS
```

The `sc1` scope bit makes these loads system-coherent, so they see peer writes. In the variants
compiled with the first bool template flag set, the same protocol runs at 16-bit granularity and
the tag lives in the high half of each 32-bit word, which the consumer strips and reassembles
into bf16 pairs before the MFMA:

```
# disasm.gfx950.s:384857-384892, pure_mla_moe_layer_kernel<1,10,Proto1,0,true,true,false>
	global_load_dwordx2 v[20:21], v[24:25], off sc1
	global_load_dwordx2 v[22:23], v[24:25], off offset:8 sc1
	s_waitcnt vmcnt(1)
	v_cmp_ne_u32_sdwa s[14:15], v20, s19 src0_sel:WORD_1 src1_sel:DWORD
	v_cmp_ne_u32_sdwa s[20:21], s19, v21 src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_sdwa s[40:41], v22, s19 src0_sel:WORD_1 src1_sel:DWORD
	v_cmp_ne_u32_sdwa s[42:43], s19, v23 src0_sel:DWORD src1_sel:WORD_1
	s_or_b64 s[14:15], s[14:15], s[20:21]
	s_or_b64 s[20:21], s[40:41], s[42:43]
	s_or_b64 vcc, s[14:15], s[20:21]
	s_cbranch_vccnz 65518                                      ; -18 dwords
	s_and_saveexec_b64 s[14:15], s[6:7]
	s_cbranch_execz 11
	s_memrealtime s[20:21]                                     ; arrival timestamp
	...
	global_store_dwordx2 v24, v[26:27], s[40:41] offset:24
	s_or_b64 exec, exec, s[14:15]
	v_alignbit_b32 v21, v21, v20, 16                           ; strip tags, rebuild bf16 pairs
	v_and_b32_e32 v21, 0xffff0000, v21
	v_and_b32_e32 v23, 0xffff0000, v25
	v_and_b32_e32 v22, 0xffff, v22
	v_or_b32_sdwa v20, v21, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_or3_b32 v21, 0, v22, v23
	ds_write_b64 v1, v[20:21]
```

The sentinel is the wrapping epoch computed in the prologue, `(step mod 65535) + 1` from an Args
word at offset `0x188`. Monotone epochs mean no flag reset, no separate signal buffer and no
release fence on the receive side. The 16-bit form pays a 2x wire cost on bf16 payload to get
16-bit arrival granularity. 21 to 39 `s_memrealtime` reads per instance write per-stage
timestamps into an Args-supplied `unsigned long*`, so the design is instrumented in place.

Producer-side release is a single L2 writeback, gated so that only 8 of the 256 workgroups issue
it. gfx950 packages 8 XCDs and blocks are distributed across them, so this is plausibly one
writeback per XCD, though the mapping is not visible in the binary:

```
# disasm.gfx950.s:326460-326469, kernel +0x4de0 .. +0x4e0c
	global_store_dwordx4 v[8:9], v[2:5], off sc1
	s_nop 1
	s_or_b64 exec, exec, s[34:35]
	s_cmp_lt_i32 s2, 8                                         ; blockIdx.x < 8
	s_cselect_b64 s[34:35], -1, 0
	s_and_b64 s[34:35], s[34:35], s[10:11]
	s_and_saveexec_b64 s[10:11], s[34:35]
	s_cbranch_execz 2
	buffer_wbl2 sc1
	s_or_b64 exec, exec, s[10:11]
```

A software device-scope barrier does appear in the artifact, and the only kernel that combines an
atomic arrival counter with an `s_sleep` back-off poll is the KV-selection kernel: counters via
`global_atomic_add`, a bitmap via `global_atomic_or`, then sleep-poll on a system-coherent load.
All four `s_sleep` in the binary are its four instances, one each.

```
# disasm.gfx950.s:935622-936276, topk_select_kernel<1>
	global_atomic_add v2, v3, v2, s[48:49] offset:1024 sc0
	global_atomic_add v4, v5, v4, s[48:49] offset:1028 sc0
	global_atomic_add v1, v2, s[48:49] offset:1040
	global_atomic_or v3, v2, s[48:49] offset:2064
	global_atomic_or v2, v1, s[48:49] offset:2068
	...
	s_sleep 1
	s_and_saveexec_b64 s[4:5], s[0:1]
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz 65520
	global_load_dword v5, v[2:3], off sc1
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e32 vcc, v5, v4
```

Across the whole binary the sync-primitive inventory is 144 `global_atomic_add`, 48
`global_atomic_swap_x2`, 24 `global_atomic_umax_x2`, 24 `global_atomic_add_x2`, 16
`global_atomic_or` and 4 `s_sleep`, confined to `sparse_index_*`, `topk_*` and
`rmsnorm_head_proj`. None in any of the three layer-body kernels. In `rmsnorm_head_proj` the
atomics read as a cross-device reduction rather than an arrival barrier
(`global_atomic_umax_x2` then `global_atomic_add_x2 ... offset:64` then two
`global_atomic_swap_x2`, disassembly lines 659027-659275); `sparse_index_fp8` carries a single
`global_atomic_add` (line 842068).

Finally, the layer kernel does not loop over layers. The largest backward branch inside it is -709
dwords, well inside one stage. All 48 instances contain exactly one `global_store_short_d16_hi`,
and the three instances inspected end with it immediately followed by `s_endpgm`; the `s_branch`
lines that appear after `s_endpgm` in the MoE kernel are the tail zero-padding decoded as
instructions, a known artefact recorded in the extraction's own README. The 560-byte `Args` blob
also cannot address 78 layers of weights.

## 3. The MoE path

Gate/up GEMM, SwiGLU, activation quantisation, down GEMM, the shared expert and the TP all-reduce
are one kernel. The MoE kernel is the only one in the artifact with an explicit (non-struct)
parameter list, which settles the question directly:

```
# kernels.tsv:131, demangled
void tilert::ops::glm_5_2::fused_moe_allreduce_v4::fused_moe_allreduce_v4_kernel<
    (tilert::cell::xfer::Proto)0, 1, 12, false>(
  __hip_bfloat16 const*,        // p0  hidden in
  float const*,                 // p1  f32 vector (norm gain)
  unsigned char const*,         // p2  quantised weight A
  unsigned char const*,         // p3  quantised weight B
  float const*,                 // p4  scales for A
  float const*,                 // p5  scales for B
  unsigned char const*,         // p6  quantised weight C
  float const*,                 // p7  scales for C
  __hip_bfloat16 const*,        // p8  bf16 weight (router)
  void* const*,                 // p9  peer pointer array
  int, int, unsigned int,
  __hip_bfloat16*,              // p13 output
  float*, unsigned int*, unsigned int*, float*, int*,
  __hip_bfloat16*, __hip_bfloat16*, unsigned int*, unsigned int,
  unsigned long*)               // p23 timestamp sink
```

Three (uint8 weight, f32 scale) pairs plus one bf16 weight plus the peer array, all in one
kernel. The prologue loads all of p0..p7 in a single wide fetch, so no weight tensor is reached
only on a cold path:

```
# disasm.gfx950.s:160901-160904, fused_moe_allreduce_v4_kernel<Proto0,1,12,false>
	s_load_dwordx16 ..., s[0:1], 0x0     ; offsets 0..63  -> p0..p7, both weights+scales and the shared weight
	s_load_dwordx2  ..., s[0:1], 0x70    ; p15
	s_load_dwordx8  ..., s[0:1], 0x80    ; p17..
	s_load_dwordx2  ..., s[0:1], 0xa0    ; p21
# disasm.gfx950.s:163516
	s_load_dwordx8  ..., s[0:1], 0x40    ; p8..p11, the router weight and the peer array, for the epilogue
```

Compute is laid out as four MFMA stages separated by barriers, with two sigmoid blocks and a
top-k block between them. Segment census on the standalone MoE kernel (`mfma` is all MFMA forms,
`cvt8` is `v_cvt_scalef32_pk_bf16_fp8`, `exp`/`rcp` count the sigmoid expansion):

```
seg0   n=91    mfma=0    cvt8=0    dsR=0    dsW=1    gL=6    gS=0   exp=0   rcp=0
seg1   n=209   mfma=0    cvt8=0    dsR=2    dsW=3    gL=20   gS=0   exp=0   rcp=5
seg2   n=337   mfma=24   cvt8=96   dsR=26   dsW=4    gL=0    gS=0   exp=0   rcp=5
seg3   n=299   mfma=0    cvt8=0    dsR=14   dsW=1    gL=18   gS=2   exp=4   rcp=20
seg4   n=176   mfma=24   cvt8=0    dsR=15   dsW=4    gL=0    gS=1   exp=0   rcp=5
seg5   n=261   mfma=0    cvt8=0    dsR=5    dsW=2    gL=8    gS=2   exp=1   rcp=5   | consts 2147483647, 2147483648, 12304, 512
seg6   n=548   mfma=24   cvt8=96   dsR=27   dsW=5    gL=24   gS=0   exp=0   rcp=5
seg7   n=231   mfma=0    cvt8=0    dsR=16   dsW=1    gL=0    gS=4   exp=4   rcp=20
seg8   n=204   mfma=0    cvt8=2    dsR=7    dsW=4    gL=15   gS=0   exp=0   rcp=0
seg9   n=212   mfma=36   cvt8=70   dsR=23   dsW=1    gL=0    gS=0   exp=0   rcp=0
seg10  n=50    mfma=0    cvt8=0    dsR=8    dsW=1    gL=0    gS=0   exp=0   rcp=0
seg11  n=41    mfma=0    cvt8=0    dsR=1    dsW=0    gL=1    gS=1   exp=0   rcp=0
seg12  n=38    mfma=0    cvt8=0    dsR=0    dsW=1    gL=2    gS=0   exp=0   rcp=0
seg13  n=137   mfma=0    cvt8=0    dsR=9    dsW=0    gL=1    gS=1   exp=0   rcp=0
```

The sigmoid is explicit, built from `exp2` with the `-log2(e)` constant and a full-precision
reciprocal, immediately after a cross-lane `ds_bpermute` reduction, which is the router gating
shape:

```
# disasm.gfx950.s:161612-161642, fused_moe_allreduce_v4_kernel
	ds_bpermute_b32 v10, v1, v8
	ds_bpermute_b32 v6, v1, v4
	ds_bpermute_b32 v11, v1, v9
	ds_bpermute_b32 v7, v1, v5
	...
	s_mov_b32 s4, 0xbfb8aa3b                                   ; -log2(e)
	v_pk_mul_f32 v[2:3], v[2:3], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], s[4:5] op_sel_hi:[1,0]
	v_exp_f32_e32 v14, v2
	v_exp_f32_e32 v15, v12
	v_exp_f32_e32 v12, v3
	v_exp_f32_e32 v13, v13
	s_movk_i32 s6, 0x7fff
	v_pk_add_f32 v[14:15], v[14:15], 1.0 op_sel_hi:[1,0]       ; 1 + exp(-x)
	v_div_scale_f32 v1, s[4:5], v15, v15, 1.0
	v_rcp_f32_e32 v16, v1
	v_fma_f32 v3, -v1, v16, 1.0
	v_fmac_f32_e32 v16, v3, v16
	v_div_scale_f32 v3, vcc, 1.0, v15, 1.0
	v_mul_f32_e32 v18, v3, v16
	v_fma_f32 v19, -v1, v18, v3
	v_fmac_f32_e32 v18, v19, v16
	v_fma_f32 v1, -v1, v18, v3
```

The expert dimension is handled by workgroup ownership rather than a grouped-GEMM descriptor:
each work item reads an owner tag out of LDS and compares it to its own work id (`s_cmp_eq_u32
s8, s81` in the quote in section 2) before touching the block. There is no `group_list`-style
cumulative offset array in the signature; the two `unsigned int*` and one `int*` outputs are the
routing side-channel.

The quantisation path is a template switch, not a separate kernel. The fourth template parameter
of `pure_mla_moe_layer_kernel` picks it, and the counts prove which:

```
# per-instance op counts, kernels.tsv + disasm.gfx950.s
co13  v_mfma_scale=0   cvt_pk_fp8_f32=0   cvt_pk_bf16_fp8=496  mfma_bf16=112  <1, 8, Proto0, 0, false, false, false>
co14  v_mfma_scale=17  cvt_pk_fp8_f32=16  cvt_pk_bf16_fp8=232  mfma_bf16=64   <1, 8, Proto0, 1, false, false, false>
co13  v_mfma_scale=0   cvt_pk_fp8_f32=0   cvt_pk_bf16_fp8=580  mfma_bf16=112  <2, 10, Proto0, 0, false, false, false>
co14  v_mfma_scale=26  cvt_pk_fp8_f32=16  cvt_pk_bf16_fp8=244  mfma_bf16=64   <2, 10, Proto0, 1, false, false, false>
```

Parameter 4 = 0 dequantises FP8 weights to bf16 and uses `v_mfma_f32_16x16x32_bf16` throughout,
which is exactly `fused_moe_allreduce_v4` (0 `v_mfma_scale` in all 30 instances). Parameter 4 = 1
quantises activations to FP8 and uses the native microscaled MFMA, which is exactly
`fused_moe_allreduce_w8a8_v4` (600 `v_mfma_scale` over 24 instances, 25 each). The two paths live
in code objects 13 and 14, 24 instances each, selected by the `test r13,r13` at `0x78da01`.

## 4. The attention path

The MLA prolog, the attention itself, the output projection and the TP all-reduce are one kernel,
which is then itself fused into the layer kernel. `pure_mla_allreduce_kernel` takes a 424-byte
`Args` struct, requests 109072 B of LDS, has 16 barriers and 3339-4523 static instructions, and
contains seven MFMA stages:

```
# segment census, pure_mla_allreduce / pure_mla_moe_layer segments 0-16
seg0   mfma=24   cvt8=96   dsR=24   dsW=4    gL=24   gS=0
seg1   mfma=0    cvt8=0    dsR=9    dsW=0    gL=1    gS=4   rcp=5
seg2   mfma=8    cvt8=32   dsR=8    dsW=3    gL=7    gS=0
seg3   mfma=6    cvt8=24   dsR=15   dsW=3    gL=15   gS=8
seg4   mfma=0    cvt8=0    dsR=11   dsW=20   gL=49   gS=0          ; KV staging, nine tagged spin loops
seg5   mfma=18   cvt8=0    dsR=36   dsW=1    gL=0    gS=0
seg6   mfma=0    cvt8=0    dsR=4    dsW=2    gL=0    gS=0   exp=4  ; 4 exp2 sites, between the two MFMA groups
seg7   mfma=16   cvt8=0    dsR=68   dsW=0    gL=0    gS=6
seg8   mfma=0    cvt8=0    dsR=0    dsW=0    gL=1    gS=1
seg9   mfma=0    cvt8=0    dsR=0    dsW=0    gL=0    gS=1   rcp=5
seg10  mfma=0    cvt8=0    dsR=0    dsW=1    gL=2    gS=0   exp=1
seg11  mfma=8    cvt8=32   dsR=13   dsW=3    gL=33   gS=0
seg12  mfma=24   cvt8=48   dsR=6    dsW=7    gL=13   gS=3
seg13  mfma=0    cvt8=0    dsR=4    dsW=1
seg14  mfma=0    cvt8=0    dsR=1    dsW=0    gL=1    gS=1
seg15  mfma=0    cvt8=0    dsR=0    dsW=1    gL=2    gS=0
seg16  mfma=0    cvt8=0    dsR=9    dsW=0    gL=1    gS=1
```

Dimension constants confirm MLA and confirm that the input RMSNorm is inside: `0x240` = 576 (the
compressed KV width, 512 latent plus 64 rope) appears at `+0xa20c`, `+0xed0c` and `+0xfe48`;
`0x45c00000` = 6144.0f, the RMSNorm divisor over hidden, appears 4 times; `0x3727c5ac`, which is
1e-5, appears 6 times; `0x3000` = 12288 = 6144 bf16 is the row stride.

The KV selection stage is not inside the attention kernel. It is a separate launch, offered in a
fused form and an unfused form:

```
# kernels.tsv:421, demangled
void tilert::ops::glm_5_2::sparse_select_s0123::sparse_select_s0123_kernel<1, true, 6>(
  __hip_bfloat16 const*, float const*, unsigned char const*, float const*, unsigned char const*,
  __hip_bfloat16*, __hip_bfloat16*, __hip_bfloat16*, unsigned int*, unsigned int*, unsigned int*,
  float const*, unsigned char const*, float const*, __hip_bfloat16*, float const*, float const*,
  float const*, int const*, __hip_bfloat16*, __hip_bfloat16*, unsigned char*, float*,
  unsigned char*, float*, int, int, unsigned int)
# kernels.tsv:356, the unfused alternative
void tilert::ops::glm_5_2::sparse_index::sparse_index_fp8_kernel<1, 1, 1, true, 8, false>(
  unsigned char const*, float const*, unsigned char const*, float const*, __hip_bfloat16 const*,
  int const*, float*, unsigned int*, int, int, int, int, int, int)
# kernels.tsv:413
void tilert::ops::glm_5_2::sparse_index::topk_select_kernel<1>(
  float const*, unsigned int*, HIP_vector_type<int, 2u>*, int*, int const*, int, int, int, int, int)
```

`sparse_select_s0123` names four fused sub-stages, takes 28 parameters including three
(uint8 weight, f32 scale) pairs and three index outputs, and is 1852-2586 instructions at
12848-51392 B of LDS and grid 256. The unfused chain is `sparse_index_fp8` or `sparse_index_ts16`
(640-4787 instructions each) followed by `topk_select`, and `topk_select` is where the on-chip
cost of selection shows up: 158896 B of LDS, within 1.5 KiB of the gfx950 per-CU ceiling, 1024
threads, 8158-13350 instructions, and the artifact's only atomic-counter plus sleep-poll barrier.
Launch configs for the index chain pass the grid in a register (`mov edi,r14d`), `blockDim` fixed:
`movabs rdx,0x100000100` for both `sparse_index_*`, so 256 threads, and
`movabs rdx,0x100000400` for `topk_select`, so 1024 threads.

The index stage feeds the attention kernel through memory, not registers. The attention kernel's
KV staging segment reads owner tags out of LDS and page payloads out of system-coherent global
memory, as quoted in section 2.

## 5. Resource shapes

All kernels are wave64. Measured resources for the 20 reachable families, over all compiled
specialisations of each (ranges where instances differ):

| kernel | compiled inst | reachable sites | LDS B | VGPR | SGPR | max wg | kernarg B | barriers | static instrs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `topk_select_kernel` | 4 | 4 | 158896 | 49-99 | 106 | 1024 | 60 | 32 | 8158-13350 |
| `pure_mla_moe_layer_kernel` | 48 | 48 | 109072-150592 | 110-230 | 92-106 | 512 | 560 | 30-34 | 6087-7864 |
| `topk_collect_cta_kernel` | 4 | 0 | 3108-24864 | 34-97 | 58-106 | 256 | 60 | 4 | 932-5425 |
| `sparse_index_ts16_kernel` | 63 | 3 | 9344-37376 | 98-512 | 41-55 | 256-512 | 328 | 3 | 653-4787 |
| `pure_mla_allreduce_kernel` | 24 | 24 | 109072-150592 | 112-128 | 80-100 | 512 | 424 | 16 | 3339-4523 |
| `sparse_index_fp8_kernel` | 57 | 3 | 5248-121088 | 73-496 | 45-72 | 256-512 | 344 | 3 | 640-4240 |
| `topk_finish_cta_kernel` | 1 | 0 | 158896 | 46 | 84 | 1024 | 56 | 13 | 3841 |
| `fused_moe_allreduce_v4_kernel` | 30 | 6 | 27312-57872 | 100-234 | 52-53 | 512 | 184 | 12-13 | 2834-3480 |
| `fused_moe_allreduce_w8a8_v4_kernel` | 24 | 6 | 27152 | 98-164 | 58-61 | 512 | 184 | 14-16 | 2968-3321 |
| `sparse_select_s0123_kernel` | 6 | 6 | 12848-51392 | 68-97 | 58-62 | 512 | 212 | 3 | 1852-2586 |
| `rmsnorm_head_proj_kernel` | 48 | 12 | 15396-105888 | 74-276 | 22-69 | 256-512 | 360 | 5-9 | 575-2297 |
| `top_p_sample_kernel` | 2 | 2 | 5212-26800 | 21-24 | 29-42 | 512 | 88 | 11 | 897-1650 |
| `top_p_local_scores_kernel` | 1 | 1 | 88948 | 60 | 42 | 512 | 48 | 11 | 1508 |
| `eh_proj_allreduce_kernel` | 10 | 10 | 12368-34336 | 34-50 | 42-54 | 768 | 128 | 7 | 544-1235 |
| `rmsnorm_upgate_silu_kernel` | 6 | 3 | 18480-56512 | 79-120 | 18-25 | 512 | 40 | 1-2 | 574-946 |
| `mtp_step_tail_kernel` | 1 | 1 | 0 | 17 | 58 | 1024 | 152 | 1 | 662 |
| `llm_preprocess_kernel` | 1 | 1 | 0 | 12 | 26 | 256 | 52 | 0 | 418 |
| `broadcast_selected_token_ids_kernel` | 1 | 1 | 0 | 10 | 16 | 256 | 32 | 0 | 311 |
| `padded_allreduce_add_kernel` | 6 | 6 | 384-1536 | 14-18 | 28-30 | 256 | 40 | 3 | 216-276 |
| `ar_tail_no_mtp_kernel` | 1 | 1 | 0 | 9 | 22 | 1024 | 64 | 0 | 103 |
| `simulate_accept_len_kernel` | 1 | 1 | 0 | 10 | 14 | 1024 | 28 | 0 | 60 |
| `glm52_prefill_next_kernel` | 1 | 1 | 0 | 2 | 11 | 1024 | 16 | 0 | 12 |

The LDS figure for the layer kernel is exactly linear in the speculation width, and it is
identical to the attention-only kernel at every width:

```
# kernels.tsv, .group_segment_fixed_size by first template parameter
tokens=1  95232 + 13840*1 = 109072     pure_mla_moe_layer  109072   pure_mla_allreduce  109072
tokens=2  95232 + 13840*2 = 122912     pure_mla_moe_layer  122912   pure_mla_allreduce  122912
tokens=4  95232 + 13840*4 = 150592     pure_mla_moe_layer  150592   pure_mla_allreduce  150592
```

95232 B of token-independent arena plus 13840 B per token. Fusing the MoE half in cost zero extra
LDS: the MoE stage reuses the attention arena, which is the whole reason a 109 KB stage boundary
is affordable. The standalone MoE kernel needs 27312 B on its own (41584 at 2 tokens, 57872 at 4
for the W8A16 form; the W8A8 form is a flat 27152 at all widths), all of which fits inside the
attention arena.

Inside that arena the LDS reference offsets cluster at a 1296-byte stride, and 0x14400 = 82944 =
64 x 1296 appears as an immediate, so 82944 B of it is a 64-slot tile of 1296-byte records.
1296 B is 648 bf16, which is the 576-wide compressed KV row padded by 72 elements. The remaining
12288 B of the token-independent arena is exactly one 6144-wide bf16 row.

```
# ds_* immediate-offset clusters, pure_mla_moe_layer_kernel<1,8,Proto0,0,false,false,false>
  [     0 ..   1472]  span=1472    refs=351  {'write': 50, 'read': 271, 'other': 30}
  [  2592 ..   2656]  span=64      refs=6
  [  3888 ..   3952]  span=64      refs=6     ; stride 1296 between cluster bases
  [  5184 ..   5184]  span=0       refs=2
  [  6480 ..   6480]  span=0       refs=2
  [  7776 ..   7776]  span=0       refs=2
  [  9072 ..   9072]  span=0       refs=2
  [ 20540 ..  25248]  span=4708    refs=52
  [ 41472 ..  41664]  span=192     refs=7
  [ 62208 ..  62272]  span=64      refs=4
  [ 64800 ..  64864]  span=64      refs=4
```

Register and spill behaviour separates the fused kernels from the index stage sharply. Zero AGPRs
and zero spills in all three layer-body kernels; the only AGPR users in the whole binary are
`sparse_index_ts16_kernel` (31 instances, up to 256 AGPRs), `sparse_index_fp8_kernel` (18
instances, up to 240) and `rmsnorm_head_proj_kernel` (2 instances, 32); the only spilling kernels
in the binary are `sparse_index_ts16` and `sparse_index_fp8`, up to 314 VGPR spills and 1132 B of
scratch. Fusing attention and MoE did not cost register pressure. Computing the index did.

Software pipelining depth, read as the deepest `s_waitcnt vmcnt(N)` per kernel, so the number of
vector-memory operations in flight at the deepest point (gfx9 permits 63):

```
  max vmcnt=62  tilert::ops::glm_5_2::sparse_index::sparse_index_ts16_kernel
  max vmcnt=54  tilert::ops::glm_5_2::rmsnorm_head_proj::rmsnorm_head_proj_kernel
  max vmcnt=45  tilert::ops::glm_5_2::sparse_index::sparse_index_fp8_kernel
  max vmcnt=31  tilert::ops::glm_5_2::fused_moe_allreduce_v4::fused_moe_allreduce_v4_kernel
  max vmcnt=24  tilert::ops::glm_5_2::fused_moe_allreduce_w8a8_v4::fused_moe_allreduce_w8a8_v4_kernel
  max vmcnt=24  tilert::ops::glm_5_2::pure_mla_moe_layer::pure_mla_moe_layer_kernel
  max vmcnt=22  tilert::ops::glm_5_2::upgate_silu::rmsnorm_upgate_silu_kernel
  max vmcnt=21  tilert::ops::cell::gemv_w8a16::gemv_w8a16_mfma_kernel
  max vmcnt=14  tilert::ops::glm_5_2::pure_mla_allreduce::pure_mla_allreduce_kernel
```

LDS double buffering is visible in the MFMA inner loops as a 1-deep `lgkmcnt` overlap: each
`v_mfma` consumes the `ds_read_b128` pair issued two iterations earlier, with
`s_waitcnt lgkmcnt(1)` releasing one while the next pair is in flight (see the quote in
section 6).

## 6. Quantisation layout

No W4 of any kind. Searching the full 1.02M-line disassembly for 4-bit unpack and FP4 conversion:

```
v_bfe_u32 \S+, \S+, [0-9]+, 4$             0
0xf0f0f0f                                  0
0x0f0f0f0f                                 0
v_cvt_scalef32_pk\S*fp4                    0
v_cvt_pk_fp4                               0
```

And on the kernel names, every quantisation token present in the artifact:

```
# kernels.tsv, name column
    174 w8a16
     57 fp8
     48 w8a8
```

The whole MFMA inventory of the binary is four opcodes:

```
  19644 v_mfma_f32_16x16x32_bf16
   9930 v_mfma_f32_4x4x4_16b_bf16
   3256 v_mfma_scale_f32_16x16x128_f8f6f4
   1984 v_mfma_f32_16x16x16_bf16
```

`v_mfma_scale_f32_16x16x128_f8f6f4` is the gfx950 microscaled MFMA that can take FP8, FP6 or FP4
operands, and the operand format is encoded in the register tuple width. All 3256 instances use
the 8-register form, which is 8 bits per element:

```
  widths(D,A,B) = (4, 8, 8)  count 3256
```

llvm-mc confirms that the tuple width is what selects the format, by rejecting a 4-bit format on
an 8-register operand:

```
$ llvm-mc -arch=amdgcn -mcpu=gfx950 -show-encoding
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], 0, v20, v21 cbsz:4 blgp:4
  error: wrong register tuple size for cbsz value 4
  error: wrong register tuple size for blgp value 4
v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:11], v[12:19], 0, v20, v21
  encoding: [0x00,0x00,0xac,0xd3,0x14,0x2b,0x02,0x00,0x00,0x08,0xad,0xd3,0x04,0x19,0x02,0x02]
```

and the real instances carry the same default format bits as that golden encoding:

```
# disasm.gfx950.s, raw encodings of v_mfma_scale_f32_16x16x128_f8f6f4 (most frequent forms)
   2450 v_mfma_scale_f32_16x16x128_f8f6f4 V, V, V, 0, v, v op_sel_hi:[0,0,0]
      8 D3AC0000 00023F1F D3AD0820 0202C95C
      8 D3AC0000 00023B1D D3AD082C 0202A12C
```

So weights are stored as `unsigned char` arrays with separate `float` scale arrays, and the bytes
are FP8 E4M3. Dequantisation happens in registers, interleaved with the MFMA, not in a separate
pass and not in a separate kernel. `v_cvt_scalef32_pk_bf16_fp8` at 46632 occurrences is the third
most frequent opcode in the binary, behind only `s_waitcnt` (60549) and `v_mov_b32_e32` (51038),
and this is what it looks like next to the MFMA it feeds:

```
# disasm.gfx950.s:384895-384941, pure_mla_moe_layer_kernel<1,10,Proto1,0,true,true,false>
	v_cvt_scalef32_pk_bf16_fp8 v20, v14, 1.0
	v_cvt_scalef32_pk_bf16_fp8 v21, v14, 1.0 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v22, v15, 1.0
	v_cvt_scalef32_pk_bf16_fp8 v23, v15, 1.0 op_sel:[1,0,0]
	ds_read_b128 v[24:27], v1
	ds_read_b128 v[28:31], v1 offset:64
	v_cvt_scalef32_pk_bf16_fp8 v14, v16, 1.0
	v_cvt_scalef32_pk_bf16_fp8 v15, v16, 1.0 op_sel:[1,0,0]
	v_cvt_scalef32_pk_bf16_fp8 v16, v17, 1.0
	v_cvt_scalef32_pk_bf16_fp8 v17, v17, 1.0 op_sel:[1,0,0]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[20:23], v[20:23], v[24:27], 0
	v_mfma_f32_16x16x32_bf16 v[14:17], v[14:17], v[28:31], v[20:23]
	ds_read_b128 v[24:27], v1 offset:128
	ds_read_b128 v[28:31], v1 offset:192
	s_nop 4
	v_cvt_scalef32_pk_bf16_fp8 v20, v10, 1.0
	...
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_bf16 v[20:23], v[20:23], v[24:27], 0
	v_fma_f32 v14, v14, v18, 0
	v_fma_f32 v15, v15, v18, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_bf16 v[10:13], v[10:13], v[28:31], v[20:23]
```

Four `v_cvt` per 8 bf16 lanes, then the MFMA, then the per-group scale applied as `v_fma_f32 ...,
v18` on the accumulator. The scale is applied after the MFMA, not folded into the dequantisation
(the `v_cvt` scale operand is the literal `1.0`), which is what lets one weight block be reused
across scale groups.

The W8A8 variant instead quantises activations forward. `v_cvt_scalef32_pk_fp8_f32` appears 432
times across 30 of the 48 layer-kernel instances, 384 times across all 24
`fused_moe_allreduce_w8a8_v4` instances, 36 times across 9 of the 24 `pure_mla_allreduce`
instances, and 24 times across 3 of the 6 `sparse_select_s0123` instances. KV cache handling in
the attention kernel is bf16 in LDS. The layer kernel also carries `0xc2fc0000`, which is
-126.0f, twice; an exponent-range constant of that value is consistent with FP8 scale clamping
but the binary does not establish the use.

Outputs are bf16 with explicit round-to-nearest-even, in every all-reduce-terminated kernel:

```
# disasm.gfx950.s:329145-329150, tail of pure_mla_moe_layer_kernel
	v_bfe_u32 v3, v2, 16, 1
	s_movk_i32 s0, 0x7fff
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[4:5]
	v_add3_u32 v2, v2, v3, s0
	global_store_short_d16_hi v[0:1], v2, off
	s_endpgm
```

## 7. Decode-specific specialisation

Speculation width is a compile-time template parameter with three values, and it is the only thing
that moves the LDS footprint. The full specialisation space of the layer kernel, 48 instances:

```
# kernels.tsv:194-241, template arguments of pure_mla_moe_layer_kernel
co13 lds=109072 vgpr=118 sgpr= 94 inst= 6087  <1, 8,  Proto0, 0, false, false, false>
co13 lds=109072 vgpr=118 sgpr= 94 inst= 6102  <1, 8,  Proto1, 0, false, false, false>
co13 lds=122912 vgpr=162 sgpr= 94 inst= 6604  <2, 8,  Proto0, 0, false, false, false>
co13 lds=122912 vgpr=160 sgpr= 94 inst= 6603  <2, 8,  Proto1, 0, false, false, false>
co13 lds=150592 vgpr=230 sgpr= 96 inst= 7181  <4, 8,  Proto0, 0, false, false, false>
co13 lds=150592 vgpr=230 sgpr= 96 inst= 7173  <4, 8,  Proto1, 0, false, false, false>
co13 lds=109072 vgpr=118 sgpr= 94 inst= 6271  <1, 10, Proto0, 0, false, false, false>
co13 lds=109072 vgpr=110 sgpr=106 inst= 6677  <1, 10, Proto1, 0, true,  false, true >
co13 lds=109072 vgpr=110 sgpr=102 inst= 6406  <1, 10, Proto1, 0, false, false, true >
co13 lds=109072 vgpr=124 sgpr= 92 inst= 6631  <1, 10, Proto1, 0, true,  true,  false>
co13 lds=109072 vgpr=118 sgpr= 92 inst= 6564  <1, 10, Proto1, 0, true,  false, false>
co13 lds=109072 vgpr=118 sgpr= 94 inst= 6289  <1, 10, Proto1, 0, false, false, false>
...                                            (same 24-way pattern in co14 with param 4 = 1)
```

Parameter 1 takes 1, 2 and 4, which is token count, matching the MTP structure the extraction's
own README read off the host code: `num_mtp=1` and `num_mtp=3` branches, so 1, 2 and 4 tokens per
layer invocation. There is a dedicated batch-1 instance in every family, and it is the cheapest:
118 VGPRs and 109072 B at width 1 against 230 VGPRs and 150592 B at width 4, with LDS exactly
`95232 + 13840 * tokens`. Parameter 2 takes 8 and 10, and scales an internal bound by two
(`s_cmpk_lt_i32 s83, 16` in the `8` instance, `s_cmpk_lt_i32 s83, 20` in the `10` instance), while
the L2-writeback guard stays at `blockIdx.x < 8` in both. Parameter 3 is
`tilert::cell::xfer::Proto` with two values, the two transfer protocols. Parameter 4 is the MoE
numeric path, 0 for W8A16 and 1 for W8A8, as established in section 3. The three bools gate the
in-kernel overlapped transfer: they only exist in combination with parameter 2 = 10 and Proto1,
and setting the first turns on 21-39 `s_memrealtime` reads and raises `global_store` from 30 to
51-60 per instance, which is the kernel writing peer buffers itself rather than leaving it to a
separate all-reduce launch.

The MoE kernels carry an additional tiling parameter that the layer kernel hides:
`fused_moe_allreduce_v4` takes 9, 12, 18, 24 and 36; `fused_moe_allreduce_w8a8_v4` takes 4, 6, 8
and 12, exactly one third of the former set on the overlap, so the W8A16 form needs three times
the tiles for the same shape.

The whole round is captured once as a HIP graph and replayed with two alternating instances per
mode, which the extraction's README establishes from `0x7700a0` (one single-step call between
`hipStreamBeginCapture` and `hipStreamEndCapture`), `0x7711c0` (`hipGraphLaunch` n times, taking
graph `(step + i) & 1`) and the instance table indexed by `card * 128 + slot * 8 + 0x48`. The
same README notes a non-graph branch in `0x7711c0` that calls the single-step function directly.
Speculation bookkeeping is four small kernels: `simulate_accept_len_kernel` (60 instructions),
`mtp_step_tail_kernel` (662), `ar_tail_no_mtp_kernel` (103, the no-MTP path) and
`eh_proj_allreduce_kernel` (544-1235, 10 specialisations). The library links against PyTorch
(`c10::cuda::CUDAStream::stream()`, `c10::detail::torchCheckFail`), so the layer loop itself is
above this library and not statically visible.

## Relevance to Ascend

Our per-layer vendor sequence is MlaPrologV3, SparseFlashAttention, transpose, O projection,
router matmul, MoeGatingTopK, MoeInitRoutingV3, GroupedMatmulSwigluQuantV2, GroupedMatmul,
MoeTokenUnpermute, shared expert as two quantised matmuls with a dequant/swiglu between, then
scale, add, cast, norm, quantise: 26 kernels, 6.5 of them MIX_AIC, 593.3 us measured against
128.9 us of bytes.

TileRT's ROCm answer merges all of that except the index stage into one kernel.
`pure_mla_moe_layer_kernel` contains the equivalent of MlaPrologV3, SparseFlashAttention, the
transpose, the O projection, the TP reduce, the router matmul, the gating sigmoid and top-k, the
routing permute, gate/up, SwiGLU, activation quantisation, down, the unpermute, the shared
expert, the second TP reduce, and the scale/add/cast/norm/quantise epilogue. What stays outside
is the KV index and selection stage (our 2048-token index/selection, their `sparse_select_s0123`
or `sparse_index_* + topk_select`) and nothing else. That is the single most important number in
this document: the reference design's answer to our 26 per layer is 1 plus the indexer.

Mapping finding by finding:

- Fusion granularity (1). Their layer boundary is the kernel boundary. Our 26-kernel sequence has
  no counterpart in the reachable set; the `gemv_w8a16*` family is the one-op-per-primitive design
  and it is compiled, registered and never called. If we are deciding how large the first AscendC
  megakernel should be, the reference answer is the whole layer minus the indexer, not the
  attention block and not the MoE block.
- No persistent kernel (2). We do not need `aclrtLaunchMega` or a persistent work queue to get the
  bulk of the win. A per-layer launch at 78 launches per round, against our ~3008, already removes
  the 11.5 ms launch floor that 82% MIX occupancy imposes. The device-sized grid matters for a
  different reason: at 256 workgroups of 512 threads with 109 KB of LDS, occupancy is one
  workgroup per CU by construction, so the fused kernel owns the machine and can treat LDS as a
  private arena across stages. The Ascend analogue is one block per AI Core with the full UB/L1
  budget reserved for the whole layer, which is the precondition for everything below.
- Stage boundaries cost barriers, not state (2, 5). 31 workgroup barriers for 32 stages, and
  fusing the MoE half onto the attention half added exactly two. On Ascend that is
  `SyncAllMegaKernel`-class cross-core sync at every stage boundary, 30-ish per layer. Our
  CrossCoreSetFlag counter limit and the 0-7 event-id budget are the constraint to check against
  that number first.
- The MoE merge needs zero extra on-chip state (3, 5). `pure_mla_moe_layer` LDS equals
  `pure_mla_allreduce` LDS at all three speculation widths: 109072, 122912, 150592. The MoE stage
  reuses the attention arena. This is the cheapest merge available to us: our
  GroupedMatmulSwigluQuantV2 + GroupedMatmul + MoeTokenUnpermute + the two shared-expert matmuls
  can be appended to the attention kernel without widening the UB/L1 reservation, provided we are
  willing to overwrite the attention working set at the seam. Their arena is 95232 B fixed plus
  13840 B per token, of which 82944 B is a 64-slot KV tile of 1296-byte records (576 compressed
  elements padded to 648) and 12288 B is one hidden row.
- The expert dimension is ownership, not a descriptor (3). No `group_list` analogue appears in the
  MoE signature; a work item reads an owner tag out of LDS and compares it to its own id. Our
  MoeInitRoutingV3 produces a routing table; their equivalent is an in-LDS owner array consumed by
  the same kernel that does the GEMM, so the permute never round-trips to HBM. That removes
  MoeInitRoutingV3 and MoeTokenUnpermute as separate kernels, and it is the one piece of their
  design that needs a routing-table layout decision from us before any code is written.
- The shared expert is in the same kernel (3). Three (uint8 weight, f32 scale) pairs in one
  parameter list, all loaded in the prologue by a single `s_load_dwordx16`. Our shared expert is
  currently two separate quantised matmuls with a dequant/swiglu between; those are three of the
  26 and they merge into the same stage.
- Attention is one kernel including prolog, O projection and the reduce (4). Our MlaPrologV3 (2.950
  ms measured against 2.745 ms of bytes, so nearly byte-bound and not the problem) and
  SparseFlashAttention (6.973 against 1.382, a 5x gap) merge, and the O projection and the TP
  reduce merge with them. Their seven MFMA stages inside the attention half tell us the fused
  shape: four before the KV staging, softmax, then two after, all through one 109 KB arena.
- Selection stays separate, and it is the expensive thing (4, 5). `topk_select_kernel` uses 158896
  B of LDS out of 160 KiB, 1024 threads, and the artifact's only atomic-counter plus sleep-poll
  device barrier. `sparse_index_*` are the only kernels in
  the binary that spill (314 VGPRs, 1132 B scratch) and the only ones that use AGPRs (up to 256).
  TileRT did not fuse selection into the layer and the resource numbers say why. Our 2048-token
  index/selection stage should stay its own kernel, and we should expect it to be the one place
  where on-chip capacity, not launch count, is the binding constraint.
- Quantisation is W8/FP8, not W4 (6). This is the sharpest divergence from our target. Their
  weights are uint8 FP8 E4M3 with f32 scales; every MFMA operand is 8 bits per element; there is
  no FP4 path compiled at all. So the artifact gives us the fusion shape but not the W4 unpack
  layout, and the question of where to put the W4 dequant relative to the cube is unanswered by
  it. What it does answer is the placement principle: dequantisation is four `v_cvt` immediately
  before the MFMA that consumes them, in the same instruction stream, with the per-group scale
  applied to the f32 accumulator after the MFMA rather than folded into the dequant. Our W4A8
  antiquant should sit in the same position relative to the cube, and the group scale should ride
  the accumulator.
- Decode specialisation is compile-time (7). Three token widths (1, 2, 4) as a template parameter,
  with the batch-1 instance taking half the registers and 27% less LDS than the width-4 instance.
  Not a runtime branch, not a dynamic grid. For us that means one AscendC kernel per speculation
  width, with the batch-1 variant tuned separately, rather than one kernel that handles all widths.
- The TP reduce is inside the kernel and needs no fence (2). Their cross-GPU protocol packs a
  16-bit epoch tag into the high half of every 32-bit word and spins on it, with the epoch being
  `(step mod 65535) + 1` so nothing is ever reset, and a single `buffer_wbl2` from 8 of 256
  workgroups as the only release. No atomics, no barrier, no host round trip. Our TP reduce is 5.35
  ms and 100% serial; this is the mechanism that makes it overlappable, and it is cheap to port in
  principle since Ascend has the same ingredients (symmetric peer buffers, a per-step counter,
  a flush). The cost is 2x wire traffic on bf16 payload, which is the trade to evaluate against our
  5.35 ms.

## Gaps

- The per-layer loop. The layer-body kernel is launched from one source-level call site and the
  stripped host code shows no enclosing loop; the library links PyTorch, so the loop is in Python
  or in a caller outside this library. The artifact therefore proves one kernel covers one layer
  (560-byte `Args`, no outer backward branch, `s_endpgm` at the inspected exits) but does not show
  the round's launch sequence. Determining it needs a runtime trace, or the TileRT Python source.
  For the same reason the artifact does not corroborate the 3-dense/75-MoE layer split: every
  `xN` in `graph-call-trees.txt` is folded template specialisations, and for every family checked
  the N is the speculation-width fan-out, not a layer count.
- The 164/165 bound. `s_cmpk_lt_i32 s4, 0xa5` then `s_cmpk_lt_i32 s81, 0xa4` gates the first stage
  to work ids 0..163 with 164 special, over 192 role-B workgroups, with a per-item stride of 12288
  B (one 6144-wide bf16 row) indexed as `(work_id * 8 + wave)`. No GLM-5.2 dimension I can derive
  gives 164 or 165, and the base pointer comes from an opaque `Args` offset (0xe8). Resolving it
  needs the `Args` struct layout, i.e. the TileRT source or a debug build.
- What the two workgroup roles do. `blockIdx.x % 8 < 2` splits 64 workgroups from 192, and role A
  is excluded from the first stage by construction (its ids land in 192..255, always above the
  clip). Which stage role A serves, and whether the split is compute/communication specialisation,
  is not determinable without following the role predicate through register reuse, which the
  stripped disassembly does not support cleanly.
- Which of the three MoE weight pointers is which. The signature gives three (uint8, f32) pairs;
  the pairing is unambiguous from the ordering, the roles (routed gate/up, routed down, shared) are
  the natural reading but are not proved by the binary. The routed and shared weights could also be
  one concatenated tensor with the shared expert as an extra expert row.
- How the non-expert MLP completes. `rmsnorm_upgate_silu_kernel` takes exactly one weight and one
  scale and emits bf16, so it covers norm + up + gate + SiLU only. No reachable kernel is an
  obvious down projection for it; `fused_moe_allreduce_*` with routing disabled is a plausible
  candidate, and so is the possibility that this kernel is not the dense-layer path at all.
  Nothing in the binary decides it.
- Whether the KV index stage runs per layer or once per round. Both the fused and unfused index
  chains sit in the same host function as the layer body, but with no visible loop the question is
  the same as the first gap. The GLM sparse-attention design implies per layer; the artifact does
  not say.
- Actual grid sizes for the MoE kernels. `fused_moe_allreduce_v4` and `_w8a8_v4` receive the packed
  dim3 in a register, so their grid is runtime-computed. The construction `blockDim = gridDim +
  256` in the low dword means a grid of 256 would give 512 threads, matching every other kernel,
  but the value itself is not static.
- The meaning of the second template parameter (8 or 10) and the MoE tiling parameter (9/12/18/24/36
  against 4/6/8/12). The former scales an internal bound by two and leaves the L2-writeback guard
  at 8; the latter is exactly 3x between the W8A16 and W8A8 forms. Neither is identified.
- Everything timing. This is a static read of a library that was never loaded. No kernel duration,
  no occupancy achieved, no bandwidth, no breakdown of which of the 32 stages dominates. The
  stage-level `s_memrealtime` instrumentation the kernel already carries would give exactly that
  from one run on an MI355X, writing into the Args-supplied `unsigned long*`.
- The CUDA build is not comparable. The extraction's README records that this library and
  `tilert-glm5-20261014-COk05W` are different source trees (`tilert::ops::glm_5_2` and
  `tilert::ops::cell::*` here against `tilert::lib::glm_5` and
  `tilert::core::executor::piped_prefetch*` there), so none of the above transfers to the CUDA
  path by name.
