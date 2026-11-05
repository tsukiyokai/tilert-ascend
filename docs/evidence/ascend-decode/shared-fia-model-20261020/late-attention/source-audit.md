# Local attention source alignment

The executed probe SHA256 is
`e32769d7f379cb772caba412c0097ab53fa02af6b52257bc61fe385012a68426`.
The CPU auditor rehashes all 10 recorded Python sources before reading raw
tensors. `device-result.json` retains their full paths and hashes, the
weight-slice hashes and the original M1 input hashes.

- Installed `ops/layernorm.py:63–67/:90–97` creates and passes the norm bias.
  Probe 227–228 supplies BF16 zero bias, matching the layer-major loader's
  synthesized zero parameter under BF16 default dtype.
- `cann_model.py:461–496` makes one M32 request with positions/slots 8192–8223,
  query cumulative length [32], KV length [8224] and ChunkedPrefill state.
  Probe 173–177/232–233 supplies those consumed vectors. Indexer builder
  `attention/indexer.py:1081–1103` sets non-PCP num_decode_tokens=0, as does
  the probe. PCP and DSA-CP are disabled and asserted.
- Installed `sfa_v1.py:1089–1120` prepares Q-a/KV-a/Q-b NZ matrices and
  per-channel dequant scales. Probe 38–82 maps rank 0's four query heads and
  the corresponding KV-b/V-up slices. Its native Prolog method supplies
  query/weight/cache quant modes 0/2/0 and QC/QR/KC scales 1.
- Actual checkpoint qk 192+64 gives scale 1/16. Default 64-dimension RoPE uses
  theta 8000000 and interleaved order; there is no Yarn multiplier.
  `deepseek_v2.py:1121–1135` and the probe's actual Rotary class agree.
- `deepseek_v2.py:1143–1167` with index offset 3/frequency 4 creates producers
  66/70 and static consumers 67–69/71–73. `indexer.py:192–201/:233–260` handles
  C8 INT8 keys, FP16 scales and Hadamard at 66; 70 uses BF16. Main MLA KV is
  BF16 in both cases. All new field bytes independently match M1.
- `cann_attention.py:90–95` refreshes producer masks and reuses them in
  consumers. The probe asserts object lifetime, while the raw audit checks
  consumer selections and equality with per-layer rebuilt FIA. All restored
  prefix values are [finite](prefix-finite.json); unused capacity is zeroed.

PrologV3 supports all attention states (`sfa_v1.py:1932–1936`), and the base
`_prepare_indexer_metadata:1571–1573` is a no-op. The probe calls the local
stages directly through `_v_up_proj`. GLM supplies no `g_proj`; no output
gate is omitted. `_finalize_o_proj:1779` is never called.

Python source and tensor identities are recorded. This launch did not save
a per-run device-ELF-open or tiling-key receipt.
