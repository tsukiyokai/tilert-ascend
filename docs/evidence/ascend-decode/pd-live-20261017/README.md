# Live prefill → HIXL → CANN decode → HTTP

On 2026-10-17, `a3-gs:trt-asc` completes one fresh 8192-token request
through the OpenAI-compatible router, TP16 vLLM-Ascend prefill, HIXL and
the TileRT CANN M1 session. The [HTTP response](response.json) returns
eight tokens with status 200; the [client receipt](result.json) records
117.59 seconds including deferred model loading.

Request `83947bb1ed8731f6`, generation one:

- All sixteen producer ranks acknowledge a synchronous HIXL WRITE of
  775,158,400 bytes each: 8192 main rows and 8191 MTP rows, layout v2.
- Prefill returns token 6500. The router forwards the same request ID and
  token to D. D emits it once, then executes seven real CANN decode steps.
- HTTP and callback tokens are
  `[6500,382,262,576,2524,5496,374,7077]`. The main position advances from
  8192 to 8199; the last output remains pending. Reset completes and the
  request slot returns to idle.

The [independent CPU audit](audit.json) binds the actual P/D HTTP calls,
sixteen sender receipts, callback clocks, request state, source overrides,
bridge library and process receipts. This is a functional M1 check;
eight-input teacher accuracy, online MTP and 2 ms performance have separate
acceptance criteria. Output seven is 374 versus 64623 in the older frozen
prefix run; this check does not localize that difference. The
[vLLM baseline](../vllm-baseline-20261017/README.md) separately records the
service's trajectory variation.

## Device-resident handoff across model loading

The first co-resident P/D attempt fails during P warmup: allocation of
386 MiB finds only 13.73 MiB free on NPU 0. Both models use memory
utilization 0.4 and P has a 1 GiB KV cache. Startup fit does not imply
warmup fit.

The successful run starts
[serve_cann_pd.py](../../../../tests/glm52/serve_cann_pd.py) with
`--load-model-after /workspace/offline/pd-live-20261017-02/load-model.json`.
It initializes the TP16 HCCL/Gloo groups and HIXL receive allocations before
loading D weights. The live request reaches inject with views of the
received device storage and waits for that file.

After all sixteen WRITE acknowledgements and the P HTTP response, the
experiment stops its own P process. Its exit receipt is zero at
14:37:14 UTC; no P processes remain when the file is created at 14:37:17.
All ranks then load CannModel inside that same pending inject, copy the
live main KV and continue decode. KV is neither saved nor reloaded from
prefix files. Physical D die `(rank+1)%16` avoids self-device connections;
logical TP shard ownership remains rank.

| Cold request stage | Time |
| --- | --- |
| Prefill HTTP call, including HIXL | 3414.2 ms |
| Inject, including gate wait and D loading | 109179.0 ms |
| Eight-token decode, including first use | 4961.4 ms |

The first decode interval is 3790.4 ms; the following six are 189.5–208.6
ms. These cold intervals are not a steady-state performance result.
HTTP reset and slot release complete before shutdown. D ignores the
observed SIGINT; SIGTERM subsequently terminates its torchrun group.
Graceful deferred-server shutdown remains to be fixed; P and router exit
normally.

## Reproduction artifacts

Server directory: `/home/shanshan/trt-asc/offline/pd-live-20261017-02/`
(`/workspace/offline/pd-live-20261017-02/` inside the container).
`source.json`, actor `launch.json` files, `flow.py`, `router/calls/`,
`timings/`, logs and `audit.py` retain the exact launch and verification.
The failed co-resident run is the sibling `pd-live-20261017-01/`.

Source is `d11505bf6769990aa1b0a870bc5ca3bbea72e2c7` plus two overrides:
deferred server `34c2e0371389` and
[prefill launcher](../../../../tests/glm52/serve_vllm_prefill.sh)
`242bf3b2c45c`. The launcher uses MTP1, a 1 GiB cache, memory utilization
0.4 and the calibrated CANN operator overlay. Both sides use
`expandable_segments:False`; the tested HIXL bridge SHA256 is
`9d1421f2033d1ee0e9d11325e7d5bdd34aa5bbc78d50c42a80079af305c3ea5d`.
