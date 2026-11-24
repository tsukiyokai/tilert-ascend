# A3 HIXL handoff checks

Environment: `a3-gs:trt-asc`, CANN 9.2.0, Torch/Torch-NPU 2.12.0. The
initial transport bridge SHA256 is
`cee498b49da610c77a50d00de200b7bc4ddf1c1304bf6a3dea19cb6f870864f5`.

| Check | Observation |
| --- | --- |
| Independent processes, die 0 → die 0 | [Connect status 103900](same-die.json); receive allocation remains unchanged. |
| Independent processes, die 0 → die 1 | [1 KiB WRITE passes](cross-die.json); all receiver bytes equal the source pattern. |
| Two processes per die, 0 → 1 held while 1 → 0 runs | [Both 1 KiB WRITEs pass](hixl-listen-port-20261017.json); four explicit internal listener ports are observed. |
| TP16, original 8192-token prompt | [All 32 prefix files verified](prefix-8k-20261017.json): main 8192 rows, MTP 8191 rows, matching parent hidden and first token 6500. |
| Production host command channel | [16 real Gloo processes](gloo-idle.json) receive all four commands in order after 141 seconds of idle time; none imports Torch-NPU. |
| TP16 metadata and admission | [30 PD checks](check-shifted-pd.json), [29 connector checks](check-shifted-connector.json), and [20 session checks](check-session.json) pass on CPU. |

Run [check_transport.py](../../../../tests/hixl/check_transport.py) with
`--sender-device 0 --receiver-device 0` or `1`; each JSON records the full
command, source hash, library hash and receiver result. The installed HIXL
`comm` engine rejects self-device connections before `TransferSync`:
[original errors](hixl-self-device-20261017/pid-30305-connect-error.log),
[engine selection](hixl-self-device-20261017/pid-30305-engine-selection.log),
and [versioned source audit](../hixl-self-device-20261017.json).

The receiver explicitly selects physical die `(LOCAL_RANK+1)%16`; its
logical rank and weight shard remain unchanged. Source rank r writes
destination rank r. Visible device order stays unchanged.

The request command channel uses a separate CPU Gloo group.
[check_control_channel.py](../../../../tests/glm52/check_control_channel.py)
executes its actual object broadcasts with CPU Gloo as the default group;
the next device run also checks coexistence with the HCCL computation group.
The earlier
[8K attempt](prefix-8k-20261017-attempt-02/inputs.json) waited
[141.05 seconds](prefix-8k-20261017-attempt-02/control-idle-check.json), then
failed at the HIXL self-device connection. Its
[receiver log](prefix-8k-20261017-attempt-02/receiver.log) and
[client error](prefix-8k-20261017-attempt-02/client.log) preserve that run.

The next 8K attempt used the shifted device mapping. An external job then
occupied all sixteen dies before prefill initialization completed:
[occupancy snapshot](prefix-8k-20261017/occupancy-conflict.json).
Only this experiment's passive receiver was stopped;
[the external processes remain](prefix-8k-20261017/occupancy-after-stop.json).
That run did not complete 8K transfer or decode continuation.

With shifted placement, P and D share each die. Their default internal
HCCL port 16666 conflicts even when HIXL endpoint ports differ. The bridge
now sets `GlobalResourceConfig` with
`{"comm_resource_config.listen_port":"<endpoint port>"}` at initialization.
The [four-process check](hixl-listen-port-20261017.json) observes ports
29200–29203 and exact bytes in both directions while the first pair remains
alive. All four processes exit. The rebuilt bridge is byte-identical to the
tested library, SHA256
`9d1421f2033d1ee0e9d11325e7d5bdd34aa5bbc78d50c42a80079af305c3ea5d`.
This check covers co-resident transport; the TP16 8K capture is separate.

The subsequent TP16 attempt initializes its communicators, then fails
exporting a non-page-aligned Torch storage through IPC. The
[allocation checks](hixl-allocation-20261017.json) reproduce this with a
7,897,088-byte suballocation under both allocator modes. Registering its
actual 20 MiB allocator block succeeds. The production transport now
resolves non-expandable blocks through `torch.npu.memory_snapshot()`;
the driver returns 207000 for `aclrtMemGetAddressRange`.
Four device checks pass: exact payload, logical metadata, one physical
registration/unregistration for two logical regions, and rejection of an
unregistered source before native WRITE. Set
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:False` before Torch initialization.

The subsequent [completed 8K capture](prefix-8k-20261017.json) contains all
sixteen rank receipts for request `ba210bea41854af3`. The
[live capture provenance](capture-provenance.json) binds it to the actual
producer: on 2026-10-17 at 14:20:20 Asia/Shanghai, its service log records
all sixteen `sent` acknowledgements, each covering 775,170,688 bytes,
8192 main rows and 8191 MTP rows. The frozen connector logs `sent` only
after synchronous HIXL WRITE and the matching receiver acknowledgement.
The same request's live vLLM response reports 8192 prompt tokens and first
token 6500; the capture response and manifest agree.

File sizes and SHA256 values match all 32 payloads. The normalized,
unrotated BF16 parent hidden at position 8191 is bitwise equal across ranks.
The frozen case is at
`/workspace/offline/prefix8k-allocation-20261017/capture/case-00`;
the producer and receiver have exited. The capture receiver loads no decode
weights and executes zero decode steps. Separate numerical and internal
timing runs consume these files. A subsequent
[live PD check](../pd-live-20261017/README.md) completes a fresh 8K prefill,
sixteen HIXL WRITEs, seven CANN decode steps and an eight-token HTTP
response. It keeps device KV alive while releasing P and loading D;
the original capture remains the input to separate numerical runs.
