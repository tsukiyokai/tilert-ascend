# Callable routed/shared body

The source generator extracts the selected BF16 routed/shared implementation
into `routed_body::Forward(TPipe&, ...)`. Its numerical statements, projection
tilings, row ownership, scratch layout and six internal barriers are preserved.
The standalone wrapper retains device initialization and one local `TPipe`;
its independent Torch namespace is `tilert_pipeline_body`.

A containing kernel owns `InitSocState`, the synchronization base, system
workspace and pipe. It must retire any earlier UB allocation before calling
the body. The body returns after issuing its final output stores. Before a
later phase consumes those outputs or reuses UB, every core must drain its
pipelines, publish stores, join a mixed-core barrier and reset the pipe.
There is no implicit final barrier inside the selected non-profiled body.

Build from the frozen parent, whose kernel SHA is checked before extraction:

```bash
bash tests/glm52/whole_layer/routed_body/build.sh \
  tests/glm52/whole_layer/routed_body/build-v1 \
  tests/glm52/routed_int4/build-pipeline-front
```

The standalone validation compares the extracted entry with the selected
parent through `check_pipeline_front.py`, using
`--native-namespace tilert_pipeline_body --baseline-namespace tilert_pipeline_front
--baseline-input bf16`. Device byte comparisons are required before composing
this entry with normalization, routing and the FFN output addition. The
extraction alone does not validate a whole layer or establish a latency gain.
