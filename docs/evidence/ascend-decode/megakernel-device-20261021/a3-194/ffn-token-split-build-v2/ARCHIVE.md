# Token-owned input-split build-v2

This archive retains all 88 frozen source files, compile commands and logs, manifests, CPU checks and the device handoff. Object files and the shared library are omitted; their hashes remain in `manifest.json` and `identity.sha256`. The complete build, including all six compiled artifacts, was independently verified on a3-194 at `/home/shanshan/tilert-megakernel-20261021/tests/glm52/whole_layer/ffn_token_split/build-v2`.

`verify_cpu.py` verifies the complete build's hashes, stages its exact portable tests with both new headers, and runs the eight contracts. The frozen x86 and AArch64 logs passed. The first x86 staging attempt used a directory shallower than the inherited test's repository-relative default; that failure log is retained. Correct staging changed no compiled source. CANN compilation and Torch schema loading are CPU evidence; device correctness and latency remain untested.
