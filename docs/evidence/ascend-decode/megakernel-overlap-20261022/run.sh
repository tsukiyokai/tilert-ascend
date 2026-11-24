#!/bin/bash
# Sweep the AIV workload under the cube. The point where the total starts to
# rise is how much vector work the cube hides for free.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
library=/out/build-int8-fused/megakernel_front.so
for passes in 0 1 2 3 4 6 8 16; do
    python3 -I /tree/tests/glm52/megakernel/check.py "$library" \
        --level prolog --device npu:0 --rows 12 --inputs 2 --repeats 5 \
        --vector-passes "$passes" \
        > "/out/overlap-$passes.json" 2>"/out/overlap-$passes.log" \
        || { echo "FAILED passes=$passes"; tail -3 "/out/overlap-$passes.log"; }
done
echo done
