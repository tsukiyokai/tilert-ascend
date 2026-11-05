#!/bin/bash
# Price a grid barrier. A skewed expert pipeline pays one a step, so this says
# what the schedule costs against the 61 microseconds it would claim.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
library=/out/build-int8-fused/megakernel_front.so
for barriers in 0 1 2 4 8 16 32; do
    python3 -I /tree/tests/glm52/megakernel/check.py "$library" \
        --level prolog --device npu:0 --rows 12 --inputs 2 --repeats 5 \
        --barriers "$barriers" \
        > "/out/barrier-$barriers.json" 2>"/out/barrier-$barriers.log" \
        || { echo "FAILED barriers=$barriers"; tail -3 "/out/barrier-$barriers.log"; }
done
echo done
