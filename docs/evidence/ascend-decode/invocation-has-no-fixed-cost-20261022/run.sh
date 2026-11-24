#!/bin/bash
# The same columns against a shrinking reduction. A fixed cost shows as the
# intercept; walking the reduction shows as the slope.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
for depth in 256 512 1024 2048 4096 6144; do
    for arm in fused split; do
        python3 -I /tree/tests/glm52/megakernel/check.py \
            "/out/build-int8-$arm/megakernel_front.so" \
            --level "depth$depth" --device npu:0 --rows 12 --inputs 2 --repeats 5 \
            > "/out/depth-$arm-$depth.json" 2>"/out/depth-$arm-$depth.log" \
            || { echo "FAILED $arm depth=$depth"; tail -2 "/out/depth-$arm-$depth.log"; }
    done
done
echo done
