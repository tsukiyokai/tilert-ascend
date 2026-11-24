#!/bin/bash
# Both arms at the aligned row stride, alternating.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
for rows in 1 12 32; do
    for pass in 1 2; do
        for arm in fused split; do
            for stride in 3040 3072; do
                python3 -I /tree/tests/glm52/megakernel/check.py \
                    "/tree/tests/glm52/megakernel/build-$arm/megakernel_front.so" \
                    --device npu:0 --rows "$rows" --inputs 8 --repeats 5 \
                    --stride "$stride" \
                    > "/out/aligned-$arm-s$stride-m$rows-p$pass.json" 2>/dev/null \
                    || echo "FAILED $arm $stride rows=$rows"
            done
        done
    done
done
echo done
