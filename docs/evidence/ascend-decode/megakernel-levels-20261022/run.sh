#!/bin/bash
# Both arms, both dependency levels, alternating, three widths.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
for arm in fused split; do
    test -f "/out/build-$arm/megakernel_front.so" || { echo "MISSING $arm"; exit 2; }
done
for level in hidden query_norm; do
    for rows in 1 12 32; do
        for pass in 1 2; do
            for arm in fused split; do
                python3 -I /tree/tests/glm52/megakernel/check.py \
                    "/out/build-$arm/megakernel_front.so" \
                    --level "$level" --device npu:0 --rows "$rows" \
                    --inputs 8 --repeats 5 \
                    > "/out/level-$level-$arm-m$rows-p$pass.json" \
                    2>"/out/level-$level-$arm-m$rows-p$pass.log" \
                    || { echo "FAILED $level $arm rows=$rows"; tail -3 "/out/level-$level-$arm-m$rows-p$pass.log"; }
            done
        done
    done
done
echo done
