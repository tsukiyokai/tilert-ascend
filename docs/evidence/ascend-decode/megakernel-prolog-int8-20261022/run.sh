#!/bin/bash
# Both dtypes, both arms, three levels, alternating.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
for dtype in int8 bf16; do
    for level in prolog query_norm; do
        for pass in 1 2; do
            for arm in fused split; do
                library="/out/build-$dtype-$arm/megakernel_front.so"
                test -f "$library" || { echo "MISSING $library"; continue; }
                python3 -I /tree/tests/glm52/megakernel/check.py "$library" \
                    --level "$level" --device npu:0 --rows 12 \
                    --inputs 8 --repeats 5 \
                    > "/out/dtype-$dtype-$level-$arm-p$pass.json" \
                    2>"/out/dtype-$dtype-$level-$arm-p$pass.log" \
                    || { echo "FAILED $dtype $level $arm"; tail -3 "/out/dtype-$dtype-$level-$arm-p$pass.log"; }
            done
        done
    done
done
echo done
