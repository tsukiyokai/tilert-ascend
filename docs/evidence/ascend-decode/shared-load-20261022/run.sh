#!/bin/bash
# N vector operations sharing one load, against N separate loads doing one each.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
library=/out/build-int8-fused/megakernel_front.so
for ops in 1 2 4 8; do
    for fused in 1 0; do
        python3 -I /tree/tests/glm52/megakernel/check.py "$library" \
            --level prolog --device npu:0 --rows 12 --inputs 2 --repeats 5 \
            --vector-passes 2 --vector-ops "$ops" --fused-ops "$fused" \
            > "/out/ops-$ops-f$fused.json" 2>"/out/ops-$ops-f$fused.log" \
            || { echo "FAILED ops=$ops fused=$fused"; tail -3 "/out/ops-$ops-f$fused.log"; }
    done
done
echo done
