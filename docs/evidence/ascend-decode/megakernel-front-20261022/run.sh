#!/bin/bash
# Alternate the arms at every width so drift cannot favour either.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
for rows in 1 2 12 32; do
    for pass in 1 2; do
        for arm in fused split; do
            python3 -I /tree/tests/glm52/megakernel/check.py \
                "/tree/tests/glm52/megakernel/build-$arm/megakernel_front.so" \
                --device npu:0 --rows "$rows" --inputs 8 --repeats 5 \
                > "/out/front-$arm-m$rows-p$pass.json" 2>/dev/null \
                || echo "FAILED $arm rows=$rows pass=$pass"
        done
    done
done
echo done
