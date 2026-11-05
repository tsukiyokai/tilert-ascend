#!/bin/bash
# Zero steps runs the halves one after the other, which is what a phase barrier
# gives. More steps interleave them inside one barrier region a step.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
library=/out/build-int8-fused/megakernel_front.so
for passes in 2 4; do
    for steps in 0 1 2 4 8; do
        python3 -I /tree/tests/glm52/megakernel/check.py "$library" \
            --level prolog --device npu:0 --rows 12 --inputs 2 --repeats 5 \
            --vector-passes "$passes" --skew-steps "$steps" \
            > "/out/skew-p$passes-s$steps.json" 2>"/out/skew-p$passes-s$steps.log" \
            || { echo "FAILED passes=$passes steps=$steps"; tail -3 "/out/skew-p$passes-s$steps.log"; }
    done
done
echo done
