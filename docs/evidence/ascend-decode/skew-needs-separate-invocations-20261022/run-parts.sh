#!/bin/bash
# The split build makes one cube invocation a part either way, so this isolates
# the schedule from the invocation count. Zero steps runs the halves one after
# the other; four steps interleaves the vector half between the four parts.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/runtime/vendor-hook-pinned/vllm-a982-py311-compatible:/runtime/vendor-hook-deps/python
library=/out/build-int8-split/megakernel_front.so
for passes in 2 4 8; do
    for steps in 0 2 4; do
        python3 -I /tree/tests/glm52/megakernel/check.py "$library" \
            --level hidden --device npu:0 --rows 12 --inputs 2 --repeats 5 \
            --vector-passes "$passes" --skew-steps "$steps" \
            > "/out/parts-p$passes-s$steps.json" 2>"/out/parts-p$passes-s$steps.log" \
            || { echo "FAILED passes=$passes steps=$steps"; tail -3 "/out/parts-p$passes-s$steps.log"; }
    done
done
echo done
