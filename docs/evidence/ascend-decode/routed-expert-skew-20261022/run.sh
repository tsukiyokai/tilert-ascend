#!/bin/bash
# The unskewed kernel against two and four expert chunks, alternating.
set -u
source /opt/ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/tree/tests/glm52/routed_int4:/tree:$PYTHONPATH
export XDG_CACHE_HOME=/tmp/cache
cd /tree
for pass in 1 2; do
    for arm in off 2 4; do
        timeout 300 python3 tests/glm52/routed_int4/bench_weight_banks.py \
            --library "/out/skew-$arm/routed_int4_torch.so" \
            --native-schedule row --device npu:0 --rows 12 --blocks 24 \
            --banks 2 --calls 4 --iterations 10 --timing-pairs 3 \
            --output "/out/rskew-$arm-p$pass.json" > /dev/null 2>"/out/rskew-$arm-p$pass.log" \
            || { echo "FAILED $arm pass=$pass"; tail -3 "/out/rskew-$arm-p$pass.log"; }
    done
done
echo done
