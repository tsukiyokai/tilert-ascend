set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
source /opt/ascend/cann-9.2.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export PYTHONPATH=/work/source-v12${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set +e
/work/tools/strace/usr/bin/strace -f -e trace=openat -o /work/receipts/mla-value-order-v6-strace.log python /work/receipts/mla-value-order-v6.py replay --input /work/inputs/mla-value-order-v5.pt --output /work/evidence/mla-value-order-v6 > /work/receipts/mla-value-order-v6-driver.log 2>&1
status=$?
echo "$status" > /work/receipts/mla-value-order-v6-exit.txt
exit "$status"
