# CPU commands already executed successfully; no NPU commands.
cd /tmp/trt-carver-ci-1955-20261029
python -B /tmp/trt-carver-1955-cpu-check.py
/tmp/trt-carver-ruff017/bin/ruff check testing/python/carver/test_ascend_specs.py
/tmp/trt-carver-ruff017/bin/ruff format --check testing/python/carver/test_ascend_specs.py
git diff --check
