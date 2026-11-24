set -e
source /opt/ascend/cann-9.2.0/set_env.sh
FFN_DOWN_N1024_PARENT=/workspace/tests/glm52/whole_layer/ffn_down_n512/build-v2 python3 -m unittest discover -s /workspace/tests/glm52/whole_layer/ffn_down_n1024 -p test_prepare.py -v
python3 /workspace/tests/glm52/whole_layer/ffn_down_n1024/build.py /workspace/tests/glm52/whole_layer/ffn_down_n512/build-v2 /workspace/tests/glm52/whole_layer/ffn_down_n1024/build-v2
