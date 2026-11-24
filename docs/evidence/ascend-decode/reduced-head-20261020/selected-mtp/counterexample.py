"""FP64 scalar norm/head example; this does not simulate a CANN kernel."""
import json
import math
from pathlib import Path


def norm(row):
    scale = math.sqrt(sum(x * x for x in row) / 2 + 1e-5)
    return [row[0] / scale, 2 * row[1] / scale]


once = norm([1.0, 1.0])
twice = norm(once)
correct = [3 * once[0], once[1]]
wrong = [3 * twice[0], twice[1]]
assert correct[0] > correct[1] and wrong[0] < wrong[1]
result = {'scope': 'Scalar semantic counterexample only', 'pre_norm': [1, 1],
          'norm_weight': [1, 2], 'head_rows': [[3, 0], [0, 1]],
          'correct_logits': correct, 'double_norm_logits': wrong,
          'correct_top1': 0, 'double_norm_top1': 1}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
