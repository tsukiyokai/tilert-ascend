"""Run the real eager TP CPU gate, then graph ownership contracts and imports."""

import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import sys
import unittest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import tp_cpu_check

    tp_cpu_check.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    eager_bytes = output.read_bytes()
    output.with_name("eager-cpu-report.json").write_bytes(eager_bytes)
    report = json.loads(eager_bytes)
    import torch
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.compilation.updatable_graph import UpdatableGraph
    import tp_graph_constructor

    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module("test_tp_graph")))
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    if torch.npu.is_initialized():
        raise RuntimeError("The graph CPU gate initialized an NPU")
    actual = (ACLGraphWrapper, UpdatableGraph, torch.npu.graph,
              tp_graph_constructor.run_device_rank)
    report.update(scope="Official ACL graph imports and CPU stream/lifetime contracts; real eager TP CPU nonmember gate; no capture or NPU execution",
                  graph_contracts_run=result.testsRun, graph_contract_log=log.getvalue(),
                  graph_sources={inspect.getfile(value): digest(inspect.getfile(value)) for value in actual},
                  graph_device_execution=False,
                  graph_script_sha256=digest(Path(__file__)))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"graph_contracts_run": result.testsRun,
                      "npu_initialized_after": False, "pass": True}), flush=True)


if __name__ == "__main__":
    main()
