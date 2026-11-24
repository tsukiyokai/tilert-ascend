"""Check the timing fixture against real Torch212 imports and six native DSOs."""

import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import tp_graph_cpu_check

    tp_graph_cpu_check.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    raw = output.read_bytes()
    output.with_name("graph-cpu-report.json").write_bytes(raw)
    report = json.loads(raw)
    import torch
    from tp_timing_native import NORM_SHA
    import tp_timing_constructor

    directory = Path(__file__).parent
    library = directory / "native_norm.so"
    if digest(library) != NORM_SHA:
        raise RuntimeError("The norm DSO differs from the timing parent")
    torch.ops.load_library(str(library))
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module("test_tp_timing")))
    if not result.wasSuccessful():
        raise RuntimeError(log.getvalue())
    runtime = Path(sys.argv[sys.argv.index("--runtime") + 1])
    indexed = Path(sys.argv[sys.argv.index("--indexed") + 1])
    shared = Path(sys.argv[sys.argv.index("--shared") + 1])
    command = [sys.executable, str(directory / "tp_timing_device.py"),
               "--runtime", str(runtime), "--output", "/tmp/timing-not-started",
               "--extension-sha256", "unused", "--physical-devices", "6", "7",
               "--adapter", str(runtime / "tilert/models/glm_5_2_ascend/attention_native.py"),
               "--indexed-library", str(indexed), "--shared-library", str(shared),
               "--sfa-library", str(runtime / "native/sfa/vendor_sfa_adapter.so")]
    controls = []
    for extra, expected, text in ((["--help"], 0, "--forward-replays"),
                                  (["--forward-widths", "1"], 2, "Timing requires M12"),
                                  (["--forward-replays", "2"], 2, "eight changing inputs")):
        run = subprocess.run(command + extra, text=True, capture_output=True, check=False)
        if run.returncode != expected or text not in run.stdout + run.stderr:
            raise RuntimeError(f"Unexpected timing CLI result: {run}")
        controls.append({"arguments": extra, "exit_code": run.returncode,
                         "stdout": run.stdout, "stderr": run.stderr})
    shell = subprocess.run(["bash", "-n", str(directory / "device_tp_timing_run.sh")],
                           capture_output=True, text=True, check=True)
    if torch.npu.is_initialized() or Path("/tmp/timing-not-started").exists():
        raise RuntimeError("The timing CPU gate entered a device worker")
    report["libraries"][str(library)] = digest(library)
    report.update(scope="Actual Torch212 six-DSO co-load, source, rank-pairing and ownership contracts; no NPU or HCCL execution",
                  timing_contracts_run=result.testsRun, timing_contract_log=log.getvalue(),
                  timing_cli_controls=controls, timing_shell_exit=shell.returncode,
                  timing_constructor_sha256=digest(inspect.getfile(tp_timing_constructor)),
                  norm_schema=str(torch.ops.tilert.norm_quant_out.default._schema),
                  timing_device_execution=False, timing_script_sha256=digest(__file__))
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"timing_contracts_run": result.testsRun,
                      "npu_initialized_after": False, "pass": True}), flush=True)


if __name__ == "__main__":
    main()
