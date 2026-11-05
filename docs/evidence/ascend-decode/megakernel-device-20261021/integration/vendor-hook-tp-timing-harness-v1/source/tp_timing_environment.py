"""Require the recorded project-default HCCL choice before communicator creation."""

import os


NAMES = ("HCCL_OP_EXPANSION_MODE", "HCCL_DETERMINISTIC", "HCCL_ALGO",
         "HCCL_ALGO_CONFIG", "HCCL_BUFFSIZE", "ASCEND_LAUNCH_BLOCKING")


def check_environment(environ=None):
    environ = os.environ if environ is None else environ
    values = {name: environ.get(name) for name in NAMES}
    if (values["HCCL_OP_EXPANSION_MODE"] != "AIV"
            or values["HCCL_DETERMINISTIC"] != "true"):
        raise RuntimeError("Set AIV and HCCL_DETERMINISTIC=true before importing Torch")
    return values
