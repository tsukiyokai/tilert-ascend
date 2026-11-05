"""Check actual runtime imports, schemas and GLM layer selection without an NPU."""

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu

    if list(Path("/dev").glob("davinci*")) or torch.npu.is_initialized():
        raise RuntimeError("The CPU gate requires no exposed or initialized NPU")

    def refuse_init(*unused_args, **unused_kwargs):
        raise RuntimeError("The CPU gate refuses NPU initialization")

    torch.npu._lazy_init = refuse_init
    torch_npu._C._npu_init = refuse_init
    report = {
        "scope": "CPU imports and co-load; no model forward or TP16 collective",
        "python": sys.version, "torch": torch.__version__,
        "torch_npu": torch_npu.__version__, "torch_path": torch.__file__,
        "torch_npu_path": torch_npu.__file__, "libraries": [], "completed": False,
    }
    try:
        for name, basename in (
                ("indexed", "whole_layer_indexed_attention.so"),
                ("shared", "whole_layer_full_attention.so"),
                ("sfa", "vendor_sfa_adapter.so")):
            path = args.runtime / "native" / name / basename
            torch.ops.load_library(str(path))
            report["libraries"].append({
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        report["schemas"] = {
            name: str(getattr(torch.ops.tilert, name).default._schema)
            for name in ("indexed_attention_prepare", "indexed_attention_out",
                         "shared_full_attention_prepare", "shared_full_attention_out")
        }
        report["sfa_schema"] = str(
            torch.ops._C_ascend.npu_sparse_flash_attention.default._schema)
        from vllm.platforms import current_platform
        current_platform.pre_register_and_update()
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        from transformers import GlmMoeDsaConfig
        from vendor_hook_glm52_config import model_config, selection

        report["classes"] = {}
        for cls in (DeepseekV2DecoderLayer, AscendSFAImpl, ACLGraphWrapper,
                    GlmMoeDsaConfig):
            path = Path(inspect.getfile(cls)).resolve()
            report["classes"][cls.__name__] = {
                "module": cls.__module__, "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        config = GlmMoeDsaConfig(**model_config())
        report["layer_selection"] = [selection(6), selection(7)]
        report["actual_config_selection"] = {
            "indexer_types": config.indexer_types,
            "index_topk_freq": config.index_topk_freq,
            "index_skip_topk_offset": config.index_skip_topk_offset,
            "index_topk_pattern": config.index_topk_pattern,
        }
        if (config.indexer_types[6:8] != ["full", "shared"]
                or config.indexer_types.count("full") != 21):
            raise RuntimeError("The real Transformers configuration changed layer selection")
        spec = importlib.util.find_spec("vllm_ascend.vllm_ascend_C")
        report["full_vendor_extension"] = None if spec is None else spec.origin
        report["norm_schema_available"] = hasattr(
            torch.ops._C_ascend, "npu_add_rms_norm_bias")
        report["completed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["npu_initialized"] = torch.npu.is_initialized()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return int(not report["completed"] or report["npu_initialized"])


if __name__ == "__main__":
    raise SystemExit(main())
