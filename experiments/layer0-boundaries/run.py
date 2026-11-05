"""Observe one real TP16 layer without loading the other 77 layers."""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Boundaries:
    def __init__(self):
        self.values = {}
        self.metadata = {}
        self.handles = []
        self.patches = []
        self.restores = []
        self.stage = None

    def save(self, name, value):
        import torch

        if name in self.values or not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected one tensor at {name}")
        self.values[name] = value.detach().clone()

    def install(self, layer):
        for name, module in (
            ("input_norm", layer.input_layernorm),
            ("post_norm", layer.post_attention_layernorm),
            ("gate_up", layer.mlp.gate_up_proj),
            ("down", layer.mlp.down_proj),
            ("o_proj", layer.self_attn.o_proj),
        ):
            self.metadata[name] = {
                "class": type(module).__module__ + "." + type(module).__name__,
                **{
                    key: getattr(module, key, None)
                    for key in (
                        "variance_epsilon",
                        "tp_rank",
                        "tp_size",
                        "input_size_per_partition",
                        "output_size_per_partition",
                        "gather_output",
                        "input_is_parallel",
                    )
                },
            }

        impl = layer.self_attn.mla_attn.mla_attn.impl
        original_sfa = impl._execute_sparse_flash_attention_process
        source = Path(inspect.getfile(original_sfa)).resolve()
        self.metadata["sfa"] = {
            "class": type(impl).__module__ + "." + type(impl).__name__,
            "scale": impl.scale,
            "source_path": str(source),
            "source_sha256": digest(source),
        }

        def sfa(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table=None,
        ):
            table = (
                attn_metadata.block_table
                if block_table is None
                else block_table
            )
            for name, value in (
                ("query_nope", ql_nope),
                ("query_rope", q_pe),
                ("key_cache", kv_cache[0]),
                ("rope_cache", kv_cache[1]),
                ("indices", topk_indices),
                ("block_table", table),
                ("query_lengths", actual_seq_lengths_query),
                ("key_lengths", actual_seq_lengths_key),
            ):
                self.save("sfa." + name, value)
            output = original_sfa(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table=block_table,
            )
            self.save("sfa.output", output)
            return output

        replacement = patch.object(
            impl, "_execute_sparse_flash_attention_process", sfa
        )
        replacement.start()
        self.patches.append(replacement)

        def input_norm(module, inputs, output):
            normalized = output[0] if isinstance(output, tuple) else output
            self.save("input_norm", normalized)

        def o_before(module, inputs):
            self.stage = "o_proj"
            self.save("o_proj.input", inputs[0])

        def o_after(module, inputs, output):
            self.save("o_proj.output", output[0])
            self.stage = None

        def attention(module, inputs, output):
            self.save("attention.output", output)

        def norm_before(module, inputs):
            self.save("post_norm.input", inputs[0])
            self.save("post_norm.residual_input", inputs[1])

        def norm_after(module, inputs, output):
            self.save("post_norm.normalized", output[0])
            self.save("post_norm.retained", output[1])

        def gate_up(module, inputs, output):
            self.save("mlp.gate_up", output[0])

        def down_before(module, inputs):
            self.stage = "down"
            self.save("mlp.down_input", inputs[0])

        def down_after(module, inputs, output):
            self.save("mlp.down_output", output[0])
            self.stage = None

        def mlp(module, inputs, output):
            self.save("mlp.output", output)

        self.handles = [
            layer.input_layernorm.register_forward_hook(input_norm),
            layer.self_attn.o_proj.register_forward_pre_hook(o_before),
            layer.self_attn.o_proj.register_forward_hook(o_after),
            layer.self_attn.register_forward_hook(attention),
            layer.post_attention_layernorm.register_forward_pre_hook(
                norm_before
            ),
            layer.post_attention_layernorm.register_forward_hook(norm_after),
            layer.mlp.gate_up_proj.register_forward_hook(gate_up),
            layer.mlp.down_proj.register_forward_pre_hook(down_before),
            layer.mlp.down_proj.register_forward_hook(down_after),
            layer.mlp.register_forward_hook(mlp),
        ]

    def reduce(self, original, value):
        if self.stage not in ("o_proj", "down"):
            raise ValueError("Unattributed TP reduction")
        self.save(self.stage + ".tp_input", value)
        output = original(value)
        self.save(self.stage + ".tp_output", output)
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for replacement in reversed(self.patches):
            replacement.stop()
        self.patches.clear()
        for restore in reversed(self.restores):
            restore()
        self.restores.clear()


def instrument(
    original, engine, factory, *args, attention_override=None, **kwargs
):
    observer = Boundaries()
    original_reduce = engine._tp_sum_original

    def observed_factory(*factory_args, **factory_kwargs):
        layer = factory(*factory_args, **factory_kwargs)
        if attention_override is not None:
            impl = layer.self_attn.mla_attn.mla_attn.impl
            observer.restores.append(attention_override(impl))
        observer.install(layer)
        return layer

    try:
        with patch.object(
            engine,
            "_tp_sum_original",
            lambda value: observer.reduce(original_reduce, value),
        ):
            result = original(engine, observed_factory, *args, **kwargs)
        expected = {
            "input_norm",
            "o_proj.input",
            "o_proj.output",
            "attention.output",
            "post_norm.input",
            "post_norm.residual_input",
            "post_norm.normalized",
            "post_norm.retained",
            "mlp.gate_up",
            "mlp.down_input",
            "mlp.down_output",
            "mlp.output",
            "o_proj.tp_input",
            "o_proj.tp_output",
            "down.tp_input",
            "down.tp_output",
            "sfa.query_nope",
            "sfa.query_rope",
            "sfa.key_cache",
            "sfa.rope_cache",
            "sfa.indices",
            "sfa.block_table",
            "sfa.query_lengths",
            "sfa.key_lengths",
            "sfa.output",
        }
        state, data, checks, coverage, sources, references = result
        checks.append(
            {
                "name": "boundary_fields",
                "passed": set(observer.values) == expected,
            }
        )
        data["boundaries"] = {
            name: value.cpu().clone()
            for name, value in observer.values.items()
        }
        data["boundary_metadata"] = observer.metadata
        sources[str(Path(__file__).resolve())] = digest(__file__)
        source = Path(inspect.getfile(original)).resolve()
        sources[str(source)] = digest(source)
        return state, data, checks, coverage, sources, references
    finally:
        observer.close()
        observer.values.clear()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-inputs-only", action="store_true")
    parser.add_argument("--expanded-value", action="store_true")
    options, remaining = parser.parse_known_args()
    sys.path[:0] = [str(options.source), str(options.source / "tests/glm52")]
    legacy = importlib.import_module("run_cann_layer_major")
    with patch.object(sys, "argv", [sys.argv[0], *remaining]):
        args = legacy.arguments()
    if not (1 <= args.layers <= 3 and args.steps == args.query_width == 1):
        parser.error(
            "This observer requires --layers 1..3 --steps 1 --query-width 1"
        )
    if args.baseline is None:
        parser.error(
            "The original full CANN capture is required as --baseline"
        )
    if options.check_inputs_only:
        from tilert.pd_vllm.ascend_cache import checkpoint_fingerprint

        rank = int(os.environ.get("RANK", "0"))
        checkpoint = checkpoint_fingerprint(args.checkpoint)
        _, _, inputs = legacy.load_capture(args.capture_case, rank, checkpoint)
        teacher, _, _ = legacy.read_teacher(args, inputs)
        baseline, _ = legacy.read_baseline(
            args, rank, checkpoint, inputs, teacher
        )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "inputs_verified": len(baseline) == 1,
                    "npu_executed": False,
                    "script_sha256": digest(__file__),
                }
            )
        )
        return 0
    original = legacy.run_layer
    attention_override = None
    if options.expanded_value:
        from expanded_attention import install

        attention_override = install

    def observed(*args, **kwargs):
        return instrument(
            original, *args, attention_override=attention_override, **kwargs
        )

    with patch.object(legacy, "run_layer", observed):
        return legacy.execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
