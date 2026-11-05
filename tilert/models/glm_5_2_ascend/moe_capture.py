"""Freeze one layer's MoE block, inputs and outputs, for offline work.

The block is one custom op: `ascend_moe_forward_shared_sp` takes the
unpermuted hidden states and the router logits and returns the shared and
routed outputs, so everything between -- the gating, the permutation, both
sets of experts, the combine, and the shared path's reduce-scatter -- is
inside a boundary a single substitution controls. `AscendMoERunner` holds it
in `_forward_entry`, an instance attribute, so replacing it is one assignment
a layer.

Which is the reason to capture rather than to write the replacement blind.
Reproducing that block has to match the numerics gate, and twice today a
kernel-level change written without a fixture had to be reverted. One short
run with this installed leaves a file that the replacement can be developed
against on a machine with no free dies, or none at all.
"""
import pathlib

CAPTURE_DIR = pathlib.Path("/workspace/experiments/moe-capture")


def install(runners, directory=CAPTURE_DIR, layers=(3,)):
    """Record `_forward_entry`'s arguments and results for `layers`.

    `runners` are the `AscendMoERunner` objects, in layer order. Only the
    layers asked for are written, because one is enough to develop against
    and each is a few megabytes of BF16.
    """
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    undo = []
    for index, runner in enumerate(runners):
        if index not in layers:
            continue
        original = runner._forward_entry

        def recording(*arguments, _original=original, _index=index, **named):
            results = _original(*arguments, **named)
            held = {
                "layer": _index,
                "arguments": [
                    argument.detach().cpu() if torch.is_tensor(argument)
                    else argument for argument in arguments
                ],
                "named": {
                    key: value.detach().cpu() if torch.is_tensor(value)
                    else value for key, value in named.items()
                },
                "results": [
                    result.detach().cpu() if torch.is_tensor(result)
                    else result for result in results
                ],
            }
            torch.save(held, directory / f"layer-{_index}.pt")
            return results

        runner._forward_entry = recording
        undo.append((runner, original))

    def restore():
        for runner, original in undo:
            runner._forward_entry = original

    return restore
