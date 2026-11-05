"""Fixed-input target diagnostics, independent of free-running acceptance."""

from contextlib import nullcontext
import hashlib
import json


def load_teacher(path, capture, width, capacity):
    teacher = json.loads(path.read_text())
    manifest = json.loads((capture / "manifest.json").read_text())
    tokens = teacher["teacher_tokens"]
    expected = teacher["expected_next_tokens"]
    # The baseline hashes its JSON prompt file, including the final newline.
    prompt = (json.dumps(manifest["prompt_token_ids"]) + "\n").encode()
    if (teacher["prompt_sha256"] != hashlib.sha256(prompt).hexdigest()
            or teacher["prompt_tokens"] != manifest["seq_len"]
            or teacher["first_token_id"] != manifest["first_token_id"]):
        raise ValueError("Frozen teacher and capture describe different inputs")
    if (len(tokens) != 32 or len(expected) != 32
            or len(tokens) % width
            or tokens[0] != teacher["first_token_id"]
            or tokens[1:] != expected[:-1]
            or any(type(token) is not int or not 0 <= token < 154880
                   for token in tokens + expected)):
        raise ValueError("All 32 teacher inputs and successors are required")
    if manifest["seq_len"] + len(tokens) > capacity:
        raise ValueError("Teacher inputs exceed cache capacity")
    return {**teacher, "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_routes(parts, collector, teacher, width, initial, directory):
    """Save actual selections in teacher-input order, outside timed runs."""
    import torch

    routes = torch.cat(parts, dim=1).transpose(0, 1).contiguous()
    counts = torch.stack([
        torch.stack([torch.bincount(layer.flatten().long(), minlength=256)
                     for layer in block.transpose(0, 1)])
        for block in routes.split(width)
    ])
    path = directory / f"teacher-m{width}-routes.pt"
    torch.save({
        "routes": routes, "first_layer": 3, "initial_position": initial,
        "input_ids": teacher["teacher_tokens"], "width": width,
        "teacher_sha256": teacher["sha256"],
        "source_sha256": collector.source_metadata(),
    }, path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(routes.shape), "first_layer": 3,
        "active_experts": (counts > 0).sum(-1).tolist(),
        "max_rows_per_expert": counts.max(-1).values.tolist(),
    }


def teacher_run(target, teacher, width, directory, observe, *,
                capture_routes=False):
    import torch

    tokens = teacher["teacher_tokens"]
    expected = teacher["expected_next_tokens"]
    initial = target.position
    rows, first_difference, first_artifact = [], None, None
    route_parts, collector = [], None
    if capture_routes:
        from tilert.models.glm_5_2_ascend.cann_choices import CannChoices

        collector = CannChoices(target.model, width)
    for begin in range(0, len(tokens), width):
        ids = tuple(tokens[begin:begin + width])
        with collector if capture_routes else nullcontext():
            result = target.verify(ids, full_logits=True)
            if capture_routes:
                _, selected = collector.finish()
                route_parts.append(torch.stack(selected[3:]).cpu())
        predictions = result.token_ids.tolist()
        observe((begin, ids, predictions, target.position))
        logits = result.logits.detach().float().cpu()
        if (len(predictions) != width or logits.ndim != 2
                or logits.shape[0] != width
                or result.last_hidden.shape[0] != width):
            raise ValueError("Teacher output rows differ from consumed inputs")
        if not torch.isfinite(logits).all():
            torch.save(logits, directory / f"teacher-m{width}-nonfinite.pt")
            raise ValueError("Nonfinite teacher logits")
        if predictions != logits.argmax(dim=-1).tolist():
            raise ValueError("Teacher predictions differ from logits argmax")
        top_values, top_ids = logits.topk(2, dim=-1)
        for row, prediction in enumerate(predictions):
            step = begin + row
            label = expected[step]
            rows.append({
                "step": step, "position": initial + step,
                "input": tokens[step], "expected": label,
                "prediction": prediction,
                "top2_ids": top_ids[row].tolist(),
                "top2_logits": top_values[row].tolist(),
                "expected_logit": float(logits[row, label]),
                "top1_margin": float(top_values[row, 0] - top_values[row, 1]),
                "expected_gap": float(logits[row, prediction]
                                      - logits[row, label]),
            })
            if prediction != label and first_difference is None:
                first_difference = step
                hidden = result.last_hidden[row].detach().cpu().clone()
                path = directory / f"teacher-m{width}-first-difference.pt"
                torch.save({
                    **rows[-1], "logits": logits[row].clone(),
                    "last_hidden": hidden,
                }, path)
                first_artifact = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    return {
        "width": width, "rows": rows, "first_difference": first_difference,
        "first_difference_file": first_artifact,
        "routing": (save_routes(route_parts, collector, teacher, width,
                                initial, directory)
                    if capture_routes else None),
        "initial_position": initial, "final_position": target.position,
        "passed": first_difference is None and len(rows) == len(tokens)
        and target.position == initial + len(tokens),
    }
