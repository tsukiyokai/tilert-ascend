import hashlib


def cache_digest(cache, position):
    import torch

    layers = []
    for layer in cache:
        fields = {}
        for name, value in layer.items():
            valid = value[:position].contiguous()
            fields[name] = {
                "dtype": str(valid.dtype), "shape": list(valid.shape),
                "sha256": hashlib.sha256(
                    valid.view(torch.uint8).cpu().numpy().tobytes(),
                ).hexdigest(),
            }
        layers.append(fields)
    return {"position": position, "layers": layers}
