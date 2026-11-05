"""Finite router witnesses; device vendor outputs define numerical acceptance."""

TOPK_PARAMETERS = {
    "k": 8,
    "k_group": 1,
    "group_count": 1,
    "group_select_mode": 1,
    "renorm": 1,
    "norm_type": 1,
    "out_flag": False,
    "routed_scaling_factor": 1.0,
    "eps": 1e-20,
}


def logits_cases(torch, rows):
    generator = torch.Generator().manual_seed(63017 + rows)
    random = torch.randn(rows, 256, generator=generator, dtype=torch.float32)
    zero = torch.zeros(256, dtype=torch.float32)
    boundary = torch.full((rows, 256), -4.0, dtype=torch.float32)
    boundary[:, :7] = 2.0
    boundary[:, 7:10] = 1.0
    negative = torch.linspace(-0.01, -2.0, 256, dtype=torch.float32)
    unequal_logits = torch.arange(256, dtype=torch.float32).repeat(rows, 1) / 64
    return (
        ("all_tied_zero", torch.zeros_like(random), zero.clone()),
        ("eighth_ninth_tie", boundary, zero.clone()),
        ("negative_correction", unequal_logits, negative),
        ("all_positive_extreme", torch.full_like(random, 1000.0), zero.clone()),
        ("all_negative_extreme", torch.full_like(random, -1000.0), zero.clone()),
        ("alternating_extreme", torch.where(
            torch.arange(256) % 2 == 0, 1000.0, -1000.0).repeat(rows, 1), zero.clone()),
        ("random", random, torch.randn(256, generator=generator) * 0.04),
        ("random_tied_pairs", random.repeat_interleave(2, dim=1)[:, :256], zero.clone()),
    )


def bridge_case(torch, rows, iteration):
    generator = torch.Generator().manual_seed(63119 + rows * 100 + iteration)
    hidden = torch.randn(rows, 6144, generator=generator).bfloat16()
    residual = torch.randn(rows, 6144, generator=generator).bfloat16()
    if iteration % 8 == 4:
        hidden.zero_()
        residual.zero_()
    elif iteration % 8 == 5:
        residual = -hidden
    elif iteration % 8 == 6:
        hidden[:, ::2] = -64
        hidden[:, 1::2] = 64
    gamma = (torch.randn(6144, generator=generator) * 0.05 + 1).bfloat16()
    gate = (torch.randn(6144, 256, generator=generator) * 0.025).bfloat16()
    bias = torch.randn(256, generator=generator) * 0.04
    return hidden, residual, gamma, gate, bias


def epilogue_reference(routed, shared):
    """Retain the owning BF16 runner's separate multiply and add stores."""
    scaled = routed * 2.5
    return scaled, scaled + shared
