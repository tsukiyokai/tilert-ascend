"""Where can 55.4 ms a round possibly go?

The measured round is 55.424 ms at width 12 and acceptance 5.441. Reaching a
2.x ms TPOT needs the round near 15 ms, because acceptance would otherwise
have to reach 25 of 25 proposed. So the question is what the round is made of,
and the first thing to establish is the floor: the bytes that must cross HBM
once a round whatever the implementation.

Shapes are the checkpoint's own config. TP16, W4 weights at half a byte a
parameter, the MLA latent at int8.
"""
LAYERS = 78
DENSE_LAYERS = 3
HIDDEN = 6144
EXPERTS_PER_TOKEN = 8
SHARED_EXPERTS = 1
MOE_INTERMEDIATE = 2048
DENSE_INTERMEDIATE = 12288
HEADS = 64
Q_LORA = 2048
KV_LORA = 512
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256
VOCAB = 154880
TP = 16
W4_BYTES = 0.5
CONTEXT = 8192
# A3 die HBM, conservative.
BANDWIDTH_GB_S = 1600.0


def attention_parameters():
    """MLA, per layer: the two low-rank pairs, and the output projection."""
    q_a = HIDDEN * Q_LORA
    q_b = Q_LORA * HEADS * (QK_NOPE + QK_ROPE)
    kv_a = HIDDEN * (KV_LORA + QK_ROPE)
    kv_b = KV_LORA * HEADS * (QK_NOPE + V_HEAD)
    out = HEADS * V_HEAD * HIDDEN
    return q_a + q_b + kv_a + kv_b + out


def moe_parameters():
    """The experts a token actually visits, plus the shared one."""
    one_expert = 3 * MOE_INTERMEDIATE * HIDDEN
    return (EXPERTS_PER_TOKEN + SHARED_EXPERTS) * one_expert


def dense_parameters():
    return 3 * DENSE_INTERMEDIATE * HIDDEN


def main():
    attention = attention_parameters() * LAYERS
    moe = moe_parameters() * (LAYERS - DENSE_LAYERS)
    dense = dense_parameters() * DENSE_LAYERS
    head = VOCAB * HIDDEN
    active = attention + moe + dense + head
    per_die = active / TP * W4_BYTES
    # The MLA latent is one vector a token a layer, read whole each step.
    latent = CONTEXT * LAYERS * (KV_LORA + QK_ROPE) * 1
    total = per_die + latent
    floor_ms = total / (BANDWIDTH_GB_S * 1e9) * 1e3

    print(f"active parameters a token   {active/1e9:.2f} G")
    print(f"  attention {attention/1e9:.2f} G   moe {moe/1e9:.2f} G"
          f"   dense {dense/1e9:.3f} G   head {head/1e9:.3f} G")
    print(f"weights a die at W4          {per_die/1e6:.0f} MB")
    print(f"MLA latent at 8K, int8       {latent/1e6:.0f} MB")
    print(f"bytes a round a die          {total/1e6:.0f} MB")
    print(f"HBM floor a round            {floor_ms:.3f} ms"
          f"  at {BANDWIDTH_GB_S:.0f} GB/s")
    print()
    measured = 55.424
    print(f"measured round               {measured:.3f} ms")
    print(f"above the floor              {measured/floor_ms:.1f}x")
    print(f"needed for a 2.x ms tpot     ~15 ms at acceptance 5.441")
    print(f"  which is                   {15/floor_ms:.1f}x the floor")
    print()
    # What the measured pieces account for.
    sfa_us = 89.4
    vendor_moe_us = 98.950 + 58.226
    mine_moe_us = 37.193
    print(f"per layer, measured pieces:  attention {sfa_us:.1f} us"
          f"   moe vendor {vendor_moe_us:.1f} us   moe mine {mine_moe_us:.1f} us")
    vendor_total = (sfa_us + vendor_moe_us) * LAYERS / 1e3
    mine_total = (sfa_us + mine_moe_us) * LAYERS / 1e3
    print(f"those alone, all layers:     vendor {vendor_total:.1f} ms"
          f"   with my moe {mine_total:.1f} ms")
    print(f"unaccounted in the round:    {measured - vendor_total:.1f} ms"
          f"  ({(measured - vendor_total)/measured*100:.0f}%)")


if __name__ == "__main__":
    main()
