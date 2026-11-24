"""Recompute the road to 2 ms at the real distinct-expert count.

The 2026-10-21 table priced the bank narrowing at two experts a rank and at
inner 1536. The checkpoint's widths are 1024 and 2048, and the count follows
from the routing: a row picks eight distinct experts of 256, so after M rows
each expert is untouched with probability (248/256)**M.
"""

import json

LAYERS = 78
ROUND_MS = 50.73
ACCEPTED = {12: 5.441, 32: 6.872}
EXPERTS, PICKS, PER_RANK = 256, 8, 16
BANDWIDTH_GB_S = 1255.0
BANK_FLOOR_US = 17.0
ATTENTION_US_A_ROW = 0.66

# Profiled at the checkpoint's real widths; see bank-narrowing-20261021.
BANK_CALLS_MB = {'gate_up': 100.7, 'down': 201.3}

STEPS = (
    ('the routed metadata as one kernel', 25.0, 'estimate, bounded by measurement'),
    ('the expert combine in the matmul epilogue', 20.0, 'estimate'),
    ('the latent and rope made contiguous', 23.0, 'estimate'),
    ('the four projections fused', 35.0, 'estimate'),
    ('one reduction a layer, not two', 34.0, 'measured, costs 13.91 GB a rank'),
    ('both reductions inside the launch', 11.5, 'measured, confounded with SHMEM'),
)


def touched_a_rank(rows):
    return PER_RANK * (1 - ((EXPERTS - PICKS) / EXPERTS) ** rows)


def bank_us(fraction):
    return sum(max(BANK_FLOOR_US, megabytes * fraction * 1e3 / BANDWIDTH_GB_S)
               for megabytes in BANK_CALLS_MB.values())


def road(rows):
    layer = ROUND_MS * 1e3 / LAYERS + ATTENTION_US_A_ROW * (rows - 12)
    touched = touched_a_rank(rows)
    narrowing = bank_us(1.0) - bank_us(touched / PER_RANK)
    trace = []
    for name, saving, basis in (('the bank narrowed to what a round touches',
                                 narrowing, 'measured'),) + STEPS:
        layer -= saving
        trace.append({'step': name, 'saving_us_a_layer': round(saving, 2),
                      'tpot_ms': round(layer * LAYERS / 1e3 / ACCEPTED[rows], 3),
                      'basis': basis})
    return {'rows': rows, 'accepted': ACCEPTED[rows],
            'touched_experts_a_rank': round(touched, 2),
            'bank_full_us': round(bank_us(1.0), 2),
            'bank_narrowed_us': round(bank_us(touched / PER_RANK), 2),
            'steps': trace, 'tpot_ms': trace[-1]['tpot_ms']}


if __name__ == '__main__':
    print(json.dumps({'width_12': road(12), 'width_32': road(32)}, indent=2))
