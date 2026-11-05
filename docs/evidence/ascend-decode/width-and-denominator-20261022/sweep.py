"""Where does width put TPOT, once the bank is priced by active groups?

Both curves saturate. Accepted tokens follow `(1 - a**(W+1)) / (1 - a)` with
`a` fitted to 7.1 at width 32, so they approach 6.33. A rank's touched experts
follow `16 * (1 - (248/256)**W)`, so they approach 16. The ratio therefore has
an interior minimum, and this finds it.
"""

import json

LAYERS = 78
ROUND_MS = 50.73
ENGINE_WIDTH, ENGINE_ACCEPTED = 12, 5.441
BANDWIDTH_GB_S = 1442.0
ATTENTION_US_A_ROW = 0.66
EXPERTS, PICKS, PER_RANK = 256, 8, 16
EXPERT_MB = 6.29 + 12.58
ACCEPTANCE = 0.86

WIDTHS = (1, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def touched(width):
    return PER_RANK * (1 - ((EXPERTS - PICKS) / EXPERTS) ** width)


def bank_us(width):
    return touched(width) * EXPERT_MB * 1e3 / BANDWIDTH_GB_S


def accepted(width):
    model = (1 - ACCEPTANCE ** (width + 1)) / (1 - ACCEPTANCE)
    calibration = ENGINE_ACCEPTED * (1 - ACCEPTANCE) / (
        1 - ACCEPTANCE ** (ENGINE_WIDTH + 1))
    return calibration * model


def row(width):
    layer = (ROUND_MS * 1e3 / LAYERS + bank_us(width) - bank_us(ENGINE_WIDTH)
             + ATTENTION_US_A_ROW * (width - ENGINE_WIDTH))
    round_ms = layer * LAYERS / 1e3
    tokens = accepted(width)
    return {'width': width, 'touched_experts': round(touched(width), 2),
            'layer_us': round(layer, 1), 'round_ms': round(round_ms, 2),
            'accepted': round(tokens, 3), 'tpot_ms': round(round_ms / tokens, 3)}


if __name__ == '__main__':
    rows = [row(width) for width in WIDTHS]
    best = min(rows, key=lambda entry: entry['tpot_ms'])
    print(json.dumps({'rows': rows, 'best': best,
                      'accepted_asymptote': round(accepted(4096), 3)}, indent=2))
