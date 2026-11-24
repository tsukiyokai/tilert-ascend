"""Split the round into what the width costs and what it does not.

Four widths were measured with the same drafter on the same machine, so a
straight line through them separates the per-round fixed cost from the
marginal cost of a verified row -- without a profiler, and without the
subtraction the roofline page had to resort to.
"""
WIDTHS = (8, 12, 16, 32)
ROUNDS = (52.787, 55.424, 58.588, 72.066)
ACCEPTED = (4.547, 5.441, 5.590, 6.053)


def fit(xs, ys):
    """Least squares, two parameters."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
             / sum((x - mean_x) ** 2 for x in xs))
    return mean_y - slope * mean_x, slope


fixed, marginal = fit(WIDTHS, ROUNDS)
print(f"round(width) = {fixed:.2f} ms + {marginal:.3f} ms a row")
for width, measured, accepted in zip(WIDTHS, ROUNDS, ACCEPTED):
    predicted = fixed + marginal * width
    print(f"  w{width:>2}  measured {measured:.3f}  fit {predicted:.3f}"
          f"  residual {measured - predicted:+.3f}"
          f"  tpot {measured / accepted:.3f}")

print()
print("what the fixed cost allows, even with perfect acceptance:")
for width in (12, 26, 32, 64):
    best = (fixed + marginal * width) / width
    print(f"  w{width:>2}  tpot floor {best:.3f} ms  (accepted == width)")

print()
print("what fixed cost a 2.5 ms tpot needs, at the acceptance measured:")
for accepted, width in ((5.441, 12), (8.0, 16), (12.0, 24)):
    needed = 2.5 * accepted - marginal * width
    print(f"  accepted {accepted:>5.2f} at w{width:>2}"
          f"  needs fixed <= {needed:.2f} ms"
          f"   ({fixed / needed:.1f}x below today)" if needed > 0 else
          f"  accepted {accepted:>5.2f} at w{width:>2}  unreachable")
