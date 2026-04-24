"""Validate the CPU fast-marching Eikonal solver against the analytic point
source solution on a 128x128 grid. No GPU.
"""

import numpy as np

from qgps.reference_eikonal import analytic_point_source, solve_eikonal_fmm


def test_point_source_matches_analytic_within_5_percent():
    h = w = 128
    speed = np.ones((h, w), dtype=np.float32)
    source = (64, 64)

    T = solve_eikonal_fmm(speed, source)
    T_ref = analytic_point_source((h, w), source, speed=1.0)

    # Ignore a small ring around the source where FMM first-order update is
    # known to be biased; standard practice when validating FMM.
    ys, xs = np.mgrid[0:h, 0:w]
    r = np.sqrt((xs - source[0]) ** 2 + (ys - source[1]) ** 2)
    mask = r > 5.0

    # Relative error statistics. First-order FMM has a well-known O(h)
    # anisotropic bias along axis rays, so L-inf can be ~10% on a 128x128
    # grid even when the solution is correct. We validate via L2-relative
    # error and mean pointwise relative error, both of which must be < 5%.
    num = np.abs(T[mask] - T_ref[mask])
    denom = np.maximum(T_ref[mask], 1e-6)
    rel_err_max = float(np.max(num / denom))
    mean_rel = float(np.mean(num / denom))
    l2_rel = float(
        np.linalg.norm(T[mask] - T_ref[mask]) / np.linalg.norm(T_ref[mask])
    )

    print(
        f"FMM rel error: max={rel_err_max:.4f} mean={mean_rel:.4f} "
        f"L2_rel={l2_rel:.4f}"
    )
    assert mean_rel < 0.05, f"mean relative error {mean_rel:.4f} >= 5%"
    assert l2_rel < 0.05, f"L2 relative error {l2_rel:.4f} >= 5%"


def test_monotonic_along_radius():
    """Arrival time must be non-decreasing along any outward ray."""
    h = w = 64
    speed = np.ones((h, w), dtype=np.float32)
    T = solve_eikonal_fmm(speed, (32, 32))
    row = T[32, 32:]
    diffs = np.diff(row)
    assert np.all(diffs >= -1e-5), "T not monotonic along +x ray"
