"""Tests for the exact joint benchmark over the per-link threshold class (raoii/benchmark.py).

These back the paper's Table II: the age-augmented evaluator reproduces the exact joint-CTMC
of joint.py under always-push, and the decentralized policy's gap to the threshold-class
optimum is governed by calibration. The surrogate (offline closed-form) calibration sits far
above the optimum, while the achieved (measured-rate) calibration is near-optimal and drives
the final hop to a zero threshold. Deterministic; small grids so CI stays fast.
"""
import numpy as np
from raoii import benchmark as bm, joint


def test_alwayspush_raoii_matches_joint():
    """Always-push mean R-AoII on the age-augmented chain reproduces -pi_M^T A^{-1} 1.
    Always-push ignores the local age, so a tiny age grid is exact up to the O(h) step bias."""
    for mus in ([2.0, 2.0], [2.0, 3.0]):
        ra, _ = bm.metrics(mus, 1.0, [0] * len(mus), 4, 0.01)
        ex = joint.analyze(mus, 1.0)["raoii"]
        assert abs(ra - ex) < 5e-3, (mus, ra, ex)


def test_calibration_governs_the_gap():
    """Table II finding: surrogate calibration sits well above the threshold optimum, while
    achieved (measured-rate) calibration is near-optimal and zeroes the final-hop threshold."""
    mu, nu, budget = 2.0, 1.0, 0.25
    G, h = bm.grid_for(mu, nu, 4.0, 0.20)            # coarse grid, fast for CI
    ts = bm.surrogate_thresholds([mu, mu], nu, budget, G, h)
    ta = bm.achieved_thresholds([mu, mu], nu, budget, G, h)
    rs, _ = bm.metrics([mu, mu], nu, ts, G, h)
    ra, _ = bm.metrics([mu, mu], nu, ta, G, h)
    assert ra < rs - 1e-3, (ra, rs)                  # achieved beats surrogate
    assert abs(ra - 0.250) < 0.03, ra                # achieved ~ threshold optimum (Table II)
    assert rs > 0.30, rs                             # surrogate ~30% above optimum (Table II 29.8%)
    assert ta[-1] == 0                               # final hop forwards aggressively
