"""Tests for the unrestricted (genie / global-state) optimum benchmark, raoii/genie.py.

Anchors are exact: always-push reproduces the closed-form mean R-AoII to machine precision and
never-transmit hits the saturation bound 1/(2nu). The constrained genie optimum lies strictly below
the threshold-class optimum (the price of decentralization), and the optimal genie withholds the
link-2 forward exactly in the double-error-cancellation state.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from raoii import genie


def test_alwayspush_matches_closed_form():
    # K=2 homogeneous always-push: exact mean R-AoII is 25/144 (mu=2) and 0.08 (mu=4)
    assert abs(genie.always_push([2.0, 2.0], 1.0) - 25 / 144) < 1e-9
    assert abs(genie.always_push([4.0, 4.0], 1.0) - 0.08) < 1e-9


def test_neartransmit_saturation_bound():
    # never transmitting -> semantic age saturates at 1/(2 nu)
    assert genie.validate()


def test_genie_below_threshold_class():
    # the genie optimum is strictly below the threshold-class joint opt (price of decentralization);
    # at mu=2, nu=1, budget=0.25 the threshold opt is 0.250 and the genie sits ~0.17 (gap ~31%)
    res = genie.genie_opt([2.0, 2.0], 1.0, 0.25, target_Lh=0.12)
    assert np.all(res["rates"] <= 0.25 + 1e-3)          # feasible
    assert res["raoii"] < 0.250 - 0.03                  # clearly below threshold-class 0.250
    gap = 100 * (0.250 - res["raoii"]) / 0.250
    assert 25.0 < gap < 38.0                             # locked ~31%


def test_cancellation_withholding():
    # the genie forwards on link 2 when the monitor is wrong (M=(0,1)) but NOT when the monitor is
    # accidentally correct by error-cancellation (M=(1,1)) -- the source-dependent, unobservable rule
    frac = genie.link_tx_fraction([2.0, 2.0], 1.0, 0.25, link=2, patterns=[(0, 1), (1, 1)],
                                  target_Lh=0.10)
    assert frac[(0, 1)] > 0.8                            # forward fixes the monitor -> transmit
    assert frac[(1, 1)] < 0.05                           # forward breaks it -> withhold
