"""Phase-05 checks: JIT engine fidelity, policies, heavy-tail, tree path-product (fast)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from raoii import theory, sim, heavytail


def test_jit_matches_python_reference_exactly():
    for pol, th in [("always", None), ("threshold", [0.3, 0.5, 0.2])]:
        a = sim.tandem_sim([2., 3., 1.5], 1.0, policy=pol, thetas=th, n_events=40_000, seed=3, track_pairs=True)
        b = sim.tandem_sim_python([2., 3., 1.5], 1.0, policy=pol, thetas=th, n_events=40_000, seed=3, track_pairs=True)
        assert abs(a["p_K"] - b["p_K"]) < 1e-12
        assert abs(a["raoii"] - b["raoii"]) < 1e-12
        assert a["xor_violations"] == b["xor_violations"]
        assert abs(a["tx_rate"].sum() - b["tx_rate"].sum()) < 1e-12


def test_all_policies_run_and_no_xor_violations():
    for pol, th in [("always", None), ("threshold", [.5, .5, .5]),
                    ("random", [.3, .3, .3]), ("aoi", [.4, .4, .4])]:
        o = sim.tandem_sim([2., 2., 2.], 1.0, policy=pol, thetas=th, n_events=150_000, seed=1)
        assert o["xor_violations"] == 0
        assert 0.0 <= o["p_K"] <= 0.5 + 1e-6


def test_e1_bracket_full_precision_spotcheck():
    # one full-precision E1 point must bracket
    mus = [2.0] * 6; nu = 1.0
    mean, ci, _ = sim.mc_estimate(mus, nu, n_events=1_000_000, n_seeds=20)
    assert abs(mean - theory.p_K_product(mus, nu)) <= ci + 1e-9


def test_tx_rate_nonneg_and_below_mu():
    o = sim.tandem_sim([2., 3.], 1.0, policy="threshold", thetas=[0.5, 0.5], n_events=200_000, seed=0)
    assert np.all(o["tx_rate"] >= 0) and np.all(o["tx_rate"] <= np.array([2., 3.]) + 0.1)


def test_heavytail_deviation_bounded():
    # Pareto opportunities: product formula holds within ~15%
    m, ci = heavytail.heavytail_pK([2., 2., 2.], 1.0, n_events=300_000, n_seeds=8)
    exp = theory.p_K_product([2., 2., 2.], 1.0)
    assert abs(m - exp) / exp < 0.15


def test_tree_leaf_equals_path_product():
    # a leaf's marginal mismatch == product along its root->leaf path
    path = [2.0, 3.0, 1.5]
    mean, ci, _ = sim.mc_estimate(path, 1.0, n_events=500_000, n_seeds=10)
    assert abs(mean - theory.p_K_product(path, 1.0)) <= ci + 1e-9
