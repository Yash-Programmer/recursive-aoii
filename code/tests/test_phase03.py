"""Phase-03 deterministic verification (fast). MC bracket tests live in E1/E3 (Phase 05).

Run:  cd code && python -m pytest tests/test_phase03.py -q
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from raoii import theory, sim, lemma1, policy


# ---- Theorem 2 (symbolic, exact) ----------------------------------------------
def test_thm2_symbolic_all_pass():
    r = theory.verify_symbolic(verbose=False)
    assert all(r.values()), [k for k, v in r.items() if not v]


def test_thm2_recursion_equals_product_numeric():
    rng = np.random.default_rng(0)
    for _ in range(50):
        K = int(rng.integers(1, 8))
        mus = rng.uniform(0.3, 6.0, K)
        nu = float(rng.uniform(0.2, 3.0))
        assert abs(theory.p_recursion(mus, nu)[-1] - theory.p_K_product(mus, nu)) < 1e-12


def test_thm2_cascade_limit_half():
    assert abs(theory.p_K_product([2.0] * 40, 1.0) - 0.5) < 1e-3


# ---- Lemma 1 (enumeration, exact) ---------------------------------------------
def test_lemma1_xor_identity_enumeration():
    for K in range(1, 7):
        ok, bad = lemma1.check_xor_identity(K)
        assert ok, (K, bad)


def test_lemma1_union_bound_enumeration():
    for N in (2, 3, 4):
        for K in range(1, 5):
            ok, strict = lemma1.check_union_bound(N, K)
            assert ok
            if N >= 3 and K >= 2:
                assert strict                      # strict slack must exist for N>=3, K>=2


def test_lemma1_cancellation_witness():
    w = lemma1.cancellation_witness()
    assert w["sum"] == 2 and w["e2e"] == 0


def test_lemma1_xor_pathwise_sim_zero_violations():
    out = sim.tandem_sim([2.0, 2.0, 2.0], 1.0, n_events=200_000, seed=7)
    assert out["xor_violations"] == 0
    assert out["slack_mean"] >= 0.0


# ---- Theorem 2* (K=1 closed form) ---------------------------------------------
def test_thm2star_K1_closed_matches_renewal_fraction():
    # E[D^R_1] = (1/2) E[L^2]/E[C]; internal consistency E[L]/E[C] = p_1
    mu, nu = 2.0, 1.0
    assert abs(theory.raoii_K1_closed(mu, nu) - nu / ((mu + nu) * (mu + 2 * nu))) < 1e-12


# ---- Theorem 3 (corrected threshold) ------------------------------------------
def test_thm3_vi_matches_renewal_ground_truth():
    mu, beta = 2.0, 1.5
    for nu in (1.5, 1.0, 0.5, 0.25):
        th_vi = policy.single_hop_vi(mu, nu, beta)["theta_vi"]
        th_star, _ = policy.optimal_threshold_renewal(mu, nu, beta)
        assert abs(th_vi - th_star) < 0.05, (nu, th_vi, th_star)


def test_thm3_d1_formula_is_wrong():
    # the finding: D1's closed form does NOT match the true optimum
    mu, nu, beta = 2.0, 1.0, 1.5
    th_star, _ = policy.optimal_threshold_renewal(mu, nu, beta)
    th_d1 = theory.threshold(mu, nu, beta)
    assert abs(th_star - th_d1) > 1.0                 # off by ~1.25 (1.60 vs 0.35)


def test_thm3_threshold_increases_with_nu():
    mu, beta = 2.0, 1.5
    ths = [policy.optimal_threshold_renewal(mu, nu, beta)[0] for nu in (0.25, 0.5, 1.0, 1.5)]
    assert all(ths[i] < ths[i + 1] for i in range(len(ths) - 1))   # CORRECTED direction


def test_thm3_rate_monotone_in_theta():
    mu, nu = 3.0, 1.0
    ths = np.linspace(0.0, 5.0, 50)
    rates = [policy.attempt_rate_at_threshold(t, mu, nu) for t in ths]
    assert all(rates[i] >= rates[i + 1] - 1e-12 for i in range(len(rates) - 1))


def test_thm3_nesting_holds_under_ordered_budgets():
    mus, nu, lams = [3.0] * 4, 1.0, [0.4, 0.3, 0.2, 0.1]
    ps = theory.p_recursion(mus, nu)
    nu_eff = [nu * (1 - 2 * (0.0 if k == 0 else ps[k - 1])) for k in range(4)]
    th = [policy.threshold_for_budget(lams[k], mus[k], nu_eff[k]) for k in range(4)]
    assert all(th[i] <= th[i + 1] + 1e-9 for i in range(3)), th    # nesting holds


def test_thm3_nesting_reverses_under_equal_budgets():
    mus, nu, lams = [3.0] * 4, 1.0, [0.25] * 4
    ps = theory.p_recursion(mus, nu)
    nu_eff = [nu * (1 - 2 * (0.0 if k == 0 else ps[k - 1])) for k in range(4)]
    th = [policy.threshold_for_budget(lams[k], mus[k], nu_eff[k]) for k in range(4)]
    assert all(th[i] >= th[i + 1] - 1e-9 for i in range(3)), th    # reversed (the finding)


# ---- Theorem 4 (R_1 = 0) ------------------------------------------------------
def test_thm4_R1_zero():
    # K=1: Delta^R_1 = c_1 exactly, so R_1 = 0
    mu, nu = 2.0, 1.0
    c1 = theory.raoii_K1_closed(mu, nu)
    assert abs(c1 - theory.raoii_K1_closed(mu, nu)) == 0.0
    assert c1 > 0
