"""Phase-04 hardening checks (fast/deterministic). Heavy MC probes live in verify_phase04.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from raoii import theory, sim, policy, nstate


# ---- N-state symmetric closed form (Proposition 2-dagger-sym) ------------------
def test_nstate_N2_collapse_to_thm2():
    nu = 1.0
    for mu in (1.0, 2.0, 5.0):
        f = nstate.subdominant_factor(nstate.sym_generator(2, nu), mu)
        assert abs(f - mu / (mu + 2 * nu)) < 1e-9


def test_nstate_resolvent_matches_thm2_product():
    nu = 1.0
    mus = [2.0, 3.0, 1.5]
    assert abs(nstate.mismatch_via_resolvent(mus, nstate.sym_generator(2, nu))
               - theory.p_K_product(mus, nu)) < 1e-9


def test_nstate_symmetric_subdominant_formula():
    nu = 1.0
    for N in (3, 4, 5):
        f = nstate.subdominant_factor(nstate.sym_generator(N, nu), 2.0)
        assert abs(f - 2.0 / (2.0 + N * nu / (N - 1))) < 1e-9


# ---- cascade-limit boundary (04c B-04) ----------------------------------------
def test_cascade_limit_boundary_geometric_mu():
    # mu_k = 4^k => sum 1/mu_k converges => p_K stays bounded away from 1/2
    nu = 1.0
    p = theory.p_K_product([4.0 ** k for k in range(1, 17)], nu)
    assert p < 0.25                          # saturates ~0.2157, NOT 0.5


def test_cascade_limit_reached_when_sum_diverges():
    assert abs(theory.p_K_product([2.0] * 40, 1.0) - 0.5) < 1e-3


# ---- threshold monotonicity sign (the corrected nesting driver, 04c [A]) -------
def test_threshold_increases_with_effective_rate():
    mu, beta = 2.0, 1.5
    ths = [policy.optimal_threshold_renewal(mu, nu, beta)[0] for nu in (0.25, 0.5, 1.0, 1.5)]
    assert all(ths[i] < ths[i + 1] for i in range(3))


# ---- erasure invariance (04e ext i, light MC) ---------------------------------
def test_erasure_invariance_mu_tilde():
    mus, nu, qs = [3.0, 2.0, 4.0], 1.0, [0.3, 0.5, 0.2]
    mut = [(1 - qs[k]) * mus[k] for k in range(3)]
    closed = theory.p_K_product(mut, nu)
    m, ci, _ = sim.mc_estimate(mus, nu, qs=qs, n_events=150_000, n_seeds=8)
    assert abs(m - closed) <= ci + 1e-9, (m, closed, ci)


# ---- concurrent-mismatch sub-multiplicativity (04b strat iii, light MC) --------
def test_concurrent_mismatch_submultiplicative():
    mus, nu = [2.0, 3.0, 1.5, 4.0], 1.0
    out = sim.tandem_sim(mus, nu, n_events=300_000, seed=5, track_pairs=True)
    pm, hop = out["pair_mismatch"], out["hop_mismatch"]
    for (i, j) in [(1, 2), (1, 4), (2, 3), (3, 4)]:
        # P(M_i & M_j) <= E[M_i] E[M_j] * (1 + small slack) : product is a valid upper bound
        assert pm[i, j] <= hop[i - 1] * hop[j - 1] * 1.2 + 1e-3, (i, j, pm[i, j])
