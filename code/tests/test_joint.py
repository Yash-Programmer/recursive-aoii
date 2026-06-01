"""Theorem 2* (time-integral, OPEN 2.2 closure) + exact sub-multiplicativity (OPEN 4.2)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sympy as sp
from raoii import joint, sim, theory


def test_joint_pK_equals_product():
    for mus in ([2.], [2., 3.], [2., 3., 1.5], [1., 2., 3., 4.]):
        assert abs(joint.analyze(mus, 1.0)["p_K"] - theory.p_K_product(mus, 1.0)) < 1e-9


def test_raoii_K1_symbolic_closed_form():
    r, nu, mu = joint.raoii_K1_symbolic()
    assert sp.simplify(r - nu / ((mu + nu) * (mu + 2 * nu))) == 0


def test_thm2star_joint_matches_sim():
    for mus in ([2., 2.], [2., 2., 2.], [3., 1., 4.]):
        ex = joint.analyze(mus, 1.0)["raoii"]
        m, ci, _ = sim.mc_estimate(mus, 1.0, n_events=600_000, n_seeds=12, key="raoii")
        assert abs(ex - m) <= ci + 1e-9, (mus, ex, m, ci)


def test_thm2star_K2_closed_form():
    # E[D^R_2]|mu1=mu2=mu = nu(2mu+nu)(3mu+4nu) / (2(mu+nu)^2(mu+2nu)^2)
    for mu, nu in [(2., 1.), (3., 1.), (1.5, 0.5)]:
        closed = nu * (2 * mu + nu) * (3 * mu + 4 * nu) / (2 * (mu + nu) ** 2 * (mu + 2 * nu) ** 2)
        assert abs(joint.analyze([mu, mu], nu)["raoii"] - closed) < 1e-9


def test_exact_submultiplicativity():
    for mus in ([2., 3., 1.5, 4.], [1.] * 5):
        K = len(mus)
        Ek = joint.hop_mismatch(mus, 1.0)
        P = joint.pair_mismatch(mus, 1.0)
        for i in range(1, K + 1):
            for j in range(i + 1, K + 1):
                assert P[i, j] <= Ek[i - 1] * Ek[j - 1] + 1e-12, (i, j)


def test_cov_M1M2_closed_form():
    # eq:cov12 : Cov(M1,M2) = -mu1 mu2 nu^2 / ((mu1+mu2)(mu1+2nu)^2(mu2+2nu)) < 0
    cov, closed, nu, mu1, mu2 = joint.cov_M1M2_symbolic()
    assert sp.simplify(cov - closed) == 0
    # strictly negative on positive rates
    for subs in ({mu1: 2., mu2: 3., nu: 1.}, {mu1: .3, mu2: 5., nu: 2.}):
        assert float(closed.subs(subs)) < 0


def test_nqd_worst_ratio_below_one():
    for mus in ([2., 2., 2.], [0.3, 5., 5.], [0.5, 1., 2., 4.], [1.] * 6):
        assert joint.nqd_worst_ratio(mus, 1.0) <= 1.0 + 1e-9
