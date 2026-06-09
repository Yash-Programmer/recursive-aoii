"""Closed forms for the R-AoII cascade + independent SymPy verification.

Notation mirrors theory/02c: nu flip rate; mu_k link rate; q_k erasure;
mu_tilde_k=(1-q_k)mu_k; alpha_k=nu/(mu_tilde_k+2nu); p_k mismatch prob;
A_k~Exp(mu_tilde_k) refresh-renewal age (NOT tau_k, the per-hop policy clock).
"""
import numpy as np


# ---------------------------------------------------------------- numeric forms
def alpha(mu, nu, q=0.0):
    mut = (1.0 - q) * mu
    return nu / (mut + 2.0 * nu)


def p_recursion(mus, nu, qs=None):
    """p_k via the linear recursion p_k=(1-2 alpha_k) p_{k-1}+alpha_k, p_0=0."""
    mus = np.atleast_1d(np.asarray(mus, float))
    qs = np.zeros_like(mus) if qs is None else np.atleast_1d(np.asarray(qs, float))
    p = 0.0
    out = []
    for mu, q in zip(mus, qs):
        a = alpha(mu, nu, q)
        p = (1.0 - 2.0 * a) * p + a
        out.append(p)
    return np.array(out)


def p_K_product(mus, nu, qs=None):
    """p_K via the closed product  1/2 [ 1 - prod mu_tilde/(mu_tilde+2nu) ]."""
    mus = np.atleast_1d(np.asarray(mus, float))
    qs = np.zeros_like(mus) if qs is None else np.atleast_1d(np.asarray(qs, float))
    mut = (1.0 - qs) * mus
    prod = np.prod(mut / (mut + 2.0 * nu))
    return 0.5 * (1.0 - prod)


def raoii_K1_closed(mu, nu, q=0.0):
    """E[Delta^R_1] = nu / ((mu_tilde+nu)(mu_tilde+2nu))  (single-hop, always-push).

    Derivation (renewal-reward): mismatch-run length L~Exp(mu_tilde+nu),
    cycle C = Exp(nu)+Exp(mu_tilde+nu); E[Delta]=(1/2)E[L^2]/E[C].
    """
    mut = (1.0 - q) * mu
    return nu / ((mut + nu) * (mut + 2.0 * nu))


def threshold(mu, nu, beta, q=0.0):
    """DEPRECATED / INCORRECT closed form (the original "D1" guess); retained ONLY as a
    documented negative control. It does NOT solve the AoII scheduling problem: the correct
    single-hop optimum is the transcendental fixed point eq:(foc), obtained by
    policy.optimal_threshold_renewal / policy.threshold_for_budget and confirmed by
    policy.single_hop_vi. This formula is off by a large factor (see
    tests/test_phase03.py::test_thm3_d1_formula_is_wrong). Do not use it for any result."""
    mut = (1.0 - q) * mu
    a = alpha(mu, nu, q)
    arg = 1.0 - 2.0 * beta * a
    if arg <= 0.0:
        return np.inf
    return np.log(1.0 / arg) / (mut + 2.0 * nu)


# ---------------------------------------------------------------- symbolic checks
def verify_symbolic(verbose=True):
    """Independent SymPy re-derivation of every closed form. Returns dict of bools."""
    import sympy as sp
    res = {}

    nu = sp.symbols("nu", positive=True)
    mu = sp.symbols("mu", positive=True)

    # (1) residual-life integral  E[phi(A)] = nu/(mu+2nu) = alpha
    a = sp.symbols("a", positive=True)
    phi = sp.Rational(1, 2) * (1 - sp.exp(-2 * nu * a))
    integral = sp.integrate(phi * mu * sp.exp(-mu * a), (a, 0, sp.oo))
    res["residual_integral_eq_alpha"] = sp.simplify(integral - nu / (mu + 2 * nu)) == 0

    # (2) recursion solves to the product, symbolic K=1..5
    ok = True
    for K in range(1, 6):
        mus = sp.symbols(f"mu1:{K+1}", positive=True)
        alphas = [nu / (m + 2 * nu) for m in mus]
        p = sp.Integer(0)
        for al in alphas:
            p = (1 - 2 * al) * p + al           # the linear recursion
        prod = sp.prod([m / (m + 2 * nu) for m in mus])
        closed = sp.Rational(1, 2) * (1 - prod)
        ok = ok and sp.simplify(p - closed) == 0
    res["recursion_eq_product_K1to5"] = bool(ok)

    # (3) back-substitution identity:  d_K=(1-2a_K) d_{K-1}  <=>  product step
    dKm1 = sp.symbols("d", positive=True)        # 1-2 p_{K-1}
    aK = sp.symbols("alpha", positive=True)
    pKm1 = (1 - dKm1) / 2
    pK = (1 - 2 * aK) * pKm1 + aK
    res["back_substitution_zero"] = sp.simplify((1 - 2 * pK) - (1 - 2 * aK) * dKm1) == 0

    # (4) K=1 reduces to the Maatouk single-hop value
    res["K1_eq_maatouk"] = sp.simplify(
        (sp.Rational(1, 2) * (1 - mu / (mu + 2 * nu))) - nu / (mu + 2 * nu)) == 0

    # (5) nu_eff monotonicity: p_k - p_{k-1} = alpha_k (1-2 p_{k-1}) >= 0 for p_{k-1}<=1/2
    p_prev, al = sp.symbols("p_prev alpha_k", positive=True)
    step = ((1 - 2 * al) * p_prev + al) - p_prev
    res["p_increment_form"] = sp.simplify(step - al * (1 - 2 * p_prev)) == 0

    # (6) threshold fixed point: theta solves theta(mu+2nu)=log(1/(1-2 beta alpha)); unique, >0
    beta = sp.symbols("beta", positive=True)
    th = sp.symbols("theta", positive=True)
    al_s = nu / (mu + 2 * nu)
    sol = sp.solve(sp.Eq(th * (mu + 2 * nu), sp.log(1 / (1 - 2 * beta * al_s))), th)
    res["threshold_solves_unique"] = (len(sol) == 1)

    # (7) E[Delta^R_1] closed form vs the renewal-reward (1/2)E[L^2]/E[C] assembly
    L_mean = 1 / (mu + nu)                # E[L], L~Exp(mu+nu)
    L2 = 2 / (mu + nu) ** 2               # E[L^2]
    C_mean = 1 / nu + 1 / (mu + nu)       # E[cycle]
    raoii = (sp.Rational(1, 2) * L2) / C_mean
    res["raoii_K1_closed"] = sp.simplify(raoii - nu / ((mu + nu) * (mu + 2 * nu))) == 0
    # internal cross-check  E[L]/E[C] == p_1
    res["raoii_K1_fraction_eq_p1"] = sp.simplify(L_mean / C_mean - nu / (mu + 2 * nu)) == 0

    if verbose:
        for k, v in res.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return res


if __name__ == "__main__":
    print("SymPy verification of R-AoII closed forms:")
    r = verify_symbolic()
    print("ALL PASS" if all(r.values()) else "SOME FAILED")
