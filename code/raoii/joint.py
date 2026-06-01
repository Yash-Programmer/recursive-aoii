"""Exact joint-CTMC analysis of the always-push cascade (closes OPEN 2.2).

State = (X, Xhat_1, ..., Xhat_K) in {0,1}^{K+1}, encoded as bits b0=X=Xhat_0, b_k=Xhat_k.
Generator: source flips b0 at rate nu; link k refreshes b_k <- b_{k-1} at rate mu_k.

Exact results (matrix-analytic; the proper way to characterize an AoII time-integral,
cf. Cosandal-Akar-Ulukus MRPH):
  * p_K = sum of stationary mass on the mismatch set M={b0 != bK}  (must equal Theorem 2).
  * Theorem 2*  E[Delta^R_K] = integral_0^inf P(continuously mismatched for s) ds
                            = -pi_M^T A^{-1} 1,
    where A is the generator restricted to M (transitions out of M killed), pi_M the
    stationary mass on M. Derivation: (t-U_K) 1{wrong} = int_0^inf 1{wrong throughout
    (t-s,t]} ds; take E and integrate the survival of the mismatch set.
  * exact pairwise P(M_i=1, M_j=1) from the joint stationary law (hardens OPEN 4.2).
"""
import numpy as np
import itertools


def build_generator(mus, nu):
    mus = np.atleast_1d(np.asarray(mus, float)); K = len(mus)
    n = 1 << (K + 1)
    Q = np.zeros((n, n))
    for s in range(n):
        # source flip: toggle bit 0, rate nu
        Q[s, s ^ 1] += nu
        # link k refresh: b_k <- b_{k-1}, rate mu_k
        for k in range(1, K + 1):
            bk = (s >> k) & 1
            bkm1 = (s >> (k - 1)) & 1
            if bk != bkm1:
                s2 = s ^ (1 << k)            # set bit k to b_{k-1}
            else:
                s2 = s                       # no change
            if s2 != s:
                Q[s, s2] += mus[k - 1]
            # if s2==s the refresh is a self-loop (no state change) -> omit
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(n)] = -Q.sum(axis=1)
    return Q, K


def stationary(Q):
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1); b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def mismatch_mask(K):
    n = 1 << (K + 1)
    return np.array([((s & 1) ^ ((s >> K) & 1)) == 1 for s in range(n)])


def analyze(mus, nu):
    Q, K = build_generator(mus, nu)
    pi = stationary(Q)
    M = mismatch_mask(K)
    pK = pi[M].sum()
    # Theorem 2*: E[Delta^R_K] = -pi_M^T A^{-1} 1, A = Q restricted to M
    A = Q[np.ix_(M, M)]
    piM = pi[M]
    raoii = -piM @ np.linalg.solve(A, np.ones(M.sum()))
    return {"p_K": pK, "raoii": raoii, "K": K, "Q": Q, "pi": pi, "M": M}


def hop_mismatch(mus, nu):
    """Exact per-hop E[M_k]=P(Xhat_{k-1}!=Xhat_k) from the joint stationary law."""
    Q, K = build_generator(mus, nu); pi = stationary(Q); n = 1 << (K + 1)
    Ek = np.zeros(K + 1)
    for s in range(n):
        for k in range(1, K + 1):
            if ((s >> (k - 1)) & 1) ^ ((s >> k) & 1):
                Ek[k] += pi[s]
    return Ek[1:]


def pair_mismatch(mus, nu):
    """Exact P(M_i=1, M_j=1) for all pairs, from the joint stationary law."""
    Q, K = build_generator(mus, nu); pi = stationary(Q); n = 1 << (K + 1)
    P = np.zeros((K + 1, K + 1))
    for s in range(n):
        mk = [((s >> (k - 1)) & 1) ^ ((s >> k) & 1) for k in range(1, K + 1)]
        for i in range(K):
            if mk[i]:
                for j in range(i + 1, K):
                    if mk[j]:
                        P[i + 1, j + 1] += pi[s]
    return P


def cov_M1M2_symbolic():
    """Symbolic Cov(M_1,M_2) for K=2 from the 8-state joint CTMC.

    Returns (cov_simplified, closed_form, nu, mu1, mu2). The paper's eq:cov12 is
        Cov(M1,M2) = -mu1*mu2*nu**2 / ((mu1+mu2)*(mu1+2nu)**2*(mu2+2nu))  < 0.
    M_1 = 1{X != Xhat_1}, M_2 = 1{Xhat_1 != Xhat_2}; bits b0=X, b1, b2.
    """
    import sympy as sp
    mu1, mu2, nu = sp.symbols("mu1 mu2 nu", positive=True)
    mus = [mu1, mu2]
    Q = sp.zeros(8, 8)
    bit = lambda s, k: (s >> k) & 1
    for s in range(8):
        Q[s, s ^ 1] += nu                                  # source flip b0
        for k in (1, 2):
            if bit(s, k) != bit(s, k - 1):
                Q[s, s ^ (1 << k)] += mus[k - 1]           # refresh b_k <- b_{k-1}
    for s in range(8):
        Q[s, s] = -sum(Q[s, j] for j in range(8) if j != s)
    p = sp.symbols("p0:8", nonnegative=True)
    pv = sp.Matrix(p)
    eqs = [(pv.T * Q)[j] for j in range(8)]
    sol = sp.solve(eqs[:7] + [sum(p) - 1], list(p), dict=True)[0]
    pi = sp.Matrix([sol[pi_] for pi_ in p])
    M1 = lambda s: bit(s, 0) ^ bit(s, 1)
    M2 = lambda s: bit(s, 1) ^ bit(s, 2)
    PM1 = sum(pi[s] for s in range(8) if M1(s))
    PM2 = sum(pi[s] for s in range(8) if M2(s))
    PM1M2 = sum(pi[s] for s in range(8) if M1(s) and M2(s))
    cov = sp.simplify(PM1M2 - PM1 * PM2)
    closed = -mu1 * mu2 * nu ** 2 / ((mu1 + mu2) * (mu1 + 2 * nu) ** 2 * (mu2 + 2 * nu))
    return cov, closed, nu, mu1, mu2


def nqd_worst_ratio(mus, nu):
    """Worst pairwise ratio P(M_i,M_j)/(E[M_i] E[M_j]); NQD holds iff <= 1."""
    Em = hop_mismatch(mus, nu); P = pair_mismatch(mus, nu); K = len(mus)
    worst = 0.0
    for i in range(1, K + 1):
        for j in range(i + 1, K + 1):
            prod = Em[i - 1] * Em[j - 1]
            if prod > 0:
                worst = max(worst, P[i, j] / prod)
    return worst


def raoii_K1_symbolic():
    """Symbolic E[Delta^R_1] from the joint CTMC -> ground-truth closed form."""
    import sympy as sp
    nu, mu = sp.symbols("nu mu", positive=True)
    # states (X,Xhat1): 00,01,10,11 ; mismatch M={01,10}
    Q = sp.zeros(4, 4)
    def idx(x, h): return x * 2 + h
    for x in (0, 1):
        for h in (0, 1):
            s = idx(x, h)
            Q[s, idx(1 - x, h)] += nu                 # source flip
            if h != x:
                Q[s, idx(x, x)] += mu                 # refresh Xhat1<-X
    for s in range(4):
        Q[s, s] = -sum(Q[s, j] for j in range(4) if j != s)
    # stationary
    pi = sp.symbols("p0 p1 p2 p3")
    eqs = [sum(pi[i] * Q[i, j] for i in range(4)) for j in range(4)]
    sol = sp.solve(eqs[:3] + [sum(pi) - 1], pi, dict=True)[0]
    piv = sp.Matrix([sol[p] for p in pi])
    M = [1, 2]                                          # 01,10
    A = Q[M, M]
    piM = sp.Matrix([piv[m] for m in M])
    raoii = -(piM.T * A.inv() * sp.ones(2, 1))[0]
    return sp.simplify(raoii), nu, mu


if __name__ == "__main__":
    import sympy as sp
    from . import theory
    print("Joint-CTMC exact analysis (closes OPEN 2.2):")
    # (1) p_K matches Theorem 2 product
    for mus in ([2.], [2., 3.], [2., 3., 1.5], [1., 2., 3., 4.]):
        a = analyze(mus, 1.0)
        print(f"  K={len(mus)} mus={mus}: p_K joint={a['p_K']:.6f} product={theory.p_K_product(mus,1.0):.6f} "
              f"E[D^R_K]={a['raoii']:.6f}")
    # (2) symbolic K=1
    r, nu, mu = raoii_K1_symbolic()
    print(f"  symbolic E[D^R_1] = {r}  (should be nu/((mu+nu)(mu+2nu)))")
    print(f"  check == nu/((mu+nu)(mu+2nu)): "
          f"{sp.simplify(r - nu/((mu+nu)*(mu+2*nu)))==0}")
