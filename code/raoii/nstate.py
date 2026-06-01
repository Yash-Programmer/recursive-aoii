"""N-state cascade: per-hop operator recursion + spectral analysis (Proposition 2†).

The joint law rho_k in Delta(S^2) of (X, Xhat_k) updates as rho_k = T_k rho_{k-1},
where T_k = (copy o resolvent) with resolvent mu_k (mu_k I - Q)^{-1} = E_{a~Exp(mu_k)}[e^{Qa}].
We work with the marginal disagreement contraction on the "innovation" subspace.

For the symmetric two-state Q=[[-nu,nu],[nu,-nu]], the cascade factor must collapse to
mu_k/(mu_k+2nu) (= 1-2 alpha_k), the subdominant resolvent eigenvalue (eig of Q = 0,-2nu).
"""
import numpy as np
from scipy.linalg import expm


def sym_generator(N, nu):
    """Symmetric N-state generator: off-diagonal rate nu/(N-1), uniform stationary."""
    Q = np.full((N, N), nu / (N - 1))
    np.fill_diagonal(Q, -nu)
    return Q


def resolvent(Q, mu):
    """E_{a~Exp(mu)}[e^{Qa}] = mu (mu I - Q)^{-1}  (Laplace transform of the semigroup)."""
    N = Q.shape[0]
    return mu * np.linalg.inv(mu * np.eye(N) - Q)


def subdominant_factor(Q, mu):
    """The candidate per-hop cascade factor: second-largest |eigenvalue| of resolvent(Q,mu).
    Resolvent eigenvalues are mu/(mu-lam) for eigenvalues lam of Q; lam=0 -> 1 (stationary),
    subdominant lam gives the contraction. For symmetric 2-state lam=-2nu -> mu/(mu+2nu)."""
    R = resolvent(Q, mu)
    ev = np.sort(np.abs(np.linalg.eigvals(R)))[::-1]
    return ev[1]


def mismatch_via_resolvent(mus, Q):
    """End-to-end P(X != Xhat_K) via the marginal innovation recursion.

    Model: at each hop, Xhat_k is set to Xhat_{k-1} a time A_k~Exp(mu_k) ago; the source has
    since moved by e^{Q A_k}. Track the distribution d_k(j) = P(Xhat_k = j | X = i) under
    symmetry (independent of i). Recursion: d_k = R_k^T d_{k-1} composed with the copy; the
    diagonal-agreement mass gives 1 - P(mismatch). We compute it by direct simulation-free
    propagation of the agreement probability using the symmetric structure.

    For symmetric Q, P(agree)_k = 1/N + (1-1/N) * prod_{j<=k} r_j, with r_j the subdominant
    factor; so P(mismatch)_K = (1-1/N)(1 - prod_j r_j).  (The N-state analog of Thm 2.)
    """
    N = Q.shape[0]
    r = np.prod([subdominant_factor(Q, mu) for mu in mus])
    return (1.0 - 1.0 / N) * (1.0 - r)


if __name__ == "__main__":
    print("N-state spectral checks:")
    # N=2 collapse: subdominant factor must equal mu/(mu+2nu)
    nu = 1.0
    for mu in (1.0, 2.0, 5.0):
        Q2 = sym_generator(2, nu)
        f = subdominant_factor(Q2, mu)
        print(f"  N=2 mu={mu}: subdominant factor={f:.6f}  mu/(mu+2nu)={mu/(mu+2*nu):.6f}  "
              f"{'OK' if abs(f-mu/(mu+2*nu))<1e-9 else 'FAIL'}")
    # N=2 mismatch must match Theorem 2 product
    from . import theory
    mus = [2.0, 3.0, 1.5]
    p_ns = mismatch_via_resolvent(mus, sym_generator(2, nu))
    p_t2 = theory.p_K_product(mus, nu)
    print(f"  N=2 cascade p_K: resolvent={p_ns:.6f}  Thm2={p_t2:.6f}  "
          f"{'OK' if abs(p_ns-p_t2)<1e-9 else 'FAIL'}")
    # N=3,4 symmetric: report the spectral-product mismatch (saturates at 1-1/N)
    for N in (3, 4):
        QN = sym_generator(N, nu)
        for K in (1, 3, 8):
            p = mismatch_via_resolvent([2.0] * K, QN)
            print(f"  N={N} K={K} mu=2: P(mismatch)={p:.4f}  (saturation 1-1/N={1-1/N:.4f})")
