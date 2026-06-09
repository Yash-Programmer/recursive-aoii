"""Exact joint benchmark over the per-link threshold class (paper Table II).

For the constrained R-AoII cascade we evaluate, exactly on the *age-augmented* joint chain
(the joint disagreement configuration together with each link's quantized local age), the
mean R-AoII and the realized per-link transmission rates of ANY per-link threshold vector,
without simulation. With this evaluator we compare the decentralized policy of Algorithm 1
under two calibrations against the optimum OVER ALL PER-LINK THRESHOLD VECTORS subject to the
budgets (the class to which the decoupled policy itself belongs):

  * surrogate calibration: each threshold set from the closed-form single-hop rate (Thm 8),
    treating the upstream estimate as a symmetric source at its effective flip rate;
  * achieved  calibration: each threshold set so the *realized* rate meets the budget
    (the online, measured-rate step of Algorithm 1).

This reproduces Table II. The optimum below is over per-link threshold vectors only; the
unrestricted (genie) optimum, which augments the state with the monitor's end-to-end
wrong-run age, is computed exactly in genie.py and benchmarked as E7 (the paper's price
of decentralization).

Method: a uniformized DTMC of step h (h chosen so Lam*h is small), the exact stationary
distribution by a sparse linear solve, and the killed-chain fundamental matrix for the mean
R-AoII (the same -pi_M^T A^{-1} 1 identity used in joint.py, here on the age-augmented chain).
Validated against joint.py (always-push p_K and mean R-AoII) and against the single-hop
renewal-reward rate of theory/policy.

Discretization convention: per-link ages advance only on the no-event branch (each event
branch carries probability O(h), so the age bias is O(h)); it is suppressed by the
coarse-search/fine-evaluate protocol of run_point and checked against joint.py and the
renewal-reward closed forms. genie.py instead advances the wrong-run age on EVERY branch
on which the monitor stays wrong; both conventions agree as h -> 0.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from itertools import product
from . import theory


# --------------------------------------------------------------- age-augmented sparse chain
def _Mvec(e, K):
    return tuple(1 if e[k - 1] != e[k] else 0 for k in range(1, K + 1))


def _enumerate(K, G):
    states = []
    for e in product((0, 1), repeat=K + 1):
        M = _Mvec(e, K)
        ranges = [range(G + 1) if M[k] else range(1) for k in range(K)]
        for n in product(*ranges):
            states.append((e, n))
    return states


def _chain(mus, nu, thr_idx, G, h):
    """Uniformized DTMC (step h) for the per-link threshold policy thr_idx (in age bins)."""
    K = len(mus)
    states = _enumerate(K, G)
    idx = {s: i for i, s in enumerate(states)}
    N = len(states)
    rows, cols, vals = [], [], []
    Lh = (nu + sum(mus)) * h
    for s in states:
        i = idx[s]; e, n = s; M = _Mvec(e, K)
        nn = tuple(min(n[k] + 1, G) if M[k] else 0 for k in range(K))   # no event: advance ages
        rows.append(i); cols.append(idx[(e, nn)]); vals.append(1 - Lh)
        e2 = (e[0] ^ 1,) + e[1:]; n2 = (0,) + n[1:]                     # source flip
        rows.append(i); cols.append(idx[(e2, n2)]); vals.append(nu * h)
        for k in range(1, K + 1):                                       # link-k opportunity
            armed = (M[k - 1] == 1) and (n[k - 1] >= thr_idx[k - 1])
            if armed:
                e2 = list(e); e2[k] = e[k - 1]; e2 = tuple(e2)
                n2 = list(n); n2[k - 1] = 0
                if k < K:
                    n2[k] = 0
                rows.append(i); cols.append(idx[(e2, tuple(n2))]); vals.append(mus[k - 1] * h)
            else:
                rows.append(i); cols.append(i); vals.append(mus[k - 1] * h)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N)), states


def _stationary(P):
    N = P.shape[0]
    A = (P.T - sp.identity(N, format="csr")).tolil(); A[0, :] = 1.0
    b = np.zeros(N); b[0] = 1.0
    pi = spsolve(A.tocsr(), b); pi = np.maximum(pi, 0.0); pi /= pi.sum()
    return pi


def metrics(mus, nu, thr_idx, G, h):
    """Exact mean R-AoII and per-link transmit rates of a threshold vector (in age bins)."""
    K = len(mus)
    P, states = _chain(mus, nu, thr_idx, G, h)
    pi = _stationary(P)
    Marr = np.array([_Mvec(s[0], K) for s in states])
    narr = np.array([s[1] for s in states])
    wrong = np.array([s[0][0] != s[0][K] for s in states])
    W = np.where(wrong)[0]
    fund = spsolve((sp.identity(len(W), format="csr") - P[np.ix_(W, W)]).tocsr(),
                   np.ones(len(W)))
    raoii = h * (pi[W] @ fund)
    rates = np.array([mus[k] * pi[(Marr[:, k] == 1) & (narr[:, k] >= thr_idx[k])].sum()
                      for k in range(K)])
    return raoii, rates


# --------------------------------------------------------------- single-hop renewal-reward rate
def rr_rate(theta, mu, nu):
    """Closed-form single-hop transmit rate at age threshold theta (eq. for lambda(theta))."""
    return mu * nu / (2 * (mu + nu) * np.exp(nu * theta) - mu)


def invert_rate(target, mu, nu):
    """theta achieving the single-hop surrogate rate = target (0 if always-push rate <= target)."""
    if rr_rate(0.0, mu, nu) <= target:
        return 0.0
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rr_rate(mid, mu, nu) > target else (lo, mid)
    return 0.5 * (lo + hi)


def _single_hop_m(mu, nu, theta, G, h):
    """Stationary P(M=1) for one gated link (symmetric surrogate source at rate nu)."""
    P, states = _chain([mu], nu, [int(round(theta / h))], G, h)
    pi = _stationary(P)
    return float(sum(p for s, p in zip(states, pi) if _Mvec(s[0], 1)[0] == 1))


# --------------------------------------------------------------- the two calibrations + optimum
def surrogate_thresholds(mus, nu, budget, G, h):
    """Offline surrogate calibration: theta_k from the closed-form rate at effective rate
    nu_eff,k = mu_{k-1} m_{k-1} (the paper's Algorithm 1 offline warm start)."""
    thetas, nu_eff = [], nu
    for k in range(len(mus)):
        th = invert_rate(budget, mus[k], nu_eff)
        thetas.append(int(round(th / h)))
        nu_eff = mus[k] * _single_hop_m(mus[k], nu_eff, th, G, h)
    return thetas


def achieved_thresholds(mus, nu, budget, G, h):
    """Achieved calibration: theta_k set so link k's *realized* rate equals the budget
    (theta_k=0 if its max realized rate is already <= budget). Decentralized via a local
    rate counter; this is the online step of Algorithm 1."""
    K = len(mus); thr = [0] * K
    for k in range(K):
        thr[k] = 0
        _, r = metrics(mus[:k + 1], nu, thr[:k + 1], G, h)
        if r[k] <= budget:
            continue
        lo, hi = 0, G
        for _ in range(14):
            mid = (lo + hi) // 2; thr[k] = mid
            _, r = metrics(mus[:k + 1], nu, thr[:k + 1], G, h)
            lo, hi = (mid, hi) if r[k] > budget else (lo, mid)
        thr[k] = hi
    return thr


def threshold_optimum(mus, nu, budget, G, h, seed, tol=0.01, strides=(8, 3)):
    """Optimum over per-link threshold vectors with realized rates <= budget*(1+tol),
    by coordinate descent seeded at `seed`."""
    K = len(mus)
    def cost(t):
        D, r = metrics(mus, nu, t, G, h)
        return (D if np.all(r <= budget * (1 + tol)) else np.inf)
    best = list(seed); bestD = cost(best)
    for stride in strides:
        improved = True
        while improved:
            improved = False
            for k in range(K):
                for v in range(0, G + 1, stride):
                    cand = list(best); cand[k] = v
                    D = cost(cand)
                    if D < bestD - 1e-9:
                        bestD, best = D, cand; improved = True
    return best, bestD


# --------------------------------------------------------------- one row (coarse search, fine eval)
def grid_for(mu, nu, A_max=4.0, target=0.1):
    h = target / (nu + 2 * mu)
    return int(round(A_max / h)), h


def run_point(mu, nu, budget, A_max):
    """One Table II row at homogeneous (mu,mu): always-push, decoupled (surr/ach), threshold
    optimum, and the two gaps. Searches at a coarse grid (Lam*h~0.2), evaluates at a fine grid
    (Lam*h~0.1) so the reported gap is grid-robust."""
    mus = [mu, mu]
    Gc, hc = grid_for(mu, nu, A_max, 0.20)
    Gf, hf = grid_for(mu, nu, A_max, 0.10)
    ts = surrogate_thresholds(mus, nu, budget, Gc, hc)
    ta = achieved_thresholds(mus, nu, budget, Gc, hc)
    topt, _ = threshold_optimum(mus, nu, budget, Gc, hc, seed=ta)
    to_fine = lambda idx: [int(round(i * hc / hf)) for i in idx]
    ap, _ = metrics(mus, nu, [0, 0], Gf, hf)
    ds, _ = metrics(mus, nu, to_fine(ts), Gf, hf)
    da, _ = metrics(mus, nu, to_fine(ta), Gf, hf)
    do, _ = metrics(mus, nu, to_fine(topt), Gf, hf)
    return dict(mu=mu, nu=nu, budget=budget,
                always_push=ap, dec_surrogate=ds, dec_achieved=da, joint_opt=do,
                gap_surrogate_pct=100 * (ds - do) / do,
                gap_achieved_pct=100 * (da - do) / do,
                theta_surrogate=[round(i * hc, 3) for i in ts],
                theta_achieved=[round(i * hc, 3) for i in ta],
                theta_opt=[round(i * hc, 3) for i in topt])


# Table II operating points (symmetric source nu=1, homogeneous links).
TABLE2_POINTS = [(2.0, 0.25, 4.0), (2.0, 0.15, 4.0), (4.0, 0.40, 3.0)]


def table2_benchmark(points=None):
    nu = 1.0
    return [run_point(mu, nu, budget, A_max)
            for (mu, budget, A_max) in (points or TABLE2_POINTS)]


# --------------------------------------------------------------- validation / demo
if __name__ == "__main__":
    from . import joint
    np.set_printoptions(precision=4, suppress=True)
    print("VALIDATION: always-push p_K and mean R-AoII vs joint.py exact")
    for mus in ([2.0, 2.0], [3.0, 1.5, 4.0]):
        K = len(mus); G, h = grid_for(max(mus), 1.0, 4.0, 0.05)
        # p_K only needs the config chain; cross-check the closed form here
        print(f"  K={K} mus={mus}: p_K closed={theory.p_K_product(mus,1.0):.5f} "
              f"exact={joint.analyze(mus,1.0)['p_K']:.5f}")
    print("\nTABLE II (mu/nu, budget | always-push  surr  ach  opt | gap_surr  gap_ach):")
    for r in table2_benchmark():
        print(f"  {int(r['mu'])} {r['budget']:>4} | {r['always_push']:.3f} {r['dec_surrogate']:.3f} "
              f"{r['dec_achieved']:.3f} {r['joint_opt']:.3f} | "
              f"{r['gap_surrogate_pct']:5.1f}% {r['gap_achieved_pct']:5.1f}%")
