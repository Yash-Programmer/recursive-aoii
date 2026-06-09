"""Exact unrestricted (genie / global-state) optimum for the constrained R-AoII cascade.

This is the benchmark behind the paper's *price of decentralization*. The decentralized per-link
threshold class (benchmark.py, Table II) is the optimal IMPLEMENTABLE structure; here we compute
the optimum over ALL causal policies that may condition on the global state -- in particular on the
end-to-end (monitor) wrong-run age, which no relay can observe (it requires the source). The gap is
the value of global coordination.

State reduction. For the genie the sufficient statistic is the disagreement pattern M=(M1..MK),
Mk = Xhat_{k-1} ^ Xhat_k (Xhat_0 = source X), together with the monitor wrong-run age a (time since
X = Xhat_K); wrong = XOR_k Mk. The per-link ages used by the threshold class do NOT enter dynamics
or cost, so the genie optimum lives on only 2^K (G+1) states. Uniformized step h (Lam = nu+sum mu):
  * no event (1-Lam h): a advances while wrong
  * source   (nu h): M1 ^= 1 (monitor wrong always toggles)  -> a resets
  * link k   (mu_k h): if transmit (armed Mk=1): Mk->0, M_{k+1}->Mk^M_{k+1} (k<K)
The genie may withhold a forward that would propagate an upstream error onto a monitor made correct
by error-cancellation (M with an even number of disagreements). R-AoII is exact via the killed-chain
fundamental matrix (no age-discretization bias); per-link USEFUL rates count armed transmits only
(Table II convention). The constrained optimum (each link rate <= budget) is found by Lagrangian
relative value iteration with a scalar-price bisection to the rate=budget boundary.

Validated (validate()): always-push reproduces the exact closed-form mean R-AoII to machine
precision and matches joint.py; never-transmit hits the saturation bound 1/(2nu).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from itertools import product
from . import theory


def _wrong(M):
    w = 0
    for m in M:
        w ^= m
    return w


def _states(K, G):
    st = []
    for M in product((0, 1), repeat=K):
        for a in (range(G + 1) if _wrong(M) else [0]):
            st.append((M, a))
    return st, {s: i for i, s in enumerate(st)}


def _newage(wM, wMp, a, G):
    """Wrong-run age after a step (h elapses on every branch): 0 if now correct or just became
    wrong, else advance one bin."""
    if not wMp:
        return 0
    return min(a + 1, G) if wM else 0


def _link_tx_M(M, k):
    """Pattern after a useful transmit on link k (1-indexed): Mk->0, M_{k+1}->Mk^M_{k+1}."""
    M = list(M)
    mk = M[k - 1]; M[k - 1] = 0
    if k < len(M):
        M[k] ^= mk
    return tuple(M)


def _precompute(K, states, idx, G):
    """Vectorized transition-index arrays and per-link armed masks (built once per (K,G))."""
    N = len(states)
    i_noev = np.empty(N, int); i_src = np.empty(N, int)
    age = np.empty(N, float); wrong = np.zeros(N, bool)
    i_idle = [np.empty(N, int) for _ in range(K)]
    i_tx = [np.empty(N, int) for _ in range(K)]
    arm = [np.zeros(N, bool) for _ in range(K)]
    for s, i in idx.items():
        M, a = s; w = _wrong(M)
        i_noev[i] = idx[(M, _newage(w, w, a, G))]
        Ms = (M[0] ^ 1,) + M[1:]; i_src[i] = idx[(Ms, _newage(w, _wrong(Ms), a, G))]
        age[i] = a; wrong[i] = bool(w)
        for k in range(1, K + 1):
            i_idle[k - 1][i] = idx[(M, _newage(w, w, a, G))]
            Mt = _link_tx_M(M, k); i_tx[k - 1][i] = idx[(Mt, _newage(w, _wrong(Mt), a, G))]
            arm[k - 1][i] = (M[k - 1] == 1)
    return dict(N=N, K=K, i_noev=i_noev, i_src=i_src, i_idle=i_idle, i_tx=i_tx,
                age=age, wrong=wrong, arm=arm)


def _build_P(mus, nu, h, pre, acts):
    K = pre["K"]; Lh = (nu + sum(mus)) * h; N = pre["N"]
    base = np.arange(N)
    rows = [base, base]; cols = [pre["i_noev"], pre["i_src"]]
    vals = [np.full(N, 1 - Lh), np.full(N, nu * h)]
    for k in range(K):
        tk = np.where(acts[k], pre["i_tx"][k], pre["i_idle"][k])
        rows.append(base); cols.append(tk); vals.append(np.full(N, mus[k] * h))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(N, N))


def _stationary(P):
    N = P.shape[0]
    A = (P.T - sp.identity(N, format="csr")).tolil(); A[0, :] = 1.0
    b = np.zeros(N); b[0] = 1.0
    pi = spsolve(A.tocsr(), b); pi = np.maximum(pi, 0.0)
    return pi / pi.sum()


def evaluate(mus, nu, h, pre, acts):
    """Exact mean R-AoII (killed-chain fundamental matrix) and per-link useful (armed) rates."""
    P = _build_P(mus, nu, h, pre, acts)
    pi = _stationary(P)
    W = np.where(pre["wrong"])[0]
    fund = spsolve((sp.identity(len(W), format="csr") - P[np.ix_(W, W)]).tocsr(), np.ones(len(W)))
    raoii = h * (pi[W] @ fund)
    rates = np.array([mus[k] * pi[acts[k] & pre["arm"][k]].sum() for k in range(pre["K"])])
    return raoii, rates, pi


def lagrangian_opt(mus, nu, h, beta, pre, iters=5000, tol=1e-11):
    """Vectorized relative value iteration: min R-AoII + sum_k beta_k * rate_k. A transmit option
    exists only when armed (idle forced otherwise, since a no-op forward cannot help)."""
    K = pre["K"]; Lh = (nu + sum(mus)) * h; N = pre["N"]
    cost = h * pre["age"] * pre["wrong"]
    V = np.zeros(N)
    for _ in range(iters):
        Vn = cost + (1 - Lh) * V[pre["i_noev"]] + nu * h * V[pre["i_src"]]
        for k in range(K):
            ci = V[pre["i_idle"][k]]; ct = beta[k] + V[pre["i_tx"][k]]
            Vn = Vn + mus[k] * h * np.where(pre["arm"][k], np.minimum(ci, ct), ci)
        g = Vn[0]; Vn -= g
        if np.max(np.abs(Vn - V)) < tol:
            V = Vn; break
        V = Vn
    acts = [pre["arm"][k] & ((beta[k] + V[pre["i_tx"][k]]) < V[pre["i_idle"][k]]) for k in range(K)]
    return acts


def grid_for(mu, nu, A_max=8.0, target_Lh=0.10):
    h = target_Lh / (nu + 2 * mu)
    return int(round(A_max / h)), h


def genie_opt(mus, nu, budget, A_max=8.0, target_Lh=0.10):
    """Min mean R-AoII s.t. each link's useful rate <= budget, over all (M, monitor-age) policies.
    R-AoII decreases with rate so the optimum is at the boundary; for homogeneous links the per-link
    rates move together, so bisect a scalar price to the largest feasible point. Returns
    (raoii, rates, beta), the unconstrained (beta=0) optimum, and the grid (G,h)."""
    K = len(mus); G, h = grid_for(max(mus), nu, A_max, target_Lh)
    states, idx = _states(K, G); pre = _precompute(K, states, idx, G)

    def trial(b):
        acts = lagrangian_opt(mus, nu, h, np.full(K, b), pre)
        ra, rates, _ = evaluate(mus, nu, h, pre, acts)
        return ra, rates

    ra0, r0 = trial(0.0)                                  # unconstrained genie
    b_hi = 1.0
    while max(trial(b_hi)[1]) > budget and b_hi < 1e4:
        b_hi *= 2.0
    lo, hi = 0.0, b_hi
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if max(trial(mid)[1]) > budget else (lo, mid)
    ra, rates = trial(hi)
    return dict(raoii=ra, rates=rates, beta=hi, raoii_unconstrained=ra0,
                rates_unconstrained=r0, G=G, h=h)


def always_push(mus, nu, A_max=8.0, target_Lh=0.10):
    """Mean R-AoII under always-push (forward at every armed opportunity)."""
    K = len(mus); G, h = grid_for(max(mus), nu, A_max, target_Lh)
    states, idx = _states(K, G); pre = _precompute(K, states, idx, G)
    ra, _, _ = evaluate(mus, nu, h, pre, [np.ones(pre["N"], bool) for _ in mus])
    return ra


def link_tx_fraction(mus, nu, budget, link, patterns, A_max=8.0, target_Lh=0.10):
    """For the constrained genie optimum, the transmit fraction of `link` (1-indexed) over its
    armed age-states, for each pattern M in `patterns`. Reveals the cancellation-withholding rule."""
    K = len(mus); res = genie_opt(mus, nu, budget, A_max, target_Lh)
    G, h = res["G"], res["h"]
    states, idx = _states(K, G); pre = _precompute(K, states, idx, G)
    acts = lagrangian_opt(mus, nu, h, np.full(K, res["beta"]), pre)
    a = acts[link - 1]; arm = pre["arm"][link - 1]
    out = {}
    for M in patterns:
        ix = [i for i, s in enumerate(states) if s[0] == tuple(M) and arm[i]]
        out[tuple(M)] = float(np.mean([a[i] for i in ix])) if ix else float("nan")
    return out


def validate(verbose=False):
    """always-push reproduces the exact closed form; never-transmit hits 1/(2nu)."""
    ok = True
    for mus, exact in ([2.0, 2.0], 25 / 144), ([4.0, 4.0], 0.08):
        ra = always_push(mus, 1.0, target_Lh=0.10)
        good = abs(ra - exact) < 1e-9
        ok = ok and good
        if verbose:
            print(f"  always-push K=2 mus={mus}: R-AoII={ra:.6f} exact={exact:.6f} {'OK' if good else 'FAIL'}")
    # never transmit -> saturation 1/(2nu)
    G, h = grid_for(2.0, 1.0, 8.0, 0.10)
    states, idx = _states(2, G); pre = _precompute(2, states, idx, G)
    acts = lagrangian_opt([2.0, 2.0], 1.0, h, [1e6, 1e6], pre)
    ra_none, _, _ = evaluate([2.0, 2.0], 1.0, h, pre, acts)
    sat = abs(ra_none - 0.5) < 1e-3
    ok = ok and sat
    if verbose:
        print(f"  never-transmit: R-AoII={ra_none:.4f} saturation 1/(2nu)=0.5 {'OK' if sat else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    print("genie.py validation:", "PASS" if validate(verbose=True) else "FAIL")
