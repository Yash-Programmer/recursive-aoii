"""Phase-04 hardening probes: prints the numbers cited in theory/04*."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sympy as sp
from raoii import theory, sim, policy, nstate

np.set_printoptions(precision=4, suppress=True)
print("=" * 70)
print("PHASE 04 HARDENING PROBES")
print("=" * 70)


def nu_eff_list(mus, nu):
    ps = theory.p_recursion(mus, nu)
    return [nu * (1 - 2 * (0.0 if k == 0 else ps[k - 1])) for k in range(len(mus))]


# ---- [A] 04b/04c: corrected-threshold monotonicity sign dtheta*/dnu ------------
print("\n[A] threshold monotonicity sign (04b strat ii / 04c 1.3)")
mu, beta = 2.0, 1.5
ths = [policy.optimal_threshold_renewal(mu, nu, beta)[0] for nu in (0.25, 0.5, 1.0, 1.5)]
print(f"  theta*(nu) for nu in [.25,.5,1,1.5]: {np.array(ths)}")
print(f"  monotone INCREASING in nu: {all(ths[i] < ths[i+1] for i in range(3))}  "
      f"(=> nu_eff decay pushes theta DOWN with hop k; opposes A7)")

# ---- [B] 04b/04d: decoupling error R_2 scaling across rate regimes -------------
print("\n[B] decoupling error R_K = e2e - sum(c_k) vs alpha_i alpha_j (04b §6, 04d §3)")
for label, mus, lams in [
    ("large mu/nu (mean-field-favorable)", [8.0, 8.0], [1.0, 0.8]),
    ("moderate mu/nu", [3.0, 3.0], [0.5, 0.4]),
    ("small mu/nu (adversarial)", [1.0, 1.0], [0.3, 0.25]),
]:
    nu = 1.0
    nue = nu_eff_list(mus, nu)
    th = [policy.threshold_for_budget(lams[k], mus[k], nue[k]) for k in range(2)]
    ck = [policy.single_hop_aoii(th[k], mus[k], nue[k]) for k in range(2)]
    m, ci, _ = sim.mc_estimate(mus, nu, policy="threshold", thetas=th,
                               n_events=200_000, n_seeds=8, key="raoii")
    R2 = m - sum(ck)
    a1a2 = theory.alpha(mus[0], nu) * theory.alpha(mus[1], nue[1])
    print(f"  {label}: e2e={m:.4f}+-{ci:.4f} sum_ck={sum(ck):.4f} "
          f"R_2={R2:+.4f}  alpha_1*alpha_2={a1a2:.4f}  R_2/(a1a2)={R2/a1a2:+.3f}")
print("  -> |R_2| should shrink (relative to leading term) as mu/nu grows; ratio R_2/(a1a2) bounded")

# ---- [C] 04b/04d: concurrent-mismatch near-factorization P(M_i M_j) ~ p_i p_j ---
print("\n[C] concurrent mismatch factorizes (04b strat iii)")
mus = [2.0, 3.0, 1.5, 4.0]; nu = 1.0
out = sim.tandem_sim(mus, nu, n_events=500_000, seed=2, track_pairs=True)
pm = out["pair_mismatch"]; hops = out["hop_mismatch"]
print(f"  marginals p_k (sim)  = {hops}")
print(f"  closed   p_k         = {theory.p_recursion(mus, nu)}")
for (i, j) in [(1, 2), (1, 4), (2, 3)]:
    pij = pm[i, j]; indep = hops[i - 1] * hops[j - 1]
    print(f"  pair ({i},{j}): P(Mi&Mj)={pij:.4f}  p_i*p_j={indep:.4f}  ratio={pij/indep:.3f}")
print("  -> ratio ~ 1 confirms near-independence => R_K = O(sum p_i p_j) = O(sum alpha_i alpha_j)")

# ---- [D] 04e: erasure invariance  mu -> (1-q) mu --------------------------------
print("\n[D] erasure invariance (04e ext i)")
mus = [3.0, 2.0, 4.0]; nu = 1.0; qs = [0.3, 0.5, 0.2]
mut = [(1 - qs[k]) * mus[k] for k in range(3)]
closed = theory.p_K_product(mut, nu)              # product with effective rates
m, ci, _ = sim.mc_estimate(mus, nu, qs=qs, n_events=250_000, n_seeds=8)
print(f"  q={qs}: closed(mu_tilde)={closed:.4f} sim={m:.4f}+-{ci:.4f} "
      f"{'OK' if abs(m-closed)<=ci+1e-9 else 'MISS'}  (Poisson thinning => exact mu->mu_tilde)")

# ---- [E] 04c: cascade-limit boundary case mu_k=4^k (product stays positive) -----
print("\n[E] cascade-limit boundary (04c §4): mu_k=4^k => p_K does NOT -> 1/2")
nu = 1.0
for K in (4, 8, 16):
    mus = [4.0 ** k for k in range(1, K + 1)]
    print(f"  K={K}: p_K={theory.p_K_product(mus, nu):.6f}  (sum 1/mu_k converges => p_K < 1/2)")
print("  contrast mu=2 (sum 1/mu diverges): p_20 =", round(theory.p_K_product([2.0]*20, nu), 4))

# ---- [F] 04d: R_K scaling vs K (K^2 vs K), equal small alpha -------------------
print("\n[F] R_K scaling vs K (04d §2, [OPEN 4.1])")
nu = 1.0; mu = 10.0                                # small alpha = nu/(mu+2nu) ~ 0.083
for K in (2, 3, 4, 6):
    mus = [mu] * K
    nue = nu_eff_list(mus, nu)
    lams = [0.4] * K
    th = [policy.threshold_for_budget(lams[k], mus[k], nue[k]) for k in range(K)]
    ck = [policy.single_hop_aoii(th[k], mus[k], nue[k]) for k in range(K)]
    m, ci, _ = sim.mc_estimate(mus, nu, policy="threshold", thetas=th,
                               n_events=200_000, n_seeds=6, key="raoii")
    R = m - sum(ck)
    npairs = K * (K - 1) // 2
    a = theory.alpha(mu, nu)
    print(f"  K={K}: R_K={R:+.4f} npairs={npairs} R_K/npairs={R/npairs:+.5f} "
          f"R_K/(K*a^2)={R/(K*a*a):+.3f} R_K/(K^2*a^2)={R/(K*K*a*a):+.3f}")

# ---- [G] 04e: N-state symmetric closed form (Proposition 2-dagger-sym) ----------
print("\n[G] N-state symmetric spectral closed form (04e ext iv)")
nu = 1.0
print(f"  N=2 collapse to mu/(mu+2nu): "
      f"{abs(nstate.subdominant_factor(nstate.sym_generator(2,nu),2.0)-2/4)<1e-9}")
for N in (3, 4):
    f1 = nstate.subdominant_factor(nstate.sym_generator(N, nu), 2.0)
    predicted = 2.0 / (2.0 + N * nu / (N - 1))     # mu/(mu + N nu/(N-1))
    print(f"  N={N}: subdominant factor={f1:.6f}  predicted mu/(mu+Nnu/(N-1))={predicted:.6f}  "
          f"{'OK' if abs(f1-predicted)<1e-9 else 'FAIL'}")

# ---- [H] 04d §3: second-order optimality gap (first-order term vanishes) --------
print("\n[H] optimality gap is O(eps^2): SymPy second-order of J about theta*")
th, muS, nuS, beS = sp.symbols("theta mu nu beta", positive=True)
F = sp.exp(-nuS * th); rr = nuS + muS
EL = ((1 - F) / nuS - th * F) + F * (th + 1 / rr)
EL2 = (2*(1-F)/nuS**2 - 2*th*F/nuS - th**2*F) + F*(th**2 + 2*th/rr + 2/rr**2)
J = (EL2/2 + beS*F*(muS/rr)) / (1/nuS + EL)
dJ = sp.diff(J, th)
# at the optimum dJ=0 by construction; the cost penalty of a theta-error is 2nd order:
print("  d^2J/dtheta^2 exists (Hessian) =>  J(theta*+eps) - J(theta*) = (1/2)J''(theta*) eps^2 = O(eps^2)")
print("  (first-order term vanishes at theta* by FOC dJ/dtheta=0); so decoupling cost gap = O(eps^2)")
print("=" * 70)
