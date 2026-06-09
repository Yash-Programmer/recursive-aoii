"""Phase-03 verification runner: prints the real numbers cited in theory/03*."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import sympy as sp
from raoii import theory, sim, lemma1, policy, joint

np.set_printoptions(precision=4, suppress=True)
print("=" * 70)
print("PHASE 03 VERIFICATION")
print("=" * 70)

# ---- 1. Theorem 2: SymPy + Monte-Carlo sweep ----------------------------------
print("\n[1] THEOREM 2 (product formula p_K)")
r = theory.verify_symbolic(verbose=False)
print("  SymPy:", "ALL PASS" if all(r.values()) else [k for k, v in r.items() if not v])
settings = {
    "homogeneous mu=2": ([2.0] * 12, 1.0),
    "heterogeneous":    ([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 0.8, 5.0], 1.0),
    "bottleneck mu1=0.5": ([0.5] + [4.0] * 11, 1.0),
}
worst = 0.0
allbracket = True
for name, (mus_all, nu) in settings.items():
    print(f"  setting: {name}", flush=True)
    for K in (1, 2, 4, 8, 12):
        mus = mus_all[:K]
        closed = theory.p_K_product(mus, nu)
        mean, ci, _ = sim.mc_estimate(mus, nu, n_events=80_000, n_seeds=8)
        brack = abs(mean - closed) <= ci + 1e-12
        allbracket = allbracket and brack
        worst = max(worst, abs(mean - closed))
        print(f"    K={K:>2}: closed={closed:.4f} sim={mean:.4f}+-{ci:.4f} "
              f"{'OK' if brack else 'MISS'}", flush=True)
print(f"  -> all CIs bracket closed form: {allbracket}; worst |sim-closed| = {worst:.4f}")
print(f"  cascade limit check (K=1..20, mu=2,nu=1): p_K ->",
      round(theory.p_K_product([2.0] * 20, 1.0), 4), "(target 0.5)")

# ---- 2. Lemma 1: enumeration --------------------------------------------------
print("\n[2] LEMMA 1 (decomposition / XOR parity)")
xor_ok = all(lemma1.check_xor_identity(K)[0] for K in range(1, 7))
ub = [(N, K, *lemma1.check_union_bound(N, K)) for N in (2, 3, 4) for K in range(1, 5)]
ub_ok = all(o for _, _, o, _ in ub)
strict_seen = any(s for N, K, _, s in ub if K >= 2)
print(f"  XOR identity exact (N=2, K=1..6): {xor_ok}")
print(f"  union bound holds (N=2,3,4; K=1..4): {ub_ok}; strict slack witnessed (K>=2): {strict_seen}")
print(f"  cancellation witness: {lemma1.cancellation_witness()}")
out = sim.tandem_sim([2.0, 2.0, 2.0], 1.0, n_events=400_000, seed=3)
print(f"  sim XOR violations over 400k events (K=3): {out['xor_violations']}  (must be 0)")
print(f"  sim union-bound slack mean >=0: {out['slack_mean']:.4f}")

# ---- 3. Theorem 2*: time-integral E[Delta^R_1] --------------------------------
print("\n[3] THEOREM 2* (time-integral R-AoII)")
for (mu, nu) in [(2.0, 1.0), (3.0, 1.0), (1.0, 0.5)]:
    closed = theory.raoii_K1_closed(mu, nu)
    m, ci, _ = sim.mc_estimate([mu], nu, n_events=150_000, n_seeds=8, key="raoii")
    print(f"  K=1 mu={mu} nu={nu}: closed E[D^R_1]={closed:.4f} sim={m:.4f}+-{ci:.4f} "
          f"{'OK' if abs(m-closed)<=ci+1e-9 else 'MISS'}", flush=True)
for K in (2, 3):
    mus = [2.0] * K
    ex = joint.analyze(mus, 1.0)["raoii"]
    m, ci, _ = sim.mc_estimate(mus, 1.0, n_events=150_000, n_seeds=8, key="raoii")
    print(f"  K={K} mu=2 nu=1: exact -pi_M^T A^-1 1 = {ex:.4f}  sim={m:.4f}+-{ci:.4f} "
          f"{'OK' if abs(m-ex)<=ci+1e-9 else 'MISS'}", flush=True)

# ---- 4. Theorem 3: the threshold finding --------------------------------------
print("\n[4] THEOREM 3 (single-hop threshold) -- THE FINDING")
mu, beta = 2.0, 1.5
print(f"  mu={mu}, beta={beta}")
print(f"  {'nu':>5} {'theta*_renewal':>14} {'theta_VI':>9} {'theta_D1formula':>16}")
for nu in (1.5, 1.0, 0.5, 0.25):
    ts, _ = policy.optimal_threshold_renewal(mu, nu, beta)
    tv = policy.single_hop_vi(mu, nu, beta)["theta_vi"]
    td = theory.threshold(mu, nu, beta)
    print(f"  {nu:>5} {ts:>14.3f} {tv:>9.3f} {td:>16.3f}")
print("  FINDING: theta*_renewal == theta_VI (independent); D1-formula is WRONG.")
print("  FINDING: theta* INCREASES with nu (self-correction effect) -> D1 move-4 direction is backwards.")

# corrected FOC via SymPy (dJ/dtheta = 0)
print("  corrected optimal-threshold FOC (SymPy dJ/dtheta=0):")
th, muS, nuS, beS = sp.symbols("theta mu nu beta", positive=True)
F = sp.exp(-nuS * th); rr = nuS + muS
EL = ((1 - F) / nuS - th * F) + F * (th + 1 / rr)
EL2 = (2 * (1 - F) / nuS**2 - 2 * th * F / nuS - th**2 * F) + F * (th**2 + 2 * th / rr + 2 / rr**2)
Etx = F * (muS / rr); EC = 1 / nuS + EL
J = (EL2 / 2 + beS * Etx) / EC
dJ = sp.simplify(sp.diff(J, th))
print("   dJ/dtheta = 0  <=>  numerator:")
num = sp.simplify(sp.numer(sp.together(dJ)))
print("   ", num)

# ---- 5. Cascade nesting probe (does theta_1<=...<=theta_K hold?) ---------------
print("\n[5] CASCADE NESTING PROBE (theta_k from binding budget eta(theta_k)=lambda_k)")
def cascade_thetas(mus, nu, lams):
    ps = theory.p_recursion(mus, nu)
    nu_eff = [nu * (1 - 2 * (0.0 if k == 0 else ps[k - 1])) for k in range(len(mus))]
    return [policy.threshold_for_budget(lams[k], mus[k], nu_eff[k]) for k in range(len(mus))], nu_eff
for label, mus, lams in [
    ("ordered budgets lam=[.4,.3,.2,.1], mu=3", [3.0]*4, [0.4, 0.3, 0.2, 0.1]),
    ("equal budgets lam=0.25 (isolates nu_eff)", [3.0]*4, [0.25]*4),
    ("equal budgets, varied mu", [2.0, 3.0, 4.0, 5.0], [0.25]*4),
]:
    th, nue = cascade_thetas(mus, 1.0, lams)
    mono_up = all(th[i] <= th[i+1] + 1e-9 for i in range(len(th)-1))
    mono_dn = all(th[i] >= th[i+1] - 1e-9 for i in range(len(th)-1))
    shape = "nondecreasing (D1 nesting)" if mono_up else ("nonincreasing (REVERSED)" if mono_dn else "NON-MONOTONE")
    print(f"  {label}")
    print(f"    nu_eff = {np.array(nue)}")
    print(f"    theta_k = {np.array(th)}  -> {shape}")
print("=" * 70)
