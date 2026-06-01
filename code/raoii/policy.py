"""Single-hop AoII SMDP via uniformized relative value iteration.

Confirms (for Theorem 3): the optimal single-hop policy is a threshold on the
disagreement age tau; the recovered threshold matches the closed form
theta = log(1/(1-2 beta alpha))/(mu+2nu); and theta is non-increasing in the
effective source rate nu (the monotonicity that drives the nesting).

State (m, i): m in {0,1} disagreement of (source, estimate); i = age bin.
Source = effective 2-state flip rate nu_eff; channel rate mu; transmit price beta.
Full joint K=2,3 value iteration (the decoupling-gap probe) is experiment E3.
"""
import numpy as np
from . import theory


def single_hop_vi(mu, nu, beta, dt=0.005, tau_max=12.0, tol=1e-11, max_iter=500_000):
    """Fine-timestep uniformization (dt small; per-step event prob = rate*dt).

    States: S0 = matched (age irrelevant); (1,i) = mismatched at age i*dt.
    Matched: flip (nu*dt) -> (1,0); else stay; cost 0; opportunities never transmit.
    Mismatched age tau=i*dt: cost tau*dt; flip (nu*dt) -> S0; opportunity (mu*dt) ->
      DECIDE transmit (cost beta, -> S0) or wait (-> age i+1); else (-> age i+1).
    Threshold = least age at which transmit is chosen at an opportunity.
    """
    assert (nu + mu) * dt < 1.0
    n = int(round(tau_max / dt))
    pf, po = nu * dt, mu * dt
    pstay = 1.0 - pf - po
    hS0 = 0.0
    hM = np.zeros(n)                         # h(1,i)
    g = 0.0
    for _ in range(max_iter):
        # mismatched sweep, backward in age (h(1,i) depends on h(1,i+1))
        hMn = np.zeros(n)
        transmit_at = np.zeros(n, dtype=np.int8)
        for i in range(n - 1, -1, -1):
            tau = i * dt
            inext = min(i + 1, n - 1)
            cont_wait = hMn[inext] if i < n - 1 else hM[inext]
            opp_val = min(beta + hS0, cont_wait)     # transmit vs wait at an opportunity
            transmit_at[i] = 1 if (beta + hS0) <= cont_wait else 0
            hMn[i] = tau * dt + pf * hS0 + po * opp_val + pstay * cont_wait - g * dt
        # matched update
        hS0n = pf * hMn[0] + (1 - pf) * hS0 - g * dt
        gn = g + hS0n                         # drive average cost so that h(S0) -> 0
        hMn = hMn - hS0n
        hS0n = 0.0
        if np.max(np.abs(hMn - hM)) < tol and abs(gn - g) < tol:
            hM, hS0, g = hMn, hS0n, gn
            break
        hM, hS0, g = hMn, hS0n, gn
    trans_bins = np.where(transmit_at == 1)[0]
    theta_vi = trans_bins[0] * dt if len(trans_bins) else np.inf
    return {"theta_vi": theta_vi, "g": g, "n": n}


def single_hop_cost_renewal(theta, mu, nu, beta):
    """Closed-form average cost J(theta)=E[AoII]+beta*rate for a threshold-theta
    single-hop policy (2-state symmetric, source rate nu, channel mu), via
    renewal-reward over matched/mismatched cycles. Ground truth for the optimum.

    Mismatch ends by source flip (rate nu) any time, or by an accepted opportunity
    (rate mu) once age>=theta. F=e^{-nu theta}=P(survive to theta).
    """
    F = np.exp(-nu * theta)
    # branch A (flip before theta): moments of the truncated Exp(nu) on (0,theta)
    EL_A = (1 - F) / nu - theta * F
    EL2_A = 2 * (1 - F) / nu**2 - 2 * theta * F / nu - theta**2 * F
    # branch B (survive to theta): L = theta + Exp(nu+mu)
    r = nu + mu
    EL_B = F * (theta + 1.0 / r)
    EL2_B = F * (theta**2 + 2 * theta / r + 2.0 / r**2)
    EL = EL_A + EL_B
    EL2 = EL2_A + EL2_B
    E_tx = F * (mu / r)                       # P(ends by accepted opportunity)
    EC = 1.0 / nu + EL                        # matched Exp(nu) + mismatch
    return (0.5 * EL2 + beta * E_tx) / EC


def attempt_rate_at_threshold(theta, mu, nu):
    """Achieved transmit-attempt rate eta(theta) of a threshold-theta policy
    (independent of beta): eta = E[transmits per cycle]/E[cycle]. Monotone-decreasing
    in theta, so a binding budget lambda pins theta via eta(theta)=lambda."""
    F = np.exp(-nu * theta)
    r = nu + mu
    EL_A = (1 - F) / nu - theta * F
    EL_B = F * (theta + 1.0 / r)
    EC = 1.0 / nu + EL_A + EL_B
    E_tx = F * (mu / r)
    return E_tx / EC


def threshold_for_budget(lam, mu, nu, hi=40.0):
    """Solve eta(theta)=lam for theta by bisection (eta decreasing in theta)."""
    if lam >= attempt_rate_at_threshold(0.0, mu, nu):
        return 0.0
    lo, hi = 0.0, hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if attempt_rate_at_threshold(mid, mu, nu) > lam:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def single_hop_aoii(theta, mu, nu):
    """The AoII cost component E[Delta_k] (no transmit price) of a threshold-theta
    single-hop policy = (1/2)E[L^2]/E[C]. This is the per-hop c_k of Theorem 4."""
    F = np.exp(-nu * theta)
    r = nu + mu
    EL_A = (1 - F) / nu - theta * F
    EL2_A = 2 * (1 - F) / nu**2 - 2 * theta * F / nu - theta**2 * F
    EL_B = F * (theta + 1.0 / r)
    EL2_B = F * (theta**2 + 2 * theta / r + 2.0 / r**2)
    EC = 1.0 / nu + EL_A + EL_B
    return (0.5 * (EL2_A + EL2_B)) / EC


def optimal_threshold_renewal(mu, nu, beta, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 15.0, 6001)
    J = np.array([single_hop_cost_renewal(t, mu, nu, beta) for t in grid])
    i = int(np.argmin(J))
    return grid[i], J[i]


if __name__ == "__main__":
    mu, beta = 2.0, 1.5
    print("Single-hop optimal threshold: renewal-reward ground truth vs VI vs D1-formula")
    print(f"  mu={mu}, beta={beta}")
    print(f"  {'nu':>5} {'theta*_renewal':>14} {'theta_VI':>9} {'theta_D1formula':>16} {'alpha':>7}")
    for nu in (1.5, 1.0, 0.5, 0.25):
        th_star, _ = optimal_threshold_renewal(mu, nu, beta)
        th_vi = single_hop_vi(mu, nu, beta)["theta_vi"]
        th_d1 = theory.threshold(mu, nu, beta)
        print(f"  {nu:>5} {th_star:>14.3f} {th_vi:>9.3f} {th_d1:>16.3f} {theory.alpha(mu,nu):>7.3f}")
    print("  (theta*_renewal is ground truth: argmin of the exact J(theta).)")
