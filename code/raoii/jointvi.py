"""E3: joint-vs-decoupled threshold optimization (the decoupling-tightness probe for 04b).

The optimal per-hop policy is a threshold (Prop. 3 / Cosandal-Akar-Ulukus), so we optimize
over the proven-optimal THRESHOLD CLASS rather than run an intractable 3-D age-grid Bellman
solver. For a fixed common Lagrange price beta, the end-to-end Lagrangian is

    J(theta) = E[Delta^R_K]  +  beta * sum_k (attempt rate_k),

minimized two ways:
  * DECOUPLED: theta_k^dec = argmin of the single-hop Lagrangian at effective rate nu_eff,k;
  * JOINT:     theta^joint = argmin over a grid of the SIMULATED end-to-end J.

The decoupling gap is |theta^joint - theta^dec| (threshold shift eps) and
J(theta^dec) - J(theta^joint) (cost penalty Delta_opt, predicted O(eps^2) by 04d).
"""
import itertools
import numpy as np
from . import theory, sim, policy


def nu_eff_list(mus, nu):
    ps = theory.p_recursion(mus, nu)
    return [nu * (1 - 2 * (0.0 if k == 0 else ps[k - 1])) for k in range(len(mus))]


def decoupled_thresholds(mus, nu, beta):
    nue = nu_eff_list(mus, nu)
    return np.array([policy.optimal_threshold_renewal(mus[k], nue[k], beta)[0]
                     for k in range(len(mus))]), nue


def end_to_end_lagrangian(mus, nu, thetas, beta, n_events, n_seeds):
    """J(theta) = E[Delta^R_K] + beta * sum_k attempt_rate_k, averaged over seeds."""
    raoii = np.zeros(n_seeds); rate = np.zeros(n_seeds)
    for s in range(n_seeds):
        o = sim.tandem_sim(mus, nu, policy="threshold", thetas=thetas,
                           n_events=n_events, seed=s)
        raoii[s] = o["raoii"]; rate[s] = o["tx_rate"].sum()
    J = raoii + beta * rate
    return J.mean(), raoii.mean(), rate.mean()


def joint_optimize(mus, nu, beta, grid_pts=15, theta_max=None, n_events=300_000, n_seeds=4):
    """Grid-search the joint-optimal threshold vector minimizing the end-to-end Lagrangian."""
    K = len(mus)
    th_dec, nue = decoupled_thresholds(mus, nu, beta)
    if theta_max is None:
        theta_max = float(max(3.0, 1.6 * th_dec.max()))
    axis = np.linspace(0.0, theta_max, grid_pts)
    best = (np.inf, None, None, None)
    for combo in itertools.product(axis, repeat=K):
        th = np.array(combo)
        J, ra, rt = end_to_end_lagrangian(mus, nu, th, beta, n_events, n_seeds)
        if J < best[0]:
            best = (J, th, ra, rt)
    J_joint, th_joint, _, _ = best
    J_dec, ra_dec, _ = end_to_end_lagrangian(mus, nu, th_dec, beta, n_events, n_seeds)
    return {
        "theta_dec": th_dec, "theta_joint": th_joint, "nu_eff": np.array(nue),
        "shift": np.abs(th_joint - th_dec), "max_shift": float(np.abs(th_joint - th_dec).max()),
        "J_dec": J_dec, "J_joint": J_joint, "cost_gap": float(J_dec - J_joint),
        "alpha_max": float(max(theory.alpha(mus[k], nue[k]) for k in range(K))),
        "grid_step": float(axis[1] - axis[0]),
    }


if __name__ == "__main__":
    for mus, nu, beta in [([3.0, 3.0], 1.0, 1.0), ([2.0, 2.0], 1.0, 1.5)]:
        r = joint_optimize(mus, nu, beta, grid_pts=13, n_events=300_000, n_seeds=4)
        print(f"K=2 mus={mus} nu={nu} beta={beta}")
        print(f"  theta_dec ={np.round(r['theta_dec'],3)}  theta_joint={np.round(r['theta_joint'],3)}")
        print(f"  max threshold shift eps={r['max_shift']:.3f} (grid step {r['grid_step']:.3f}, "
              f"alpha_max={r['alpha_max']:.3f})")
        print(f"  cost gap J_dec-J_joint={r['cost_gap']:+.5f}  (predicted O(eps^2)={r['max_shift']**2:.4f})")
