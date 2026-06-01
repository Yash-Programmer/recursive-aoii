"""E5: heavy-tail (Pareto) opportunity sensitivity + tree path-product.

The product formula assumes Poisson (memoryless) opportunities, so PASTA + the Exp(mu)
residual hold. Under heavy-tailed (Pareto) inter-arrivals the residual is no longer Exp(mu),
so the product formula need not hold exactly. This per-process simulator (CTMC source +
per-link Pareto renewals, always-push) quantifies the deviation.

Pareto inter-arrival with shape a and mean matched to 1/mu (scale x_m = (1/mu)*(a-1)/a).
a in (2,3): finite mean, heavy (infinite variance for a<=2; we use a=2.5).
"""
import numpy as np
from numba import njit
from . import theory


@njit(cache=False)
def _heavytail_kernel(K, nu, mus, a_shape, n_events, burnin, seed):
    np.random.seed(seed)
    xhat = np.zeros(K + 1, np.int8)
    next_t = np.zeros(K + 1)                       # next event time: index 0 = source flip
    # init schedule
    next_t[0] = np.random.exponential(1.0 / nu)
    xm = np.empty(K)
    for k in range(K):
        xm[k] = (1.0 / mus[k]) * (a_shape - 1.0) / a_shape
        next_t[k + 1] = xm[k] / np.random.random() ** (1.0 / a_shape)   # Pareto inverse-CDF
    t = 0.0
    last_agree = 0.0
    total = 0.0
    mism = 0.0
    raoii = 0.0
    for it in range(n_events):
        # find next event (min over K+1 processes)
        jmin = 0
        tmin = next_t[0]
        for k in range(1, K + 1):
            if next_t[k] < tmin:
                tmin = next_t[k]; jmin = k
        dt = tmin - t
        counting = it >= burnin
        m = 1 if xhat[0] != xhat[K] else 0
        if counting:
            total += dt
            if m:
                a0 = t - last_agree; a1 = tmin - last_agree
                raoii += 0.5 * (a1 * a1 - a0 * a0); mism += dt
        if m == 0:
            last_agree = tmin
        t = tmin
        if jmin == 0:                              # source flip
            xhat[0] ^= 1
            next_t[0] = t + np.random.exponential(1.0 / nu)
        else:                                      # link jmin always-push refresh
            xhat[jmin] = xhat[jmin - 1]
            next_t[jmin] = t + xm[jmin - 1] / np.random.random() ** (1.0 / a_shape)
        if xhat[0] == xhat[K]:
            last_agree = t
    return mism / total, raoii / total


def heavytail_pK(mus, nu, a_shape=2.5, n_events=500_000, n_seeds=12):
    mus = np.asarray(mus, float)
    vals = np.array([_heavytail_kernel(len(mus), float(nu), mus, float(a_shape),
                                        n_events, n_events // 10, s)[0] for s in range(n_seeds)])
    return vals.mean(), 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))


if __name__ == "__main__":
    _heavytail_kernel(2, 1.0, np.array([2.0, 2.0]), 2.5, 100, 10, 0)   # warm up
    print("E5 heavy-tail (Pareto a=2.5) vs exponential product formula:")
    for mus in ([2.0, 2.0, 2.0], [3.0, 1.0, 4.0]):
        m, ci = heavytail_pK(mus, 1.0)
        exp_pred = theory.p_K_product(mus, 1.0)
        print(f"  mus={mus}: Pareto p_K={m:.4f}+-{ci:.4f}  exp product={exp_pred:.4f}  "
              f"deviation={m-exp_pred:+.4f} ({100*(m-exp_pred)/exp_pred:+.1f}%)")
