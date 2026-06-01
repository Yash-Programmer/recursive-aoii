"""Event-driven CTMC tandem simulator with exact R-AoII age accounting.

Two engines with identical semantics:
  * `_kernel_njit` — numba-compiled hot loop (default; ~1000x faster);
  * `tandem_sim_python` — the reference implementation, kept for cross-checking.
Both consume the SAME pre-drawn randomness (dts, evs, erase_u), so for identical
inputs they produce bit-identical outputs (see code/tests/test_sim_engines.py).

Source X = Xhat[0] (perfect sensor, A3). Links k=1..K: Xhat[k] copies Xhat[k-1] at
link-k opportunities (Poisson mu_k) when the policy transmits and the channel does
not erase (decide-then-erase, A11). Symmetric two-state source flips at rate nu (A6).

Accounting invariants (the #1 bug source):
  * monitor R-AoII uses last_agree_mon for {X != Xhat[K]} (source-vs-monitor);
  * per-hop policy clock tau_k uses last_hop_agree[k] for {Xhat[k-1]!=Xhat[k]},
    updated by a pre-event and post-event pass so a link broken at t has tau=0.
"""
import numpy as np
from numba import njit


@njit(cache=False)
def _kernel_njit(K, nu, mus, qs, thetas, policy_code, n_events, burnin,
                 dts, evs, erase_u, trans_u, track_pairs):
    # policy_code: 0=always, 1=mismatch-threshold (ours/greedy), 2=random-at-rate,
    #              3=AoI-threshold (semantic-blind: transmit on refresh-age >= theta)
    xhat = np.zeros(K + 1, np.int8)
    last_agree_mon = 0.0
    last_hop_agree = np.zeros(K + 1)
    last_refresh = np.zeros(K + 1)               # for AoI-threshold (refresh age)
    t = 0.0
    total_time = 0.0
    mismatch_time = 0.0
    raoii_int = 0.0
    hop_mismatch_time = np.zeros(K + 1)
    slack_time = 0.0
    xor_violations = 0
    pair_time = np.zeros((K + 1, K + 1))
    tx_count = np.zeros(K + 1)                    # transmit attempts per link (for rate)

    for it in range(n_events):
        dt = dts[it]
        t_next = t + dt
        counting = it >= burnin
        m_mon = 1 if xhat[0] != xhat[K] else 0

        if counting:
            total_time += dt
            if m_mon:
                a0 = t - last_agree_mon
                a1 = t_next - last_agree_mon
                raoii_int += 0.5 * (a1 * a1 - a0 * a0)
                mismatch_time += dt
            sumMk = 0
            xorv = 0
            for k in range(1, K + 1):
                mk = 1 if xhat[k - 1] != xhat[k] else 0
                hop_mismatch_time[k] += mk * dt
                sumMk += mk
                xorv ^= mk
            slack_time += (sumMk - m_mon) * dt
            if xorv != m_mon:
                xor_violations += 1
            if track_pairs:
                for i in range(1, K + 1):
                    if xhat[i - 1] != xhat[i]:
                        for j in range(i + 1, K + 1):
                            if xhat[j - 1] != xhat[j]:
                                pair_time[i, j] += dt
        if m_mon == 0:
            last_agree_mon = t_next

        t = t_next
        for k in range(1, K + 1):                  # pre-event pass
            if xhat[k - 1] == xhat[k]:
                last_hop_agree[k] = t
        ev = evs[it]
        if ev == 0:
            xhat[0] ^= 1
        else:
            k = ev
            mk = xhat[k - 1] != xhat[k]
            if policy_code == 0:                   # always-push
                transmit = True
            elif policy_code == 1:                 # mismatch-threshold (per-hop disagreement age)
                transmit = mk and ((t - last_hop_agree[k]) >= thetas[k - 1])
            elif policy_code == 2:                 # random at rate (thetas[k-1] = prob)
                transmit = trans_u[it] < thetas[k - 1]
            else:                                  # 3: AoI-threshold (refresh age, semantic-blind)
                transmit = (t - last_refresh[k]) >= thetas[k - 1]
            if transmit:
                if it >= burnin:
                    tx_count[k] += 1.0
                if erase_u[it] >= qs[k - 1]:
                    xhat[k] = xhat[k - 1]
                    last_refresh[k] = t
        for k in range(1, K + 1):                  # post-event pass
            if xhat[k - 1] == xhat[k]:
                last_hop_agree[k] = t
        if xhat[0] == xhat[K]:
            last_agree_mon = t

    return (total_time, mismatch_time, raoii_int, hop_mismatch_time,
            slack_time, xor_violations, pair_time, tx_count)


def _draw(mus, nu, n_events, seed):
    rng = np.random.default_rng(seed)
    rates = np.concatenate(([nu], mus))
    Lam = rates.sum()
    cum = np.cumsum(rates) / Lam
    dts = rng.exponential(1.0 / Lam, n_events)
    evs = np.searchsorted(cum, rng.random(n_events)).astype(np.int64)
    erase_u = rng.random(n_events)
    trans_u = rng.random(n_events)             # for random-at-rate policy
    return dts, evs, erase_u, trans_u


_POLICY_CODE = {"always": 0, "threshold": 1, "random": 2, "aoi": 3}


def _assemble(K, res):
    total_time, mismatch_time, raoii_int, hop_mm, slack_time, xorv, pair_time, tx = res
    return {
        "p_K": mismatch_time / total_time,
        "raoii": raoii_int / total_time,
        "hop_mismatch": hop_mm[1:] / total_time,
        "slack_mean": slack_time / total_time,
        "xor_violations": int(xorv),
        "total_time": total_time,
        "pair_mismatch": pair_time / total_time,
        "tx_rate": tx[1:] / total_time,             # achieved attempt rate per link
    }


def tandem_sim(mus, nu, qs=None, policy="always", thetas=None,
               n_events=300_000, seed=0, burnin_frac=0.1, track_pairs=False):
    mus = np.atleast_1d(np.asarray(mus, float))
    K = len(mus)
    qs = np.zeros(K) if qs is None else np.atleast_1d(np.asarray(qs, float))
    thetas = (np.zeros(K) if thetas is None else np.atleast_1d(np.asarray(thetas, float)))
    code = _POLICY_CODE[policy]
    dts, evs, erase_u, trans_u = _draw(mus, nu, n_events, seed)
    res = _kernel_njit(K, float(nu), mus, qs, thetas, code,
                       n_events, int(burnin_frac * n_events), dts, evs, erase_u,
                       trans_u, track_pairs)
    return _assemble(K, res)


def tandem_sim_python(mus, nu, qs=None, policy="always", thetas=None,
                      n_events=300_000, seed=0, burnin_frac=0.1, track_pairs=False):
    """Reference (un-JIT'd) implementation — semantics identical to tandem_sim."""
    mus = np.atleast_1d(np.asarray(mus, float))
    K = len(mus)
    qs = np.zeros(K) if qs is None else np.atleast_1d(np.asarray(qs, float))
    thetas = (np.zeros(K) if thetas is None else np.atleast_1d(np.asarray(thetas, float)))
    use_threshold = (policy != "always")
    dts, evs, erase_u, trans_u = _draw(mus, nu, n_events, seed)
    xhat = np.zeros(K + 1, dtype=np.int8)
    t = 0.0
    last_agree_mon = 0.0
    last_hop_agree = np.zeros(K + 1)
    burnin = burnin_frac * n_events
    total_time = mismatch_time = raoii_int = slack_time = 0.0
    hop_mm = np.zeros(K + 1)
    pair_time = np.zeros((K + 1, K + 1))
    tx_count = np.zeros(K + 1)
    xorv = 0

    def hop_pass(tnow):
        for k in range(1, K + 1):
            if xhat[k - 1] == xhat[k]:
                last_hop_agree[k] = tnow

    for it in range(n_events):
        dt = dts[it]; t_next = t + dt; counting = it >= burnin
        m_mon = int(xhat[0] != xhat[K])
        if counting:
            total_time += dt
            if m_mon:
                a0 = t - last_agree_mon; a1 = t_next - last_agree_mon
                raoii_int += 0.5 * (a1 * a1 - a0 * a0); mismatch_time += dt
            s = 0; xv = 0
            for k in range(1, K + 1):
                mk = int(xhat[k - 1] != xhat[k]); hop_mm[k] += mk * dt; s += mk; xv ^= mk
            slack_time += (s - m_mon) * dt
            if xv != m_mon:
                xorv += 1
            if track_pairs:
                for i in range(1, K + 1):
                    if xhat[i - 1] != xhat[i]:
                        for j in range(i + 1, K + 1):
                            if xhat[j - 1] != xhat[j]:
                                pair_time[i, j] += dt
        if m_mon == 0:
            last_agree_mon = t_next
        t = t_next
        hop_pass(t)
        ev = evs[it]
        if ev == 0:
            xhat[0] ^= 1
        else:
            k = ev; mk = (xhat[k - 1] != xhat[k])
            transmit = True if not use_threshold else bool(mk and (t - last_hop_agree[k]) >= thetas[k - 1])
            if transmit:
                if counting:
                    tx_count[k] += 1.0
                if erase_u[it] >= qs[k - 1]:
                    xhat[k] = xhat[k - 1]
        hop_pass(t)
        if xhat[0] == xhat[K]:
            last_agree_mon = t
    return _assemble(K, (total_time, mismatch_time, raoii_int, hop_mm, slack_time, xorv, pair_time, tx_count))


def mc_estimate(mus, nu, qs=None, policy="always", thetas=None,
                n_events=300_000, n_seeds=12, key="p_K"):
    vals = np.array([tandem_sim(mus, nu, qs, policy, thetas, n_events, seed=s)[key]
                     for s in range(n_seeds)])
    return vals.mean(), 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)), vals


if __name__ == "__main__":
    from . import theory
    import time
    tandem_sim([2.0, 2.0], 1.0, n_events=100)        # warm up JIT
    t = time.time()
    out = tandem_sim([2.0] * 6, 1.0, n_events=2_000_000, seed=1)
    print(f"JIT: 2e6 events K=6 in {time.time()-t:.2f}s  "
          f"p_K={out['p_K']:.4f} closed={theory.p_K_product([2.0]*6,1.0):.4f} "
          f"xor_viol={out['xor_violations']}")
