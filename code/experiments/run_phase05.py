"""Phase-05 experiment runner (IEEE-precision). Saves results to experiments/*.json.

E1 verify Theorem 2 (full sweep)   E2 parity (Lemma 1)   E3 decoupling (matched-budget)
E4 policy comparison (K=6)         E5 heavy-tail + tree path-product
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from raoii import theory, sim, policy, lemma1, jointvi, heavytail

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments")
OUT = os.path.abspath(OUT); os.makedirs(OUT, exist_ok=True)


def save(obj, name):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else o.tolist())
NE, NS = 1_000_000, 20                       # IEEE precision
np.set_printoptions(precision=4, suppress=True)
sim.tandem_sim([2.0, 2.0], 1.0, n_events=100)   # warm JIT


def nu_eff_list(mus, nu):
    ps = theory.p_recursion(mus, nu)
    return [nu * (1 - 2 * (0.0 if k == 0 else ps[k - 1])) for k in range(len(mus))]


def rate_for_aoi(mus, nu, target, hi=30.0):    # calibrate AoI-threshold to a total rate
    K = len(mus)
    def tot(th):
        o = sim.tandem_sim(mus, nu, policy="aoi", thetas=[th] * K, n_events=200_000, seed=0)
        return o["tx_rate"].sum()
    lo, hi = 0.0, hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if tot(mid) > target: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)


# ---- E1: Theorem 2 product formula, full sweep --------------------------------
print("E1: Theorem 2 (10^6 events x 20 seeds)"); t0 = time.time()
e1 = {"settings": {}}
settings = {
    "homogeneous": ([2.0] * 12, 1.0),
    "heterogeneous": ([1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 0.8, 5.0], 1.0),
    "bottleneck": ([0.5] + [4.0] * 11, 1.0),
}
worst = 0.0; allok = True
for name, (mus_all, nu) in settings.items():
    rows = []
    for K in range(1, 13):
        mus = mus_all[:K]
        closed = theory.p_K_product(mus, nu)
        mean, ci, _ = sim.mc_estimate(mus, nu, n_events=NE, n_seeds=NS)
        ok = abs(mean - closed) <= ci + 1e-12
        allok = allok and ok; worst = max(worst, abs(mean - closed))
        rows.append({"K": K, "closed": closed, "sim": mean, "ci": ci, "ok": ok})
    e1["settings"][name] = rows
e1["all_bracket"] = bool(allok); e1["worst_abs_err"] = worst
print(f"  all CIs bracket: {allok}; worst |sim-closed|={worst:.4f}  ({time.time()-t0:.0f}s)")
save(e1, "E1_data.json")

# ---- E2: parity (Lemma 1) -----------------------------------------------------
e2 = {"xor_exact_N2": all(lemma1.check_xor_identity(K)[0] for K in range(1, 9))}
o = sim.tandem_sim([2.0] * 4, 1.0, n_events=2_000_000, seed=0)
e2["sim_xor_violations"] = o["xor_violations"]; e2["slack_mean"] = o["slack_mean"]
save(e2, "E2_data.json")
print(f"E2: XOR exact={e2['xor_exact_N2']}, sim violations={e2['sim_xor_violations']}")

# ---- E3: decoupling tightness (matched-budget reallocation) -------------------
print("E3: decoupling (matched-budget reallocation)")
mus, nu, beta = [3.0, 3.0], 1.0, 1.0
th_dec, nue = jointvi.decoupled_thresholds(mus, nu, beta)
def raoii_rate(th, ne=500_000, ns=12):
    ra = np.array([sim.tandem_sim(mus, nu, policy="threshold", thetas=th, n_events=ne, seed=s)["raoii"] for s in range(ns)])
    rt = np.array([sim.tandem_sim(mus, nu, policy="threshold", thetas=th, n_events=ne, seed=s)["tx_rate"].sum() for s in range(ns)])
    return ra.mean(), 1.96 * ra.std(ddof=1) / np.sqrt(ns), rt.mean()
ra_dec, ci_dec, R_dec = raoii_rate(th_dec)
import itertools
best = (9, None)
for a, b in itertools.product(np.linspace(0, 4, 17), repeat=2):
    ra, _, rt = raoii_rate([a, b], ne=250_000, ns=4)
    if abs(rt - R_dec) <= 0.08 * R_dec and ra < best[0]:
        best = (ra, (a, b))
e3 = {"theta_dec": list(th_dec), "raoii_dec": ra_dec, "ci": ci_dec, "total_rate": R_dec,
      "best_realloc_raoii": best[0], "realloc_gain": ra_dec - best[0]}
save(e3, "E3_data.json")
print(f"  decoupled raoii={ra_dec:.4f} at rate {R_dec:.3f}; best reallocation gain={ra_dec-best[0]:+.4f} "
      f"(<=0 => decoupled near-optimal at matched budget)")

# ---- E4: policy comparison at matched total rate, K=6 -------------------------
print("E4: policy comparison (K=6, matched rate)")
mus6 = [2.0, 2.5, 1.5, 3.0, 1.0, 2.0]; nu = 1.0
nue6 = nu_eff_list(mus6, nu)
lam = 0.18                                    # per-hop target attempt rate
th_ours = [policy.threshold_for_budget(lam, mus6[k], nue6[k]) for k in range(6)]    # cascade-aware
th_greedy = [policy.threshold_for_budget(lam, mus6[k], nu) for k in range(6)]       # myopic (nominal nu)
aoi_th = rate_for_aoi(mus6, nu, 6 * lam)
probs = [min(1.0, lam / mus6[k]) for k in range(6)]
def comp(pol, th):
    m, ci, _ = sim.mc_estimate(mus6, nu, policy=pol, thetas=th, n_events=NE, n_seeds=NS, key="raoii")
    mp, _, _ = sim.mc_estimate(mus6, nu, policy=pol, thetas=th, n_events=NE, n_seeds=NS, key="p_K")
    rt = sim.tandem_sim(mus6, nu, policy=pol, thetas=th, n_events=NE, seed=0)["tx_rate"].sum()
    return {"raoii": m, "ci": ci, "p_K": mp, "rate": rt}
e4 = {"nested_ours": comp("threshold", th_ours), "per_hop_greedy": comp("threshold", th_greedy),
      "aoi_blind": comp("aoi", [aoi_th] * 6), "random": comp("random", probs)}
save(e4, "E4_data.json")
for k, v in e4.items():
    print(f"  {k:16s}: raoii={v['raoii']:.4f}+-{v['ci']:.4f}  p_K={v['p_K']:.4f}  rate={v['rate']:.3f}")

# ---- E5: heavy-tail + tree path-product ---------------------------------------
print("E5: heavy-tail + tree")
heavytail._heavytail_kernel(2, 1.0, np.array([2.0, 2.0]), 2.5, 100, 10, 0)
e5 = {"heavytail": []}
for mus in ([2.0, 2.0, 2.0], [3.0, 1.0, 4.0]):
    m, ci = heavytail.heavytail_pK(mus, 1.0, n_events=NE, n_seeds=NS)
    e5["heavytail"].append({"mus": mus, "pareto_pK": m, "ci": ci,
                            "exp_product": theory.p_K_product(mus, 1.0)})
# tree: a leaf's marginal == path-product (validated by the tandem along its path)
path = [2.0, 3.0, 1.5]                         # root->leaf path rates
m, ci, _ = sim.mc_estimate(path, 1.0, n_events=NE, n_seeds=NS)
e5["tree_leaf"] = {"path": path, "sim_pK": m, "ci": ci, "path_product": theory.p_K_product(path, 1.0)}
save(e5, "E5_data.json")
for h in e5["heavytail"]:
    print(f"  heavytail mus={h['mus']}: pareto={h['pareto_pK']:.4f} exp={h['exp_product']:.4f}")
print(f"  tree leaf (path {path}): sim={m:.4f} path-product={e5['tree_leaf']['path_product']:.4f}")
print(f"\nALL E1-E5 DONE in {time.time()-t0:.0f}s")
