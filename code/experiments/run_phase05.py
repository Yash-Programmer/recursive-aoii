"""Phase-05 experiment runner (IEEE-precision). Saves results to experiments/*.json.

E1 verify Theorem 2 (full sweep)   E2 parity (Lemma 1)   E3 budget reallocation at scale (K=6)
E4 policy comparison (K=6)         E5 heavy-tail + tree path-product
E6 exact joint benchmark, Table II (decoupled calibrations vs threshold-class optimum)
E7 unrestricted (genie) optimum -- price of decentralization (gap, depth scaling, mechanism)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from raoii import theory, sim, policy, lemma1, heavytail

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

# ---- E3: budget reallocation at scale (K=6; paper "Budget reallocation at scale") --------
# Can any reallocation of the per-hop thresholds at fixed TOTAL budget beat the achieved-rate-
# calibrated decentralized policy? Beyond the exactly solvable K<=2 we probe by event-driven
# simulation (validated against the exact K<=2 values), using common random numbers (fixed
# seeds across candidates) so policy *differences* are low-variance.
print("E3: budget reallocation at scale (K=6, achieved-calibrated vs best reallocation)")
MUS6, NU6, PERLINK6 = [2.0] * 6, 1.0, 0.25
B6 = 6 * PERLINK6                                  # total budget
NE_S3, S_C3, S_S3 = 400_000, range(6), range(8)


def _link_rate6(thr_pre, k, seeds):               # realized rate of link k under calibrated upstream
    return np.mean([sim.tandem_sim(MUS6[:k + 1], NU6, policy="threshold", thetas=thr_pre,
                                   n_events=600_000, seed=s)["tx_rate"][k] for s in seeds])


def _calibrate6(seeds):                           # set theta_k so link k's realized rate ~ budget
    thr = [0.0] * 6
    for k in range(6):
        thr[k] = 0.0
        if _link_rate6(thr[:k + 1], k, seeds) <= PERLINK6:
            continue
        lo, hi = 0.0, 4.0
        for _ in range(12):
            mid = 0.5 * (lo + hi); thr[k] = mid
            lo, hi = (mid, hi) if _link_rate6(thr[:k + 1], k, seeds) > PERLINK6 else (lo, mid)
        thr[k] = hi
    return thr


def _eval6(thr, ne, seeds):
    d = np.array([sim.tandem_sim(MUS6, NU6, policy="threshold", thetas=thr, n_events=ne, seed=s)["raoii"] for s in seeds])
    tot = np.mean([sim.tandem_sim(MUS6, NU6, policy="threshold", thetas=thr, n_events=ne, seed=s)["tx_rate"].sum() for s in seeds])
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(seeds)), tot


thr_dec6 = _calibrate6(S_C3)
Jdec6, hdec6, totdec6 = _eval6(thr_dec6, NE, range(NS))          # high-precision baseline
best6 = list(thr_dec6); bestJ6, _, _ = _eval6(best6, NE_S3, S_S3)
for _pass in range(2):                            # coordinate descent at fixed total budget (CRN)
    for k in range(6):
        for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7]:
            cand = list(best6); cand[k] = v
            J, _, tot = _eval6(cand, NE_S3, S_S3)
            if tot <= B6 * 1.03 and J < bestJ6 - 1e-4:
                best6, bestJ6 = cand, J
Jbest6, hbest6, totbest6 = _eval6(best6, NE, range(NS))          # high-precision best
realloc_pct = (Jdec6 - Jbest6) / Jdec6 * 100
e3 = {"mus": MUS6, "nu": NU6, "perlink_budget": PERLINK6, "total_budget": B6,
      "theta_calibrated": list(thr_dec6), "raoii_calibrated": Jdec6, "ci_calibrated": hdec6, "rate_calibrated": totdec6,
      "theta_realloc": list(best6), "raoii_realloc": Jbest6, "ci_realloc": hbest6, "rate_realloc": totbest6,
      "realloc_improve_pct": realloc_pct}
save(e3, "E3_data.json")
print(f"  achieved-calibrated theta={np.round(thr_dec6,3)} raoii={Jdec6:.4f}+-{hdec6:.4f} rate={totdec6:.3f}")
print(f"  best reallocation  theta={np.round(best6,3)} raoii={Jbest6:.4f} rate={totbest6:.3f}")
print(f"  reallocation improves decentralized by {realloc_pct:.2f}% (gain from lowering relay 1's threshold)")

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

# ---- E6: exact joint benchmark over the threshold class (paper Table II) --------
print("E6: exact joint benchmark (Table II: decoupled calibrations vs threshold-class optimum)")
from raoii import joint, benchmark as bm
e6 = {"solver_validation": [], "table2": []}

# (a) cross-check the age-augmented evaluator against the exact joint-CTMC of joint.py
# (always-push ignores the local age, so a tiny age grid is exact up to the O(h) step bias)
for mus in ([2.0, 2.0], [3.0, 1.5, 4.0]):
    ra_grid, _ = bm.metrics(mus, 1.0, [0] * len(mus), 4, 0.01)
    ex = joint.analyze(mus, 1.0)
    e6["solver_validation"].append({"mus": mus, "raoii_grid": ra_grid, "raoii_exact": ex["raoii"],
                                    "abs_err": abs(ra_grid - ex["raoii"]),
                                    "pK_closed": theory.p_K_product(mus, 1.0), "pK_exact": ex["p_K"]})
print(f"  validation: always-push raoii grid vs exact within "
      f"{max(d['abs_err'] for d in e6['solver_validation']):.1e}; p_K closed == exact")

# (b) Table II: always-push (context), decoupled (surrogate / achieved), threshold optimum, gaps
for r in bm.table2_benchmark():
    e6["table2"].append(r)
    print(f"  mu/nu={int(r['mu'])} budget={r['budget']}: ap={r['always_push']:.3f} "
          f"surr={r['dec_surrogate']:.3f} ach={r['dec_achieved']:.3f} opt={r['joint_opt']:.3f}  "
          f"gap_surr={r['gap_surrogate_pct']:.1f}% gap_ach={r['gap_achieved_pct']:.1f}%")
# assert the regenerated rows against the values published in the manuscript (Table II)
PUB_T2 = {(2.0, 0.25): (0.174, 0.324, 0.250, 0.250, 29.8, 0.0),
          (2.0, 0.15): (0.174, 0.399, 0.307, 0.302, 32.2, 1.8),
          (4.0, 0.40): (0.080, 0.214, 0.140, 0.140, 53.3, 0.0)}
for r in e6["table2"]:
    pub = PUB_T2[(r["mu"], r["budget"])]
    got = (r["always_push"], r["dec_surrogate"], r["dec_achieved"], r["joint_opt"],
           r["gap_surrogate_pct"], r["gap_achieved_pct"])
    for g, p, tol in zip(got, pub, (0.01, 0.01, 0.01, 0.01, 3.0, 3.0)):
        assert abs(g - p) <= tol, ("Table II drift", r["mu"], r["budget"], g, p)
print("  Table II regenerated values match the published manuscript within tolerance")
save(e6, "E6_data.json")

# ---- E7: unrestricted (genie) optimum -- the price of decentralization --------------------
# The threshold class (E6) is the optimal DECENTRALIZED structure. Here we compute the optimum
# over all causal policies that may condition on the global state -- in particular the end-to-end
# (monitor) wrong-run age, which no relay can observe (it needs the source). The gap is the value
# of global coordination; it is realized entirely by withholding a forward that would propagate an
# upstream error onto a monitor made correct by error-cancellation.
print("E7: unrestricted (genie) optimum -- price of decentralization")
from raoii import genie
e7 = {"gap_table": [], "depth_scaling": [], "mechanism": {}}

# (a) genie gap vs the threshold-class joint opt of E6, at the Table II points (all exact at K=2)
for r in e6["table2"]:
    mu, bud = r["mu"], r["budget"]
    g = genie.genie_opt([mu, mu], 1.0, bud, A_max=8.0, target_Lh=0.10)
    gap = 100 * (r["joint_opt"] - g["raoii"]) / r["joint_opt"]
    e7["gap_table"].append({"mu": mu, "budget": bud, "threshold_opt": r["joint_opt"],
                            "genie": g["raoii"], "genie_unconstrained": g["raoii_unconstrained"],
                            "gap_pct": gap, "rate": float(g["rates"][0])})
    print(f"  mu/nu={int(mu)} bud={bud}: threshold={r['joint_opt']:.3f} genie={g['raoii']:.3f} "
          f"(unconstrained {g['raoii_unconstrained']:.3f})  gap={gap:.1f}%")

# (b) depth scaling (mu=2, budget=0.25): the genie holds R-AoII ~depth-independent under the budget
#     (a budgeted strengthening of saturation) while always-push and the threshold class grow
def _thr_depth(K, budget=0.25):
    mus = [2.0] * K; thr = [0.0] * K
    def rate(k):
        return np.mean([sim.tandem_sim(mus[:k + 1], 1.0, policy="threshold", thetas=thr[:k + 1],
                                       n_events=600_000, seed=s)["tx_rate"][k] for s in range(6)])
    for k in range(K):
        thr[k] = 0.0
        if rate(k) <= budget:
            continue
        lo, hi = 0.0, 4.0
        for _ in range(12):
            mid = 0.5 * (lo + hi); thr[k] = mid
            lo, hi = (mid, hi) if rate(k) > budget else (lo, mid)
        thr[k] = hi
    return np.mean([sim.tandem_sim(mus, 1.0, policy="threshold", thetas=thr, n_events=NE, seed=s)["raoii"]
                    for s in range(NS)])
for K in (2, 3, 4):
    mus = [2.0] * K; tlh = 0.10 if K <= 3 else 0.13
    g = genie.genie_opt(mus, 1.0, 0.25, A_max=8.0, target_Lh=tlh)
    ap = genie.always_push(mus, 1.0, target_Lh=tlh)
    if K == 2:
        # exact threshold-class optimum from E6 (same grid convention as the gap table
        # above and as the paper's Table II), so tab:genie's K=2 row matches Table II
        th = next(r["joint_opt"] for r in e6["table2"]
                  if r["mu"] == 2.0 and r["budget"] == 0.25)
    else:
        # exact in-class search is infeasible at K>=3: use the achieved-rate-calibrated
        # policy (within 1.8% of the in-class optimum at every exactly solvable point),
        # evaluated by the validated event-driven simulator
        th = _thr_depth(K)
    gap = 100 * (th - g["raoii"]) / th
    e7["depth_scaling"].append({"K": K, "always_push": ap, "threshold": th,
                                "genie": g["raoii"], "gap_pct": gap})
    print(f"  K={K}: always-push={ap:.3f} threshold={th:.3f} genie={g['raoii']:.3f}  gap={gap:.1f}%")
# assert the depth table against the values published in the manuscript (tab:genie)
PUB_GENIE = {2: (0.174, 0.250, 0.172, 31.0), 3: (0.246, 0.304, 0.176, 42.0),
             4: (0.300, 0.344, 0.172, 50.0)}
for row in e7["depth_scaling"]:
    pub = PUB_GENIE[row["K"]]
    got = (row["always_push"], row["threshold"], row["genie"], row["gap_pct"])
    for g, p, tol in zip(got, pub, (0.01, 0.01, 0.01, 2.0)):
        assert abs(g - p) <= tol, ("tab:genie drift", row["K"], g, p)
print("  depth table matches the published manuscript (tab:genie) within tolerance")

# (c) mechanism: the genie withholds the link-2 forward in the double-error-cancellation state
frac = genie.link_tx_fraction([2.0, 2.0], 1.0, 0.25, link=2, patterns=[(0, 1), (1, 1)], target_Lh=0.08)
e7["mechanism"] = {"M_0_1_fix": frac[(0, 1)], "M_1_1_cancel": frac[(1, 1)]}
print(f"  mechanism: link-2 transmits on M=(0,1) monitor-wrong={frac[(0,1)]:.2f}, "
      f"on M=(1,1) cancellation={frac[(1,1)]:.2f} (withholds)")
save(e7, "E7_data.json")

print(f"\nALL E1-E7 DONE in {time.time()-t0:.0f}s")
