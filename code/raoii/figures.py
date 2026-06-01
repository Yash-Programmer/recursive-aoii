"""Generate the paper figures from theory closed forms + experiments/*.json.

Vector PDF, viridis/cividis, NeurIPS column width (also fine for IEEEtran two-column).
Run after experiments/run_phase05.py. Figures -> figures/plots/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import theory

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "experiments"))
PLOTS = os.path.abspath(os.path.join(HERE, "..", "..", "figures", "plots"))
os.makedirs(PLOTS, exist_ok=True)
HALF = (3.4, 2.6)                                  # column width (inches)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _load(name):
    p = os.path.join(EXP, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_cascade_curve():
    """F3: p_K vs K with the ->1/2 asymptote and sim CIs (E1)."""
    e1 = _load("E1_data.json")
    fig, ax = plt.subplots(figsize=HALF)
    Ks = np.arange(1, 13)
    for name, color in [("homogeneous", None), ("heterogeneous", None), ("bottleneck", None)]:
        if e1 and name in e1["settings"]:
            rows = e1["settings"][name]
            closed = [r["closed"] for r in rows]
            sim = [r["sim"] for r in rows]; ci = [r["ci"] for r in rows]
            ax.plot(Ks, closed, "-", label=f"{name} (closed)")
            ax.errorbar(Ks, sim, yerr=ci, fmt="o", ms=3, capsize=2, alpha=0.7)
    ax.axhline(0.5, ls="--", color="k", lw=0.8, label=r"$\frac{1}{2}$ (cascade limit)")
    ax.set_xlabel("hops $K$"); ax.set_ylabel(r"$p_K$ (end-to-end mismatch)")
    ax.legend(fontsize=6); ax.set_title("End-to-end error probability: closed form vs simulation")
    fig.savefig(os.path.join(PLOTS, "F3_cascade_curve.pdf")); plt.close(fig)


def fig_bottleneck_heatmap():
    """F5: p_K over (mu, nu) showing the smallest-mu bottleneck (closed form)."""
    mus = np.linspace(0.3, 6, 80); nus = np.linspace(0.1, 3, 80)
    Z = np.array([[theory.p_K_product([m, 3.0, 3.0], n) for m in mus] for n in nus])
    fig, ax = plt.subplots(figsize=HALF)
    im = ax.pcolormesh(mus, nus, Z, cmap="cividis", shading="auto")
    fig.colorbar(im, ax=ax, label=r"$p_3$")
    ax.set_xlabel(r"bottleneck $\mu_1$"); ax.set_ylabel(r"flip rate $\nu$")
    ax.set_title("Bottleneck: smallest $\\mu_k$ dominates $p_K$")
    fig.savefig(os.path.join(PLOTS, "F5_bottleneck.pdf")); plt.close(fig)


def fig_policy_comparison():
    """F6: end-to-end R-AoII by policy at matched rate (E4)."""
    e4 = _load("E4_data.json")
    if not e4:
        return
    labels = {"nested_ours": "nested (ours)", "per_hop_greedy": "per-hop greedy",
              "aoi_blind": "AoI (blind)", "random": "random"}
    keys = list(labels); vals = [e4[k]["raoii"] for k in keys]; cis = [e4[k]["ci"] for k in keys]
    fig, ax = plt.subplots(figsize=HALF)
    ax.bar(range(len(keys)), vals, yerr=cis, capsize=3,
           color=plt.cm.viridis(np.linspace(0.15, 0.85, len(keys))))
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([labels[k] for k in keys], rotation=20, fontsize=6)
    ax.set_ylabel(r"avg R-AoII $\overline{\Delta^R_K}$"); ax.set_title("Policy comparison ($K=6$, common budget)")
    fig.savefig(os.path.join(PLOTS, "F6_policy_comparison.pdf")); plt.close(fig)


def fig_time_integral():
    """T2*: exact E[Delta^R_K] (joint-CTMC) vs sim, K=1..6 (closes OPEN 2.2)."""
    from . import joint, sim
    Ks = range(1, 7); ex = []; sm = []; ci = []
    for K in Ks:
        mus = [2.0] * K
        ex.append(joint.analyze(mus, 1.0)["raoii"])
        m, c, _ = sim.mc_estimate(mus, 1.0, n_events=400_000, n_seeds=10, key="raoii")
        sm.append(m); ci.append(c)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(list(Ks), ex, "-o", ms=3, label=r"exact $-\pi_M^\top A^{-1}\mathbf{1}$")
    ax.errorbar(list(Ks), sm, yerr=ci, fmt="s", ms=3, capsize=2, alpha=0.7, label="simulation")
    ax.set_xlabel("hops $K$"); ax.set_ylabel(r"$\overline{\Delta^R_K}$ (R-AoII)")
    ax.set_title(r"Mean Recursive AoII vs cascade length ($\mu{=}2,\nu{=}1$)")
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(PLOTS, "F3b_time_integral.pdf")); plt.close(fig)


def fig_nesting():
    """F4: threshold nesting vs reversal (ordered vs equal budgets)."""
    from . import policy
    mus = [3.0] * 4; nu = 1.0
    ps = theory.p_recursion(mus, nu)
    nue = [nu * (1 - 2 * (0 if k == 0 else ps[k - 1])) for k in range(4)]
    th_ord = [policy.threshold_for_budget(l, mus[k], nue[k]) for k, l in enumerate([.4, .3, .2, .1])]
    th_eq = [policy.threshold_for_budget(0.25, mus[k], nue[k]) for k in range(4)]
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(range(1, 5), th_ord, "-o", ms=4, label=r"ordered $\lambda$ (nests $\uparrow$)")
    ax.plot(range(1, 5), th_eq, "-s", ms=4, label=r"equal $\lambda$ (reverses $\downarrow$)")
    ax.set_xlabel("hop index $k$"); ax.set_ylabel(r"threshold $\theta_k$"); ax.set_xticks(range(1, 5))
    ax.set_title("Nesting: budget vs $\\nu_{eff}$ gradient"); ax.legend(fontsize=6)
    fig.savefig(os.path.join(PLOTS, "F4_nesting.pdf")); plt.close(fig)


def fig_tandem_schematic():
    """F1: the K-hop tandem (source -> relays -> monitor)."""
    fig, ax = plt.subplots(figsize=(6.5, 1.5)); ax.axis("off")
    K = 4; xs = np.arange(K + 1)
    labels = [r"$X{=}\hat X_0$"] + [r"$\hat X_%d$" % k for k in range(1, K)] + [r"$\hat X_K$"]
    roles = ["source"] + ["relay"] * (K - 1) + ["monitor"]
    cols = plt.cm.viridis(np.linspace(0.2, 0.85, K + 1))
    for i, (x, lab, role, c) in enumerate(zip(xs, labels, roles, cols)):
        ax.scatter([x], [0], s=900, color=c, zorder=3, edgecolor="k")
        ax.text(x, 0, lab, ha="center", va="center", fontsize=7, color="w", zorder=4)
        ax.text(x, -0.42, role, ha="center", va="center", fontsize=6, style="italic")
        if i < K:
            ax.annotate("", xy=(x + 0.78, 0), xytext=(x + 0.22, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.2))
            ax.text(x + 0.5, 0.22, r"$\mu_%d,q_%d$" % (i + 1, i + 1), ha="center", fontsize=6)
    ax.set_xlim(-0.6, K + 0.6); ax.set_ylim(-0.7, 0.5)
    ax.set_title("F1: $K$-hop tandem cascade", fontsize=8)
    fig.savefig(os.path.join(PLOTS, "F1_tandem.pdf")); plt.close(fig)


def fig_samplepath():
    """F2: the visual DEFINITION of R-AoII (source, monitor estimate, mismatch runs, sawtooth age)."""
    t = np.linspace(0, 10, 1000)
    # toy step processes
    Xt = np.where((t % 4) < 2, 0, 1).astype(float)
    Xh = np.zeros_like(t)
    for i, ti in enumerate(t):
        Xh[i] = 1 if (2.3 < ti < 5.5) or (7.0 < ti < 9.2) else 0
    mism = (Xt != Xh).astype(float)
    # age sawtooth: ramps during mismatch, resets at agreement
    U = 0.0; age = np.zeros_like(t)
    for i in range(len(t)):
        if mism[i]:
            age[i] = t[i] - U
        else:
            age[i] = 0.0; U = t[i]
    fig, axs = plt.subplots(3, 1, figsize=(3.6, 3.4), sharex=True)
    axs[0].step(t, Xt, where="post", lw=1.2, label=r"$X(t)$")
    axs[0].step(t, Xh + 0.04, where="post", lw=1.2, ls="--", label=r"$\hat X_K(t)$")
    axs[0].set_yticks([0, 1]); axs[0].legend(fontsize=6, loc="upper right"); axs[0].set_ylabel("state")
    axs[1].fill_between(t, 0, mism, step="post", alpha=0.4, color="crimson")
    axs[1].set_ylabel(r"$M_K^{e2e}$"); axs[1].set_yticks([0, 1])
    axs[2].plot(t, age, color="crimson", lw=1.3)
    axs[2].set_ylabel(r"$\Delta^R_K(t)$"); axs[2].set_xlabel("time $t$")
    axs[0].set_title("F2: R-AoII = age of the current mismatch run", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "F2_samplepath.pdf")); plt.close(fig)


def make_all():
    fig_cascade_curve(); fig_bottleneck_heatmap(); fig_policy_comparison()
    fig_time_integral(); fig_nesting(); fig_tandem_schematic(); fig_samplepath()
    print("figures ->", PLOTS)


if __name__ == "__main__":
    make_all()
