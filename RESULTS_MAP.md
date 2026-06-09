# Results → code map

Each theorem / equation in the paper, and where it is reproduced and checked.

| Paper result | Statement | Reproduced in | Checked by |
|---|---|---|---|
| **Lemma 1** (cascade decomposition / parity) | monitor wrong ⇔ odd number of links disagree | `raoii/lemma1.py` | exhaustive enumeration (`tests/test_phase03.py`); simulator parity invariant |
| **Theorem 2** (product formula) | `p_K = ½[1 − ∏ μ_k/(μ_k+2ν)]` | `raoii/theory.py::p_K_product`, `raoii/joint.py::analyze` | SymPy recursion→product + Monte-Carlo CI (`tests/`); E1 sweep |
| **Lemma** (parity-moment factorization) | `E[∏ S_k] = ∏(1−2α_k)` for source-anchored prefixes | `raoii/joint.py` | `verify/verify_strongaccept.py` (V3) |
| **Theorem** (mean R-AoII) | `E[Δ^R_K] = −π_Mᵀ A⁻¹ 1`; elementary at K=1,2,3 | `raoii/joint.py::analyze`, `raoii_K1_symbolic` | SymPy K=1,2,3 + Monte-Carlo (`tests/test_joint.py`) |
| **Lemma** (universal age bound) | `E[Δ^R_K] ≤ p_K/ν` (every policy; `< 1/(2ν)` under always-push, tight at never-transmit) | `raoii/joint.py` | `verify/verify_strongaccept.py` (V1, V7) |
| **Corollary** (bottleneck) | slowest link dominates `∂p_K/∂μ_k` | `raoii/theory.py` | SymPy derivative sign (symbolic stage) |
| **Theorem** (threshold structure) | per-relay threshold fixed point `2ν(μ+ν)(θ−βν)=(μ−ν)−μe^{−νθ}` | `raoii/policy.py` | SymPy unique-root + value-iteration match (`tests/`) |
| **Proposition** (decoupling gap) | gap exactly computable over the per-link threshold class; vanishes only as μ/ν→∞; at finite rates the surrogate calibration sits 30–53% above the threshold-class optimum while the achieved (measured-rate) calibration is within 1.8%, so the gap is governed by calibration; bounded throughout by `G_K ≤ 2R`; at K=6 no reallocation at fixed total budget beats the achieved-calibrated policy by more than a few percent | `raoii/benchmark.py` (exact K=2), `raoii/sim.py` (K=6) | **E6 exact joint benchmark, Table II** + E3 K=6 reallocation (`experiments/run_phase05.py`); `tests/test_benchmark.py` |
| **Result** (price of decentralization) | threshold rule = optimal *decentralized* structure; the unrestricted (genie) optimum on the global state (M, monitor wrong-run age) lies 21–35% below it at K=2, growing with depth (≈31→42→50% at K=2→4); the entire gap is error-cancellation withholding (forward 0% in the accidentally-correct state `M=(1,1)`), which needs the source and so is unobservable to relays; the genie holds R-AoII depth-independent under the budget | `raoii/genie.py` (exact, general K) | **E7** (`experiments/run_phase05.py`); `tests/test_genie.py` (always-push exact, saturation, genie<threshold, cancellation withholding) |
| **Lemma** (causal conditional independence) | `M_i ⟂ M_j | upstream path` ⇒ assumption-free envelope | `raoii/joint.py` | `verify/verify_strongaccept.py` (V5) |
| **Theorem** (sum-decomposition / NQD) | `Cov(M_1,M_2) = −μ₁μ₂ν²/[(μ₁+μ₂)(μ₁+2ν)²(μ₂+2ν)] < 0` | `raoii/joint.py::cov_M1M2_symbolic`, `nqd_worst_ratio` | SymPy exact + NQD≤1 enumeration (`tests/test_joint.py`) |
| **Proposition** (symmetric N-state) | `p_K = (1−1/N)[1 − ∏ μ_k/(μ_k+Nν/(N−1))]` | `raoii/nstate.py` | SymPy spectral + Monte-Carlo (`tests/test_phase04.py`) |
| **Extension** (erasure channels) | `μ_k ↦ (1−q_k)μ_k` | `raoii/sim.py`, `raoii/theory.py` | `tests/test_phase04.py` |
| **Extension** (tree topology) | leaf error = product along root path | `raoii/heavytail.py` (E5) | `experiments/run_phase05.py` |
| **Figures F1–F6** | all paper figures | `raoii/figures.py` | `python code/reproduce.py` → `figures/plots/` |
