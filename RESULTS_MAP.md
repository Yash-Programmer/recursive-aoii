# Results → code map

Each theorem / equation in the paper, and where it is reproduced and checked.

| Paper result | Statement | Reproduced in | Checked by |
|---|---|---|---|
| **Lemma 1** (cascade decomposition / parity) | monitor wrong ⇔ odd number of links disagree | `raoii/lemma1.py` | exhaustive enumeration (`tests/test_phase03.py`); simulator parity invariant |
| **Theorem 2** (product formula) | `p_K = ½[1 − ∏ μ_k/(μ_k+2ν)]` | `raoii/theory.py::p_K_product`, `raoii/joint.py::analyze` | SymPy recursion→product + Monte-Carlo CI (`tests/`); E1 sweep |
| **Lemma** (parity-moment factorization) | `E[∏ S_k] = ∏(1−2α_k)` for source-anchored prefixes | `raoii/joint.py` | `verify/verify_strongaccept.py` (V3) |
| **Theorem** (mean R-AoII) | `E[Δ^R_K] = −π_Mᵀ A⁻¹ 1`; elementary at K=1,2,3 | `raoii/joint.py::analyze`, `raoii_K1_symbolic` | SymPy K=1,2,3 + Monte-Carlo (`tests/test_joint.py`) |
| **Lemma** (universal age bound) | `E[Δ^R_K] ≤ p_K/ν < 1/(2ν)` (saturation) | `raoii/joint.py` | `verify/verify_strongaccept.py` (V1, V7) |
| **Corollary** (bottleneck) | slowest link dominates `∂p_K/∂μ_k` | `raoii/theory.py` | SymPy derivative sign (symbolic stage) |
| **Theorem** (threshold structure) | per-relay threshold fixed point `2ν(μ+ν)(θ−βν)=(μ−ν)−μe^{−νθ}` | `raoii/policy.py` | SymPy unique-root + value-iteration match (`tests/`) |
| **Proposition** (decoupling near-optimality) | gap exactly computable, vanishes as μ/ν→∞ | `raoii/jointvi.py` | E3 joint-vs-decoupled (`experiments/run_phase05.py`) |
| **Lemma** (causal conditional independence) | `M_i ⟂ M_j | upstream path` ⇒ assumption-free envelope | `raoii/joint.py` | `verify/verify_strongaccept.py` (V5) |
| **Theorem** (sum-decomposition / NQD) | `Cov(M_1,M_2) = −μ₁μ₂ν²/[(μ₁+μ₂)(μ₁+2ν)²(μ₂+2ν)] < 0` | `raoii/joint.py::cov_M1M2_symbolic`, `nqd_worst_ratio` | SymPy exact + NQD≤1 enumeration (`tests/test_joint.py`) |
| **Proposition** (symmetric N-state) | `p_K = (1−1/N)[1 − ∏ μ_k/(μ_k+Nν/(N−1))]` | `raoii/nstate.py` | SymPy spectral + Monte-Carlo (`tests/test_phase04.py`) |
| **Extension** (erasure channels) | `μ_k ↦ (1−q_k)μ_k` | `raoii/sim.py`, `raoii/theory.py` | `tests/test_phase04.py` |
| **Extension** (tree topology) | leaf error = product along root path | `raoii/heavytail.py` (E5) | `experiments/run_phase05.py` |
| **Figures F1–F6** | all paper figures | `raoii/figures.py` | `python code/reproduce.py` → `figures/plots/` |
