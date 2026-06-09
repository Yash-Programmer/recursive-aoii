"""One-command reproduction of every number, test, and figure in the paper.

    python code/reproduce.py            # symbolic checks + tests + experiments + figures
    python code/reproduce.py --fast     # skip the 10^6-event sweep (quick CI check)

All randomness is seeded; results are deterministic across runs.
"""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=HERE)


def main():
    fast = "--fast" in sys.argv
    t0 = time.time()
    print("=" * 60, "\n REPRODUCING R-AoII RESULTS\n", "=" * 60)

    # 1. Symbolic verification (Theorem 2, 2*, threshold, bottleneck)
    from raoii import theory
    r = theory.verify_symbolic(verbose=True)
    assert all(r.values()), "SYMBOLIC CHECK FAILED"

    # 2. Deterministic + statistical tests (engine match, Lemma 1, thresholds, N-state, erasure)
    rc = run([sys.executable, "-m", "pytest", "tests/", "-q"])
    assert rc == 0, "TESTS FAILED"

    # 3. Experiments E1-E6 (full precision unless --fast)
    if not fast:
        rc = run([sys.executable, "experiments/run_phase05.py"])
        assert rc == 0, "EXPERIMENTS FAILED"

    # 4. Figures
    from raoii import figures
    figures.make_all()

    print(f"\nDONE in {time.time()-t0:.0f}s. All symbolic checks, tests, experiments, figures reproduced.")


if __name__ == "__main__":
    main()
