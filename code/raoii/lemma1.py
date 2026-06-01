"""Exhaustive enumeration checks for Lemma 1 (deductive, exact integer equality)."""
import itertools


def check_xor_identity(K):
    """N=2: 1{Xhat_{-1}!=Xhat_K} == XOR_k 1{Xhat_{k-1}!=Xhat_k}, all 2^{K+1} tuples."""
    for tup in itertools.product((0, 1), repeat=K + 1):     # (Xhat_0..Xhat_K), Xhat_{-1}=Xhat_0=X
        e2e = int(tup[0] != tup[K])
        xor = 0
        for k in range(1, K + 1):
            xor ^= int(tup[k - 1] != tup[k])
        if xor != e2e:
            return False, tup
    return True, None


def check_union_bound(N, K):
    """Any N: 1{X!=Xhat_K} <= sum_k 1{Xhat_{k-1}!=Xhat_k}; return (ok, found_strict)."""
    ok, strict = True, False
    for tup in itertools.product(range(N), repeat=K + 1):    # tup[0]=X=Xhat_0
        e2e = int(tup[0] != tup[K])
        s = sum(int(tup[k - 1] != tup[k]) for k in range(1, K + 1))
        if e2e > s:
            ok = False
        if e2e < s:
            strict = True
    return ok, strict


def cancellation_witness():
    """The canonical N=3, K=2 strict-slack instance from 03a §3.5: X=1, Xhat_1=2, Xhat_2=1."""
    tup = (1, 2, 1)                                          # (X=Xhat_0, Xhat_1, Xhat_2)
    e2e = int(tup[0] != tup[2])
    M1 = int(tup[0] != tup[1])
    M2 = int(tup[1] != tup[2])
    return {"tuple": tup, "M1": M1, "M2": M2, "sum": M1 + M2, "e2e": e2e}


if __name__ == "__main__":
    print("Lemma 1 enumeration:")
    for K in range(1, 7):
        ok, _ = check_xor_identity(K)
        print(f"  XOR identity (N=2, K={K}): {'PASS' if ok else 'FAIL'}")
    for N in (2, 3, 4):
        for K in range(1, 5):
            ok, strict = check_union_bound(N, K)
            tag = "strict-witnessed" if strict else "tight(=)"
            print(f"  union bound (N={N}, K={K}): {'PASS' if ok else 'FAIL'} [{tag}]")
    w = cancellation_witness()
    print("  cancellation witness (N=3,K=2):", w,
          "-> sum=2 but e2e=0" if w["sum"] == 2 and w["e2e"] == 0 else "UNEXPECTED")
