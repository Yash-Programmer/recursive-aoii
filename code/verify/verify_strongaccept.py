import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from raoii import joint
ok=lambda b:"PASS" if b else "**FAIL**"
TINY=1e-9

def popcount(z):
    c=0
    while z: c+=z&1; z>>=1
    return c

def build_Z(mus,nu):
    """Edit D: the disagreement field Z=(Z_1..Z_K) as a Markov chain on {0,1}^K.
       z_k stored at bit k-1. Source toggles z_1 (rate nu); link k (rate mu_k):
       if z_k==1 -> z_k:=0 and toggle z_{k+1} (k<K) / remove (k=K)."""
    K=len(mus); n=1<<K; Q=np.zeros((n,n))
    for z in range(n):
        Q[z, z^1]+=nu                                   # source toggles z_1 (bit0)
        for k in range(1,K+1):
            if (z>>(k-1))&1:
                z2=z & ~(1<<(k-1))                       # z_k -> 0
                if k<K: z2 ^= (1<<k)                     # toggle z_{k+1}
                if z2!=z: Q[z,z2]+=mus[k-1]
    np.fill_diagonal(Q,0.0); Q[np.diag_indices(n)]=-Q.sum(axis=1)
    return Q,n

def stat(Q):
    n=Q.shape[0]; A=np.vstack([Q.T,np.ones(n)]); b=np.zeros(n+1); b[-1]=1
    pi,*_=np.linalg.lstsq(A,b,rcond=None); return pi

def analyze_Z(mus,nu):
    Q,n=build_Z(mus,nu); pi=stat(Q)
    wrong=np.array([popcount(z)&1==1 for z in range(n)])
    pK=pi[wrong].sum()
    A=Q[np.ix_(wrong,wrong)]; piM=pi[wrong]
    raoii=-piM@np.linalg.solve(A,np.ones(wrong.sum()))
    return pK,raoii

RATEVECS=[[2.],[2.,3.],[1.,2.,3.],[2.,3.,1.5],[1.,2.,3.,4.],[5.,1.,4.,2.,3.],
          [0.5,0.8,1.2,2.0],[3.,3.,3.,3.,3.]]
NUS=[0.5,1.0,2.0]

print("="*78)
print("V1 - Lemma 6a (CROWN JEWEL): raoii <= p_K/nu < 1/(2nu), for all K/rates")
print("="*78)
allpass=True
for nu in NUS:
    for mus in RATEVECS:
        a=joint.analyze(mus,nu)
        pK,rao=a['p_K'],a['raoii']
        c1=rao<=pK/nu+TINY; c2=pK<0.5+TINY; c3=rao<1/(2*nu)+TINY
        allpass &= c1 and c2 and c3
        if not(c1 and c2 and c3):
            print(f"  FAIL nu={nu} mus={mus}: rao={rao:.5f} pK/nu={pK/nu:.5f} 1/2nu={1/(2*nu):.5f}")
print(f"  all (raoii<=p_K/nu<1/2nu) over {len(NUS)*len(RATEVECS)} configs: {ok(allpass)}")
# tightness in frozen corner
for nu in [1.0]:
    a=joint.analyze([1e-4],nu); print(f"  frozen mu->0, K=1: raoii={a['raoii']:.5f} -> 1/(2nu)={1/(2*nu):.5f} (tight)")

print("="*78)
print("V2 - w_bar = max_i(-A^{-1}1)_i <= 1/nu  (mean residual wrong-run)")
print("="*78)
allpass=True
for nu in NUS:
    for mus in RATEVECS:
        Q,K=joint.build_generator(mus,nu); pi=joint.stationary(Q); M=joint.mismatch_mask(K)
        A=Q[np.ix_(M,M)]; w=-np.linalg.solve(A,np.ones(M.sum())); wbar=w.max()
        c=wbar<=1/nu+TINY; allpass&=c
        if not c: print(f"  FAIL nu={nu} mus={mus}: wbar={wbar:.5f} 1/nu={1/nu:.5f}")
print(f"  all (wbar<=1/nu): {ok(allpass)}")

print("="*78)
print("V3 - m_k <= alpha_k for all k  (per-hop disagreement <= source factor)")
print("="*78)
allpass=True; mono=True
for nu in NUS:
    for mus in RATEVECS:
        m=joint.hop_mismatch(mus,nu); mus_a=np.asarray(mus,float)
        alpha=nu/(mus_a+2*nu)
        c=np.all(m<=alpha+TINY); allpass&=c
        if not c: print(f"  FAIL nu={nu} mus={mus}: m={m} alpha={alpha}")
        # equal-rate monotonicity of m_k
        if len(set(mus))==1 and len(mus)>=3:
            mono &= np.all(np.diff(m)<=TINY)
print(f"  all (m_k<=alpha_k): {ok(allpass)}")
print(f"  equal-rate m_k nonincreasing in k: {ok(mono)}")

print("="*78)
print("V4 - Edit C surrogate: nu_eff_k = mu_{k-1} m_{k-1} < nu, nonincreasing")
print("="*78)
allpass=True; mono=True
for nu in NUS:
    for mus in RATEVECS:
        if len(mus)<2: continue
        m=joint.hop_mismatch(mus,nu); mus_a=np.asarray(mus,float)
        nueff=np.array([mus_a[k-1]*m[k-1] for k in range(1,len(mus))])  # k=2..K
        c=np.all(nueff<nu+TINY); allpass&=c
        if not c: print(f"  FAIL nu={nu} mus={mus}: nueff={nueff}")
        if len(set(mus))==1: mono &= np.all(np.diff(nueff)<=TINY)
print(f"  all (nu_eff_k<nu): {ok(allpass)}")
print(f"  equal-rate nu_eff nonincreasing: {ok(mono)}")

print("="*78)
print("V5 - Edit B: NQD worst ratio<=1 AND unconditional min-bound P(Mi,Mj)<=min(mi,mj)")
print("="*78)
nqd=True; frechet=True
for nu in NUS:
    for mus in RATEVECS:
        if len(mus)<2: continue
        r=joint.nqd_worst_ratio(mus,nu); nqd &= r<=1+1e-7
        m=joint.hop_mismatch(mus,nu); P=joint.pair_mismatch(mus,nu); K=len(mus)
        for i in range(1,K+1):
            for j in range(i+1,K+1):
                frechet &= P[i,j]<=min(m[i-1],m[j-1])+TINY
print(f"  NQD (worst ratio<=1, all configs): {ok(nqd)}")
print(f"  unconditional Frechet P(Mi,Mj)<=min(mi,mj): {ok(frechet)}")

print("="*78)
print("V6 - Edit D: Z-chain (2^K states) reproduces p_K and raoii of joint (2^{K+1})")
print("="*78)
allpass=True
for nu in [1.0,2.0]:
    for mus in RATEVECS:
        a=joint.analyze(mus,nu); pZ,rZ=analyze_Z(mus,nu)
        c=abs(pZ-a['p_K'])<1e-7 and abs(rZ-a['raoii'])<1e-7; allpass&=c
        if not c: print(f"  FAIL nu={nu} mus={mus}: Z=({pZ:.6f},{rZ:.6f}) joint=({a['p_K']:.6f},{a['raoii']:.6f})")
print(f"  Z-chain == joint for p_K and raoii: {ok(allpass)}")

print("="*78)
print("V7 - saturation: raoii monotone increasing in K, bounded (nu=1,mu=2)")
print("="*78)
vals=[joint.analyze([2.0]*K,1.0)['raoii'] for K in range(1,11)]
incs=np.diff(vals)
print(f"  monotone increasing: {ok(np.all(incs>0))}; all < 1/(2nu)=0.5: {ok(all(v<0.5 for v in vals))}")
print(f"  increments: {['%.4f'%i for i in incs]}")
