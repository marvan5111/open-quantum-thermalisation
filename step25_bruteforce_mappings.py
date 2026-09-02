import numpy as np, scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj
from itertools import permutations

gamma0,gamma1=0.15,0.10
dt=0.04
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N=4; I=np.eye(N,dtype=complex)
Ld=np.zeros((N*N,N*N),dtype=complex)
for C in [C0,C1]:
    Cd=C.conj().T@C
    Ld+=np.kron(C.conj(),C)
    Ld+=-0.5*np.kron(I,Cd)
    Ld+=-0.5*np.kron(Cd.T,I)
P=scipy.linalg.expm(Ld*(dt/2.0))
# extract kraus
d=int(np.sqrt(P.shape[0]))
J=np.zeros((d*d,d*d),dtype=complex)
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                J[i*d+k,j*d+l]=P[i + j*d, k + l*d]
vals,vecs=np.linalg.eigh(J)
Korig=[]
for idx,val in enumerate(vals):
    if val>1e-14:
        v=vecs[:,idx]
        K=np.sqrt(val)*v.reshape((d,d),order='F')
        Korig.append(K)
print('extracted',len(Korig),'kraus')
# transforms to try
trans_funcs = [
    ('K', lambda K: K),
    ('K_T', lambda K: K.T),
    ('K_H', lambda K: K.conj().T),
    ('K_conj', lambda K: K.conj()),
    ('K_Tconj', lambda K: K.T.conj()),
]
placements=['ancilla_major','system_major']
best=(1e9,None)
import math
for name,tf in trans_funcs:
    # apply transform to original list
    Klist = [tf(K) for K in Korig]
    # try permutations of Klist order
    for perm in permutations(range(len(Klist))):
        Ks=[Klist[i] for i in perm]
        # enforce completeness via polar correction
        M=sum(K.conj().T@K for K in Ks)
        try:
            inv_sqrtM=scipy.linalg.inv(scipy.linalg.sqrtm(M))
        except Exception:
            continue
        Kp=[K @ inv_sqrtM for K in Ks]
        # build V columns
        m=len(Kp)
        Vcols=[]
        for s in range(d):
            v=np.zeros((d*m,),dtype=complex)
            for k,K in enumerate(Kp): v[k*d:(k+1)*d]=K[:,s]
            Vcols.append(v)
        V=np.column_stack(Vcols)
        # build U by placing columns for ancilla=0 input
        D=d*m
        U=np.zeros((D,D),dtype=complex)
        for s in range(d): U[:,0*d+s]=Vcols[s]
        filled=[False]*D
        for s in range(d): filled[0*d+s]=True
        for j in range(D):
            if filled[j]: continue
            cand=np.zeros((D,),dtype=complex); cand[j]=1.0
            for k in range(D):
                if filled[k]: cand -= np.vdot(U[:,k],cand)*U[:,k]
            n=np.linalg.norm(cand)
            if n>1e-12:
                U[:,j]=cand/n; filled[j]=True
        for j in range(D):
            if not filled[j]:
                cand=np.random.randn(D)+1j*np.random.randn(D)
                for k in range(D):
                    if filled[k]: cand -= np.vdot(U[:,k],cand)*U[:,k]
                n=np.linalg.norm(cand)
                if n>1e-12: U[:,j]=cand/n; filled[j]=True
        # ensure unitary
        errU=np.linalg.norm(U.conj().T@U - np.eye(D))
        # compute S_from_U with ancilla_major ordering
        E=[]
        for i in range(d):
            for j in range(d):
                M=np.zeros((d,d),dtype=complex); M[i,j]=1.0; E.append(M)
        # do ancilla_major
        Scols=[]
        for M0 in E:
            rho_joint=np.zeros((d*m,d*m),dtype=complex)
            rho_joint[0:d,0:d]=M0
            out=U @ rho_joint @ U.conj().T
            rho_out=np.zeros((d,d),dtype=complex)
            for k in range(m): rho_out+=out[k*d:(k+1)*d, k*d:(k+1)*d]
            Scols.append(rho_out.reshape((d*d,),order='F'))
        S_am=np.column_stack(Scols)
        maxdiff_am=np.max(np.abs(S_am - P))
        # system_major
        Scols=[]
        for M0 in E:
            rho_joint=np.zeros((d*m,d*m),dtype=complex)
            for s in range(d):
                for t in range(d):
                    rho_joint[s + 0*d, t + 0*d] = M0[s,t]
            out=U @ rho_joint @ U.conj().T
            rho_out=np.zeros((d,d),dtype=complex)
            for a in range(m):
                for s in range(d):
                    for t in range(d):
                        rho_out[s,t] += out[s + a*d, t + a*d]
            Scols.append(rho_out.reshape((d*d,),order='F'))
        S_sm=np.column_stack(Scols)
        maxdiff_sm=np.max(np.abs(S_sm - P))
        # pick best of these two
        local_best = min((maxdiff_am,'ancilla'), (maxdiff_sm,'system'))
        if local_best[0] < best[0]:
            best=(local_best[0], name, perm, local_best[1], errU)
            print('NEW BEST', best)
print('DONE best=',best)
