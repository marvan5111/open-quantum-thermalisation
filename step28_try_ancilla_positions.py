import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj

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
# Choi-root Kraus with transpose fix

d=int(np.sqrt(P.shape[0]))
J=np.zeros((d*d,d*d),dtype=complex)
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                J[i*d+k,j*d+l]=P[i + j*d, k + l*d]
vals,vecs=np.linalg.eigh(J)
nonzero=[i for i,v in enumerate(vals) if v>1e-16]
Vch = np.column_stack([np.sqrt(vals[i])*vecs[:,i] for i in nonzero])
# reshape columns with transpose
Klist=[]
for col in range(Vch.shape[1]):
    K = Vch[:,col].reshape((d,d), order='F').T
    Klist.append(K)
# enforce completeness
M=sum(K.conj().T@K for K in Klist)
inv_sqrtM=scipy.linalg.inv(scipy.linalg.sqrtm(M))
Kp=[K @ inv_sqrtM for K in Klist]
print('Kp count', len(Kp))
# Build Vcols
m=len(Kp)
Vcols=[]
for s in range(d):
    v=np.zeros((d*m,),dtype=complex)
    for k,K in enumerate(Kp):
        v[k*d:(k+1)*d]=K[:,s]
    Vcols.append(v)

best=(1e9,None)
# try placing Vcols into ancilla position a0 (0..m-1), i.e., columns indices a0*d + s
for a0 in range(m):
    D=d*m
    U=np.zeros((D,D),dtype=complex)
    filled=[False]*D
    for s in range(d):
        idx = a0*d + s
        U[:, idx] = Vcols[s]
        filled[idx]=True
    # fill remaining columns
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
    # test S_from_U
    E=[]
    for i in range(d):
        for j in range(d):
            M=np.zeros((d,d),dtype=complex); M[i,j]=1.0; E.append(M)
    Scols=[]
    for M in E:
        rho_joint=np.zeros((d*m,d*m),dtype=complex)
        # ancilla-major: ancilla index fastest
        rho_joint[0*d:0*d+d, 0*d:0*d+d] = M
        out = U @ rho_joint @ U.conj().T
        rho_out=np.zeros((d,d),dtype=complex)
        for k in range(m): rho_out += out[k*d:(k+1)*d, k*d:(k+1)*d]
        Scols.append(rho_out.reshape((d*d,),order='F'))
    S_from_U = np.column_stack(Scols)
    err = np.max(np.abs(S_from_U - P))
    print('a0',a0,'err',err)
    if err < best[0]: best=(err, a0)

print('best placement',best)
