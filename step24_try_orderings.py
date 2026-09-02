import numpy as np, scipy.linalg
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
# extract kraus

d=int(np.sqrt(P.shape[0]))
J=np.zeros((d*d,d*d),dtype=complex)
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                J[i*d+k,j*d+l]=P[i + j*d, k + l*d]
vals,vecs=np.linalg.eigh(J)
Klist=[]
for idx,val in enumerate(vals):
    if val>1e-14:
        v=vecs[:,idx]
        K=np.sqrt(val)*v.reshape((d,d),order='F')
        Klist.append(K)
# adjust
M=sum(K.conj().T@K for K in Klist)
inv_sqrtM=scipy.linalg.inv(scipy.linalg.sqrtm(M))
Kp=[K @ inv_sqrtM for K in Klist]
# Build Vcols
m=len(Kp)
Vcols=[]
for s in range(d):
    v=np.zeros((d*m,),dtype=complex)
    for k,K in enumerate(Kp):
        v[k*d:(k+1)*d]=K[:,s]
    Vcols.append(v)
V=np.column_stack(Vcols)
# build U explicitly as before
D=d*m
U=np.zeros((D,D),dtype=complex)
for s in range(d): U[:, 0*d + s] = Vcols[s]
# fill remaining
filled=[False]*D
for s in range(d): filled[0*d + s]=True
for j in range(D):
    if filled[j]: continue
    cand=np.zeros((D,),dtype=complex); cand[j]=1.0
    for k in range(D):
        if filled[k]: cand -= np.vdot(U[:,k], cand)*U[:,k]
    norm=np.linalg.norm(cand)
    if norm>1e-12:
        U[:,j]=cand/norm; filled[j]=True
for j in range(D):
    if not filled[j]:
        cand=np.random.randn(D)+1j*np.random.randn(D)
        for k in range(D):
            if filled[k]: cand -= np.vdot(U[:,k],cand)*U[:,k]
        norm=np.linalg.norm(cand)
        if norm>1e-12: U[:,j]=cand/norm; filled[j]=True
print('U unitary err', np.linalg.norm(U.conj().T@U - np.eye(D)))
# Try both ordering conventions when building rho_joint and partial trace
E=[]
for i in range(d):
    for j in range(d):
        M=np.zeros((d,d),dtype=complex); M[i,j]=1.0; E.append(M)

for ordering in ['ancilla_major','system_major']:
    Scols=[]
    for M0 in E:
        rho_joint=np.zeros((d*m,d*m),dtype=complex)
        if ordering=='ancilla_major':
            # place ancilla |0><0| kron M at block 0
            rho_joint[0:d,0:d] = M0
            out = U @ rho_joint @ U.conj().T
            rho_out = np.zeros((d,d),dtype=complex)
            for k in range(m): rho_out += out[k*d:(k+1)*d, k*d:(k+1)*d]
        else:
            # system-major ordering: index = s + a*d
            # build rho_joint such that entries [s + a*d, t + b*d] = (ancilla_ab * M_st)
            # for ancilla |0><0|, ancilla_ab = 1 only when a=b=0
            # so positions s + 0*d, t + 0*d correspond to M[s,t]
            for s in range(d):
                for t in range(d):
                    rho_joint[s + 0*d, t + 0*d] = M0[s,t]
            out = U @ rho_joint @ U.conj().T
            # partial trace over ancilla: sum_a (out[s + a*d, t + a*d]) ??? need to map
            rho_out = np.zeros((d,d),dtype=complex)
            for a in range(m):
                for s in range(d):
                    for t in range(d):
                        rho_out[s,t] += out[s + a*d, t + a*d]
        Scols.append(rho_out.reshape((d*d,),order='F'))
    S = np.column_stack(Scols)
    print('ordering',ordering,'maxdiff',np.max(np.abs(S - P)))

# Also compare S_from_Kp directly
S_from_Kp=sum(np.kron(K.conj(),K) for K in Kp)
print('S_from_Kp vs P maxdiff',np.max(np.abs(S_from_Kp - P)))
