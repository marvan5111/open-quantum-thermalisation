import numpy as np, scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis

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
# adjust K to enforce completeness
M=sum(K.conj().T@K for K in Klist)
inv_sqrtM=scipy.linalg.inv(scipy.linalg.sqrtm(M))
Kp=[K @ inv_sqrtM for K in Klist]
# Build V columns v_s = sum_k |k> ⊗ K_k e_s
m=len(Kp)
Vcols=[]
for s in range(d):
    v=np.zeros((d*m,),dtype=complex)
    for k,K in enumerate(Kp):
        vec=K[:,s]
        v[k*d:(k+1)*d]=vec
    Vcols.append(v)
V=np.column_stack(Vcols)
print('V^dag V norm', np.linalg.norm(V.conj().T@V - np.eye(d)))
# Construct U explicitly: set columns corresponding to input ancilla=0 (indices col = 0*d + s) to Vcols
D=d*m
U=np.zeros((D,D),dtype=complex)
# place specified columns
for s in range(d):
    col_idx = 0*d + s
    U[:, col_idx] = Vcols[s]
# Now fill remaining columns by Gram-Schmidt while preserving existing ones
filled = [0]*D
for s in range(d): filled[0*d + s]=1
col = d
for j in range(D):
    if filled[j]: continue
    # candidate basis vector
    cand=np.zeros((D,),dtype=complex)
    cand[j]=1.0
    # orthonormalize
    for k in range(D):
        if filled[k]:
            proj = np.vdot(U[:,k], cand)
            cand -= proj * U[:,k]
    norm = np.linalg.norm(cand)
    if norm>1e-12:
        U[:, j] = cand/norm
        filled[j]=1
# ensure unfilled filled by random orthonormal
for j in range(D):
    if not filled[j]:
        cand=np.random.randn(D)+1j*np.random.randn(D)
        for k in range(D):
            if filled[k]: cand -= np.vdot(U[:,k],cand)*U[:,k]
        norm=np.linalg.norm(cand)
        if norm>1e-12:
            U[:,j]=cand/norm; filled[j]=1
# Validate U is unitary (approx)
print('||U^dag U - I||', np.linalg.norm(U.conj().T@U - np.eye(D)))
# Build S_from_U
E=[]
for i in range(d):
    for j in range(d):
        M=np.zeros((d,d),dtype=complex); M[i,j]=1.0; E.append(M)
Scols=[]
for M0 in E:
    rho_joint=np.zeros((d*m,d*m),dtype=complex)
    rho_joint[0:d,0:d]=M0
    out=U @ rho_joint @ U.conj().T
    rho_out=np.zeros((d,d),dtype=complex)
    for k in range(m): rho_out += out[k*d:(k+1)*d, k*d:(k+1)*d]
    Scols.append(rho_out.reshape((d*d,),order='F'))
S_from_U=np.column_stack(Scols)
print('max diff S_from_U vs P', np.max(np.abs(S_from_U - P)))
# Also direct Kraus
S_from_K=sum(np.kron(K.conj(),K) for K in Kp)
print('max diff S_from_K vs P', np.max(np.abs(S_from_K - P)))
PY
