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
print('extracted',len(Klist),'kraus')
# check completeness
M=sum(K.conj().T@K for K in Klist)
print('||M-I||',np.linalg.norm(M-np.eye(d)))
# adjust K
sqrtM = scipy.linalg.sqrtm(M)
inv_sqrtM = scipy.linalg.inv(sqrtM)
Kp=[K @ inv_sqrtM for K in Klist]
M2=sum(K.conj().T@K for K in Kp)
print('||M2-I||',np.linalg.norm(M2-np.eye(d)))
# build V from Kp
m=len(Kp)
V=np.zeros((d*m,d),dtype=complex)
for k,K in enumerate(Kp): V[k*d:(k+1)*d,:]=K
print('V^dag V diff',np.linalg.norm(V.conj().T@V - np.eye(d)))
# build U from V
Qv,R=np.linalg.qr(V)
U=np.zeros((d*m,d*m),dtype=complex)
U[:,:d]=Qv[:,:d]
col=d
for i in range(d*m):
    if col>=d*m: break
    cand=np.zeros((d*m,),dtype=complex); cand[i]=1.0
    for j in range(col): cand-=np.vdot(U[:,j],cand)*U[:,j]
    n=np.linalg.norm(cand)
    if n>1e-12: U[:,col]=cand/n; col+=1
while col<d*m:
    cand=np.random.randn(d*m)+1j*np.random.randn(d*m)
    for j in range(col): cand-=np.vdot(U[:,j],cand)*U[:,j]
    n=np.linalg.norm(cand)
    if n>1e-12: U[:,col]=cand/n; col+=1
# build S_from_U
E=[]
for i in range(d):
    for j in range(d):
        M=np.zeros((d,d),dtype=complex); M[i,j]=1.0; E.append(M)
Scols=[]
for M in E:
    rho_joint=np.zeros((d*m,d*m),dtype=complex)
    rho_joint[0:d,0:d]=M
    out=U @ rho_joint @ U.conj().T
    rho_out=np.zeros((d,d),dtype=complex)
    for k in range(m): rho_out+=out[k*d:(k+1)*d,k*d:(k+1)*d]
    Scols.append(rho_out.reshape((d*d,),order='F'))
S_from_U=np.column_stack(Scols)
print('max diff S_from_U vs P',np.max(np.abs(S_from_U-P)))
# Also check S_from_Kp directly
S_from_Kp=sum(np.kron(K.conj(),K) for K in Kp)
print('max diff S_from_Kp vs P',np.max(np.abs(S_from_Kp - P)))
PY
