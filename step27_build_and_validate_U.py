import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis
import csv

# params
gamma0,gamma1=0.15,0.10
T_final=6.0
n_list=[10,30,60,150,300,600]

sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N=4; I=np.eye(N,dtype=complex)

# function to get P_D_half and Kp from Choi-root with transpose fix
def get_P_and_Kp(dt):
    Ld=np.zeros((N*N,N*N),dtype=complex)
    for C in [C0,C1]:
        Cd=C.conj().T@C
        Ld+=np.kron(C.conj(),C)
        Ld+=-0.5*np.kron(I,Cd)
        Ld+=-0.5*np.kron(Cd.T,I)
    P=scipy.linalg.expm(Ld*(dt/2.0))
    # Choi
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
    return P, Kp

# function to build U from Kp via V columns and completion (place V columns in ancilla=0 block)
def build_U_from_Kp(Kp):
    d=Kp[0].shape[0]
    m=len(Kp)
    Vcols=[]
    for s in range(d):
        v=np.zeros((d*m,),dtype=complex)
        for k,K in enumerate(Kp):
            v[k*d:(k+1)*d]=K[:,s]
        Vcols.append(v)
    D=d*m
    U=np.zeros((D,D),dtype=complex)
    # place columns for ancilla=0 input
    for s in range(d): U[:,0*d+s]=Vcols[s]
    filled=[False]*D
    for s in range(d): filled[0*d+s]=True
    # fill remaining columns by Gram-Schmidt
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
    return U

results=[]
for n_steps in n_list:
    dt=T_final/n_steps
    P, Kp = get_P_and_Kp(dt)
    m=len(Kp)
    print('n',n_steps,'dt',dt,'Kp count',m)
    U = build_U_from_Kp(Kp)
    # validate S_from_U
    d=N
    E=[]
    for i in range(d):
        for j in range(d):
            M=np.zeros((d,d),dtype=complex); M[i,j]=1.0; E.append(M)
    Scols=[]
    for M in E:
        rho_joint=np.zeros((d*m,d*m),dtype=complex)
        rho_joint[0:d,0:d]=M
        out = U @ rho_joint @ U.conj().T
        rho_out=np.zeros((d,d),dtype=complex)
        for k in range(m): rho_out += out[k*d:(k+1)*d, k*d:(k+1)*d]
        Scols.append(rho_out.reshape((d*d,),order='F'))
    S_from_U = np.column_stack(Scols)
    maxdiff = np.max(np.abs(S_from_U - P))
    frob = np.linalg.norm(S_from_U - P, ord='fro')
    print('  map diff max',maxdiff,'frob',frob)
    # Compose and compare
    # build H from earlier constants
    sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
    H_q = 1.0 * tensor(sx, sx) + 0.7 * tensor(sz, I2) + 1.1 * tensor(I2, sz)
    H = H_q.full()
    U_unitary = scipy.linalg.expm(-1j * H * dt)
    U_super = np.kron(U_unitary.conj(), U_unitary)
    S_circ_full = S_from_U @ U_super @ S_from_U
    S_classical = P @ U_super @ P
    d2=d*d
    psi0 = (basis(2,0)+basis(2,1)).unit()
    psi_full = tensor(psi0, basis(2,0))
    rho0 = psi_full.proj().full()
    vec0 = rho0.reshape((d2,), order='F')
    S_circ_n = np.linalg.matrix_power(S_circ_full, n_steps)
    S_class_n = np.linalg.matrix_power(S_classical, n_steps)
    rho_circ_final = (S_circ_n @ vec0).reshape((d,d), order='F')
    rho_class_final = (S_class_n @ vec0).reshape((d,d), order='F')
    max_final = np.max(np.abs(rho_circ_final - rho_class_final))
    print('  final diff', max_final)
    results.append([n_steps, dt, m, maxdiff, frob, max_final])

with open('/home/marvan-mahamood/qsim/stinespring_validated_results.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['n_steps','dt','ancilla_dim','maxdiff_map','frob_map','max_final_diff'])
    writer.writerows(results)
print('\nSaved /home/marvan-mahamood/qsim/stinespring_validated_results.csv')
