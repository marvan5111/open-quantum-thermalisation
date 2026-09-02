import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis
import csv

# params
gamma0,gamma1=0.15,0.10
T_final=6.0
n_list=[10,30,60,150,300,600]

sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = 1.0 * tensor(sx, sx) + 0.7 * tensor(sz, I2) + 1.1 * tensor(I2, sz)
H = H_q.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N=4; I=np.eye(N,dtype=complex)

def get_exact_Kp(dt):
    Ld=np.zeros((N*N,N*N),dtype=complex)
    for C in [C0,C1]:
        Cd=C.conj().T@C
        Ld+=np.kron(C.conj(),C)
        Ld+=-0.5*np.kron(I,Cd)
        Ld+=-0.5*np.kron(Cd.T,I)
    P=scipy.linalg.expm(Ld*(dt/2.0))
    # Choi-root with transpose fix
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
    # reshape with transpose
    Klist=[Vch[:,col].reshape((d,d),order='F').T for col in range(Vch.shape[1])]
    # enforce completeness
    M=sum(K.conj().T@K for K in Klist)
    inv_sqrtM=scipy.linalg.inv(scipy.linalg.sqrtm(M))
    Kp=[K @ inv_sqrtM for K in Klist]
    return Kp, P

results=[]
for n_steps in n_list:
    dt=T_final/n_steps
    Kp_half, P_half = get_exact_Kp(dt)
    # exact Strang: S_exact = P_half @ exp(-iH dt) @ P_half
    U = scipy.linalg.expm(-1j * H * dt)
    U_super = np.kron(U.conj(), U)
    # exact dissipator superop
    S_exact_half = sum(np.kron(K.conj(),K) for K in Kp_half)
    S_exact_full = S_exact_half @ U_super @ S_exact_half
    
    # initial state
    psi0 = (basis(2,0)+basis(2,1)).unit()
    psi_full = tensor(psi0, basis(2,0))
    rho0 = psi_full.proj().full()
    vec0 = rho0.reshape((N*N,), order='F')
    
    # apply n steps via matrix power
    S_n = np.linalg.matrix_power(S_exact_full, n_steps)
    rho_final = (S_n @ vec0).reshape((N,N), order='F')
    
    # compare to classical Strang (using exact Liouvillian exp)
    Ld=np.zeros((N*N,N*N),dtype=complex)
    for C in [C0,C1]:
        Cd=C.conj().T@C
        Ld+=np.kron(C.conj(),C)
        Ld+=-0.5*np.kron(I,Cd)
        Ld+=-0.5*np.kron(Cd.T,I)
    P_class_half = scipy.linalg.expm(Ld * (dt/2.0))
    S_class_full = P_class_half @ U_super @ P_class_half
    S_class_n = np.linalg.matrix_power(S_class_full, n_steps)
    rho_class_final = (S_class_n @ vec0).reshape((N,N), order='F')
    
    # per-step metric
    S_exact_half_err = np.max(np.abs(S_exact_half - P_class_half))
    
    # final diff
    max_diff = np.max(np.abs(rho_final - rho_class_final))
    frob_diff = np.linalg.norm(rho_final - rho_class_final, ord='fro')
    
    print(f'n={n_steps:4d} dt={dt:.5f} per-step-err={S_exact_half_err:.6e} final-max={max_diff:.6e} final-frob={frob_diff:.6e}')
    results.append([n_steps, dt, S_exact_half_err, max_diff, frob_diff])

with open('/home/marvan-mahamood/qsim/exact_kraus_convergence.csv','w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['n_steps','dt','per_step_error','final_max_diff','final_frob_diff'])
    writer.writerows(results)
print('\nSaved /home/marvan-mahamood/qsim/exact_kraus_convergence.csv')
