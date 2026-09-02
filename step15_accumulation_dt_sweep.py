import numpy as np
import scipy.linalg
from qutip import mesolve, basis, sigmax, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import csv

# parameters
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_list = [10, 30, 60, 150, 300, 600]

# initial state
from qutip import basis
psi0 = tensor((basis(2,0)+basis(2,1)).unit(), basis(2,0))
rho0 = psi0.proj().full()
N = rho0.shape[0]
vec0 = rho0.reshape((N*N,), order='F')

# build H and collapse operators for QuTiP exact reference
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2)
C1 = np.sqrt(gamma1) * tensor(I2, sm)

# compute QuTiP exact final state (independent of splitting)
result = mesolve(H_q, psi0, [T_final], c_ops=[C0, C1])
rho_qutip = result.states[-1].full()

# helpers
I = np.eye(N, dtype=complex)

def L_D_from_C(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cmat = C.full()
        Cd = Cmat.conj().T @ Cmat
        Ld += np.kron(Cmat.conj(), Cmat)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return Ld

from math import asin, sqrt

def build_kraus_from_dilation(gamma_step):
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    qc3 = QuantumCircuit(3)
    qc3.swap(1,2)
    qc3.cry(theta, 0, 1)
    qc3.cx(1, 0)
    qc3.swap(1,2)
    U8 = Operator(qc3).data
    K_list = []
    for k in range(2):
        K = np.zeros((4,4), dtype=complex)
        for sp in range(4):
            row = (k << 2) | sp
            for s in range(4):
                col = (0 << 2) | s
                K[sp, s] = U8[row, col]
        K_list.append(K)
    return K_list

rows = []
print('Running accumulation vs dt sweep for n_list=', n_list)
for n_steps in n_list:
    dt = T_final / n_steps
    dt_half = dt / 2.0
    # classical maps
    L_D = L_D_from_C([C0, C1])
    P_D_half = scipy.linalg.expm(L_D * dt_half)
    U = scipy.linalg.expm(-1j * H_q.full() * dt)
    U_super = np.kron(U.conj(), U)
    S_classical = P_D_half @ U_super @ P_D_half
    # circuit maps
    gamma0_half = 1 - np.exp(-gamma0 * dt_half)
    gamma1_half = 1 - np.exp(-gamma1 * dt_half)
    K0 = build_kraus_from_dilation(gamma0_half)
    K1 = build_kraus_from_dilation(gamma1_half)
    S0 = sum(np.kron(K.conj(), K) for K in K0)
    S1 = sum(np.kron(K.conj(), K) for K in K1)
    S_circ_half = S1 @ S0
    S_circ_full = S_circ_half @ U_super @ S_circ_half

    # per-step metric
    diff_half = S_circ_half - P_D_half
    max_half = np.max(np.abs(diff_half))
    frob_half = np.linalg.norm(diff_half, ord='fro')
    rel_half = frob_half / np.linalg.norm(P_D_half, ord='fro')

    # final after n steps via matrix power
    S_classical_n = np.linalg.matrix_power(S_classical, n_steps)
    S_circ_n = np.linalg.matrix_power(S_circ_full, n_steps)
    vec_class_final = S_classical_n @ vec0
    vec_circ_final = S_circ_n @ vec0
    rho_class_final = vec_class_final.reshape((N,N), order='F')
    rho_circ_final = vec_circ_final.reshape((N,N), order='F')

    # comparisons
    def metrics(A,B):
        max_e = np.max(np.abs(A-B))
        frob = np.linalg.norm(A-B, ord='fro')
        rel = frob / np.linalg.norm(B, ord='fro')
        return max_e, frob, rel
    max_classical_vs_qutip, frob_cq, rel_cq = metrics(rho_class_final, rho_qutip)
    max_circ_vs_qutip, frob_ct, rel_ct = metrics(rho_circ_final, rho_qutip)
    max_circ_vs_class, frob_cc, rel_cc = metrics(rho_circ_final, rho_class_final)

    print(f'n={n_steps:4d} dt={dt:.5f} per-step max={max_half:.6e} final circ-vs-class max={max_circ_vs_class:.6e} circ-vs-qutip max={max_circ_vs_qutip:.6e}')

    rows.append([n_steps, dt, max_half, frob_half, rel_half, max_circ_vs_class, frob_cc, rel_cc, max_circ_vs_qutip, frob_ct, rel_ct, max_classical_vs_qutip, frob_cq, rel_cq])

# save CSV
header = ['n_steps','dt','max_half','frob_half','rel_half','max_circ_vs_class','frob_cc','rel_cc','max_circ_vs_qutip','frob_ct','rel_ct','max_classical_vs_qutip','frob_cq','rel_cq']
with open('/home/marvan-mahamood/qsim/accumulation_vs_dt_sweep.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
print('\nSaved /home/marvan-mahamood/qsim/accumulation_vs_dt_sweep.csv')
