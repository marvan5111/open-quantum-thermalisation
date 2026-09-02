import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import csv

# parameters
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_list = [10, 30, 60, 150, 300, 600]

# build H
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H = H_q.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0_base = np.sqrt(gamma0) * tensor(sm, I2).full()
C1_base = np.sqrt(gamma1) * tensor(I2, sm).full()
N = H.shape[0]
I = np.eye(N, dtype=complex)

# helpers
def L_D_from_C(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return Ld

def build_kraus_from_dilation(gamma_step):
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    qc3 = QuantumCircuit(3)
    qc3.swap(1, 2)
    qc3.cry(theta, 0, 1)
    qc3.cx(1, 0)
    qc3.swap(1, 2)
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

# metrics
def metrics(A, B):
    diff = A - B
    max_elem = np.max(np.abs(diff))
    frob = np.linalg.norm(diff, ord='fro')
    rel_frob = frob / np.linalg.norm(B, ord='fro')
    return max_elem, frob, rel_frob

rows = []
print('Running sweep over n:', n_list)
for n_steps in n_list:
    dt = T_final / n_steps
    dt_half = dt / 2.0
    # collapse operators scaled? base C had sqrt(gamma). For finite time, we still use same C and build L_D
    C0 = C0_base
    C1 = C1_base
    L_D = L_D_from_C([C0, C1])
    P_D_half = scipy.linalg.expm(L_D * dt_half)
    # per-qubit P_D halves
    P_D0_half = scipy.linalg.expm(L_D_from_C([C0]) * dt_half)
    P_D1_half = scipy.linalg.expm(L_D_from_C([C1]) * dt_half)

    # circuit Kraus for each qubit half-step
    gamma0_half = 1 - np.exp(-gamma0 * dt_half)
    gamma1_half = 1 - np.exp(-gamma1 * dt_half)
    K0 = build_kraus_from_dilation(gamma0_half)
    K1 = build_kraus_from_dilation(gamma1_half)
    S0 = sum(np.kron(K.conj(), K) for K in K0)
    S1 = sum(np.kron(K.conj(), K) for K in K1)
    S_circ_half = S1 @ S0

    # per-step metrics
    max_h, frob_h, rel_h = metrics(S_circ_half, P_D_half)

    # full Strang per-step
    U = scipy.linalg.expm(-1j * H * dt)
    U_super = np.kron(U.conj(), U)
    S_circ_full = S_circ_half @ U_super @ S_circ_half
    S_classical = P_D_half @ U_super @ P_D_half
    max_f, frob_f, rel_f = metrics(S_circ_full, S_classical)

    # also record per-qubit errors
    max_q0, frob_q0, rel_q0 = metrics(S0, P_D0_half)
    max_q1, frob_q1, rel_q1 = metrics(S1, P_D1_half)

    print(f'n={n_steps:4d} dt={dt:.5f} per-step max={max_h:.6e} rel_frob={rel_h:.6e} full-step max={max_f:.6e} rel_frob={rel_f:.6e}')
    rows.append([n_steps, dt, max_q0, frob_q0, rel_q0, max_q1, frob_q1, rel_q1, max_h, frob_h, rel_h, max_f, frob_f, rel_f])

# save CSV
header = ['n_steps','dt','max_q0','frob_q0','rel_q0','max_q1','frob_q1','rel_q1','max_half','frob_half','rel_half','max_full','frob_full','rel_full']
with open('/home/marvan-mahamood/qsim/trotter_circuit_sweep_results.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
print('\nSaved /home/marvan-mahamood/qsim/trotter_circuit_sweep_results.csv')
