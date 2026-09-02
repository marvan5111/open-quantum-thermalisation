import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# parameters
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0

sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N = 4
I4 = np.eye(N, dtype=complex)

def L_D_from_C(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I4, Cd)
        Ld += -0.5 * np.kron(Cd.T, I4)
    return Ld

# circuit map for damp half-step

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

L_D = L_D_from_C([C0, C1])
print('Evaluating generator mismatch for circuit half-step damping map:')
for dt in [0.6, 0.2, 0.1, 0.04, 0.02, 0.01]:
    dt_half = dt/2.0
    gamma0_half = 1 - np.exp(-gamma0 * dt_half)
    gamma1_half = 1 - np.exp(-gamma1 * dt_half)
    K0 = build_kraus_from_dilation(gamma0_half)
    K1 = build_kraus_from_dilation(gamma1_half)
    S0 = sum(np.kron(K.conj(), K) for K in K0)
    S1 = sum(np.kron(K.conj(), K) for K in K1)
    S_circ_half = S1 @ S0
    P_D_half = scipy.linalg.expm(L_D * dt_half)
    # generator estimates
    A_circ = (S_circ_half - np.eye(N*N, dtype=complex)) / dt_half
    A_target = L_D
    diff = A_circ - A_target
    max_diff = np.max(np.abs(diff))
    frob = np.linalg.norm(diff, ord='fro')
    print(f'dt={dt:.5f}  max|A_circ - L_D|={max_diff:.6e}  frob={frob:.6e}')
    # compare mismatch per superop step itself
    step_err = np.max(np.abs(S_circ_half - P_D_half))
    print(f'      step_err(max)={step_err:.6e}  rel_frob={np.linalg.norm(S_circ_half-P_D_half,ord="fro")/np.linalg.norm(P_D_half,ord="fro"):.6e}')
