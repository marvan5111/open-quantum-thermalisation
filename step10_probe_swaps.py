import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# params
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_steps = 150
dt = T_final / n_steps
dt_half = dt/2.0

# set up collapse ops
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N = 4
I = np.eye(N, dtype=complex)

def L_D_from_C(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return Ld

P_D0_half = scipy.linalg.expm(L_D_from_C([C0]) * dt_half)
P_D1_half = scipy.linalg.expm(L_D_from_C([C1]) * dt_half)

# helper to build K with custom swap logic

def build_kraus_custom(qubit_idx, gamma_step, swap_flag):
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    qc3 = QuantumCircuit(3)
    swapped = False
    if swap_flag:
        qc3.swap(1,2)
        swapped = True
    qc3.cry(theta, 0, 1)
    qc3.cx(1, 0)
    if swapped:
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

# try swap_flag False/True and both qubits
for swap_flag in [False, True]:
    print('\n--- swap_flag =', swap_flag, '---')
    gamma0_half = 1 - np.exp(-gamma0 * dt_half)
    gamma1_half = 1 - np.exp(-gamma1 * dt_half)
    K0 = build_kraus_custom(0, gamma0_half, swap_flag)
    K1 = build_kraus_custom(1, gamma1_half, swap_flag)
    def superop(Klist):
        S=np.zeros((N*N,N*N),dtype=complex)
        for K in Klist:
            S += np.kron(K.conj(), K)
        return S
    S0 = superop(K0)
    S1 = superop(K1)
    print('max diff S0 vs P_D0_half:', np.max(np.abs(S0 - P_D0_half)))
    print('max diff S1 vs P_D1_half:', np.max(np.abs(S1 - P_D1_half)))

# Also try per-qubit different swap choices: swap only for qubit1
print('\n--- mixed: swap only for qubit1 ---')
K0 = build_kraus_custom(0, 1 - np.exp(-gamma0*dt_half), False)
K1 = build_kraus_custom(1, 1 - np.exp(-gamma1*dt_half), True)
S0 = sum(np.kron(K.conj(),K) for K in K0)
S1 = sum(np.kron(K.conj(),K) for K in K1)
print('max diff S0 vs P_D0_half:', np.max(np.abs(S0 - P_D0_half)))
print('max diff S1 vs P_D1_half:', np.max(np.abs(S1 - P_D1_half)))

# and swap only for qubit0
print('\n--- mixed: swap only for qubit0 ---')
K0 = build_kraus_custom(0, 1 - np.exp(-gamma0*dt_half), True)
K1 = build_kraus_custom(1, 1 - np.exp(-gamma1*dt_half), False)
S0 = sum(np.kron(K.conj(),K) for K in K0)
S1 = sum(np.kron(K.conj(),K) for K in K1)
print('max diff S0 vs P_D0_half:', np.max(np.abs(S0 - P_D0_half)))
print('max diff S1 vs P_D1_half:', np.max(np.abs(S1 - P_D1_half)))
