import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# params
gamma0, gamma1 = 0.15, 0.10

from math import asin, sqrt

def build_kraus_full(gamma_step):
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    qc3 = QuantumCircuit(3)
    qc3.swap(1,2)
    qc3.cry(theta, 0, 1)
    qc3.cx(1,0)
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

from qutip import sigmax, sigmaz, qeye, tensor
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N=4; I=np.eye(N,dtype=complex)

# Build classical L_D
L_D = np.zeros((N*N,N*N), dtype=complex)
for C in [C0, C1]:
    Cd = C.conj().T @ C
    L_D += np.kron(C.conj(), C)
    L_D += -0.5 * np.kron(I, Cd)
    L_D += -0.5 * np.kron(Cd.T, I)

print('Testing symmetric composition against exp(L_D dt/2)')
for dt in [0.6,0.2,0.1,0.04,0.02,0.01]:
    dt_half = dt/2.0
    # symmetric composition across qubits: S1(dt/4) * S0(dt/2) * S1(dt/4)
    g0_half = 1 - np.exp(-gamma0 * dt_half)
    g1_quarter = 1 - np.exp(-gamma1 * (dt_half/2.0))
    g1_half = 1 - np.exp(-gamma1 * dt_half)
    # build Kraus
    K1_q = build_kraus_full(g1_quarter)
    K0_h = build_kraus_full(g0_half)
    # build S1_quarter, S0_half
    S1_q = sum(np.kron(K.conj(), K) for K in K1_q)
    S0_h = sum(np.kron(K.conj(), K) for K in K0_h)
    S_sym = S1_q @ S0_h @ S1_q
    P_D_half = scipy.linalg.expm(L_D * dt_half)
    max_elem = np.max(np.abs(S_sym - P_D_half))
    frob = np.linalg.norm(S_sym - P_D_half, ord='fro')
    print(f'dt={dt:.5f} max_elem={max_elem:.6e} frob={frob:.6e}')

