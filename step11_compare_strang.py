import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Params (same as prior)
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_steps = 150

dt = T_final / n_steps
print('Using n_steps=', n_steps, 'dt=', dt)

# Build H and collapse ops
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H = H_q.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N = H.shape[0]
I = np.eye(N, dtype=complex)

# Dissipator superoperator builder
def L_D_from_C(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return Ld

L_D_full = L_D_from_C([C0, C1])

# exact half-step superoperator via exp(L_D * dt/2)
dt_half = dt/2.0
P_D_half = scipy.linalg.expm(L_D_full * dt_half)

# Build circuit-derived Kraus for each qubit half-step using unconditional swap-based embedding
from math import asin, sqrt

def build_kraus_from_dilation(qubit_idx, gamma_step):
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    qc3 = QuantumCircuit(3)
    # always swap system qubits so target sits at index 1 adjacent to ancilla at 0
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

gamma0_half = 1 - np.exp(-gamma0 * dt_half)
gamma1_half = 1 - np.exp(-gamma1 * dt_half)
K0 = build_kraus_from_dilation(0, gamma0_half)
K1 = build_kraus_from_dilation(1, gamma1_half)

# superops
S0 = sum(np.kron(K.conj(), K) for K in K0)
S1 = sum(np.kron(K.conj(), K) for K in K1)
S_circ_half = S1 @ S0

# exact unitary and its superoperator
U = scipy.linalg.expm(-1j * H * dt)
U_super = np.kron(U.conj(), U)

# Compose full Strang steps
S_circ_full = S_circ_half @ U_super @ S_circ_half
S_classical = P_D_half @ U_super @ P_D_half

# Compare
def compare(A, B, name='A vs B'):
    max_elem = np.max(np.abs(A - B))
    frob = np.linalg.norm(A - B, ord='fro')
    rel_frob = frob / np.linalg.norm(B, ord='fro')
    print(f"\n{name}: max_elem_diff={max_elem:.6e}, frob_diff={frob:.6e}, rel_frob={rel_frob:.6e}")

compare(S_circ_full, S_classical, 'Circuit Strang vs Classical Strang')

# Also report per-step differences (just to sanity-check)
compare(S_circ_half, P_D_half, 'Circuit half vs P_D_half (sanity)')

# Save results
np.save('/home/marvan-mahamood/qsim/S_circ_full.npy', S_circ_full)
np.save('/home/marvan-mahamood/qsim/S_classical.npy', S_classical)
print('\nSaved S_circ_full.npy and S_classical.npy')
