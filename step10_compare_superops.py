import numpy as np
import scipy.linalg
from qutip import basis, sigmax, sigmaz, qeye, tensor, Qobj

# Params
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

# Build dissipator superoperators
def L_D_from_C(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return Ld

L_D_full = L_D_from_C([C0, C1])
L_D0 = L_D_from_C([C0])
L_D1 = L_D_from_C([C1])

# exact half-step superoperator via exp(L_D * dt/2)
dt_half = dt/2.0
P_D_half = scipy.linalg.expm(L_D_full * dt_half)
P_D0_half = scipy.linalg.expm(L_D0 * dt_half)
P_D1_half = scipy.linalg.expm(L_D1 * dt_half)

# Build circuit-derived Kraus for each qubit half-step using swap-based embedding
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# function to get full-system kraus operators from dilation unitary (swap-based)
def build_kraus_from_dilation(qubit_idx, gamma_step):
    # build 2-qubit dilation unitary U2 on (ancilla,target)
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    qc3 = QuantumCircuit(3)
    # always swap system qubits 1 and 2 so the target sits at index 1 next to ancilla at 0
    qc3.swap(1, 2)
    qc3.cry(theta, 0, 1)
    qc3.cx(1, 0)
    qc3.swap(1, 2)
    U8 = Operator(qc3).data
    # extract Kraus K_k = <k_anc| U |0_anc>
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

# compute gamma_step half for each qubit
gamma0_half = 1 - np.exp(-gamma0 * dt_half)
gamma1_half = 1 - np.exp(-gamma1 * dt_half)
K0 = build_kraus_from_dilation(0, gamma0_half)
K1 = build_kraus_from_dilation(1, gamma1_half)

# build superoperators S0 and S1 for K-lists: S = sum_k kron(K.conj(), K)
def superop_from_kraus(K_list):
    S = np.zeros((N*N, N*N), dtype=complex)
    for K in K_list:
        S += np.kron(K.conj(), K)
    return S

S0 = superop_from_kraus(K0)
S1 = superop_from_kraus(K1)
# circuit half-step full superoperator: apply S0 then S1 (that's how code did it)
S_circ_half = S1 @ S0

# Compare to P_D_half
max_elem_diff = np.max(np.abs(S_circ_half - P_D_half))
frob_norm = np.linalg.norm(S_circ_half - P_D_half, ord='fro')
rel_frob = frob_norm / np.linalg.norm(P_D_half, ord='fro')

print('\nComparison: circuit half-step superop vs exp(L_D * dt/2)')
print('max elementwise diff:', max_elem_diff)
print('Frobenius norm of diff:', frob_norm)
print('Relative Frobenius:', rel_frob)

# Also per-qubit compare
max0 = np.max(np.abs(S0 - P_D0_half))
max1 = np.max(np.abs(S1 - P_D1_half))
print('\nPer-qubit comparisons:')
print('max elementwise diff S0 vs exp(L_D0 dt/2):', max0)
print('max elementwise diff S1 vs exp(L_D1 dt/2):', max1)

# Save matrices for inspection
np.save('/home/marvan-mahamood/qsim/S_circ_half.npy', S_circ_half)
np.save('/home/marvan-mahamood/qsim/P_D_half.npy', P_D_half)
print('\nSaved S_circ_half.npy and P_D_half.npy')
