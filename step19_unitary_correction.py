import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# params
gamma0, gamma1 = 0.15, 0.10

# system operators
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = 1.0 * tensor(sx, sx) + 0.7 * tensor(sz, I2) + 1.1 * tensor(I2, sz)
H = H_q.full()
N = H.shape[0]
I = np.eye(N, dtype=complex)

# build L_D
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
Ld = np.zeros((N*N, N*N), dtype=complex)
for C in [C0, C1]:
    Cd = C.conj().T @ C
    Ld += np.kron(C.conj(), C)
    Ld += -0.5 * np.kron(I, Cd)
    Ld += -0.5 * np.kron(Cd.T, I)

# helpers
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

# pick small dt to estimate generator
dt = 0.01
dt_half = dt/2.0
gamma0_half = 1 - np.exp(-gamma0 * dt_half)
gamma1_half = 1 - np.exp(-gamma1 * dt_half)
K0 = build_kraus_full(gamma0_half)
K1 = build_kraus_full(gamma1_half)
S0 = sum(np.kron(K.conj(), K) for K in K0)
S1 = sum(np.kron(K.conj(), K) for K in K1)
S_circ_half = S1 @ S0

# estimate A_circ = (S - I)/dt_half
A_circ = (S_circ_half - np.eye(N*N, dtype=complex)) / dt_half
A_target = Ld
B = A_circ - A_target
print('norm of B (generator mismatch) frob:', np.linalg.norm(B, ord='fro'))

# We want to find small Hermitian h (4x4) so that its superoperator K = -1j*(I kron h - h.T kron I) approximates anti-Hermitian part of B
# Build target anti-Hermitian part
B_ah = 0.5 * (B - B.conj().T)
print('norm anti-Hermitian part:', np.linalg.norm(B_ah, ord='fro'))

# Parameterize h as 16 real parameters (Hermitian 4x4 has 16 real dim) and solve least-squares
# vec(K) = M * vec(h_params)
# build basis for Hermitian matrices
H_basis = []
for i in range(4):
    for j in range(4):
        M = np.zeros((4,4), dtype=complex)
        if i==j:
            M[i,j] = 1.0
        elif i<j:
            M[i,j] = 1.0/2
            M[j,i] = 1.0/2
        else:
            M[i,j] = -0.5j
            M[j,i] = 0.5j
        H_basis.append(M)
# actually above basis is messy; use Hermitian basis from standard construction
H_basis = []
for i in range(4):
    M = np.zeros((4,4), dtype=complex)
    M[i,i]=1.0
    H_basis.append(M)
for i in range(4):
    for j in range(i+1,4):
        M = np.zeros((4,4), dtype=complex)
        M[i,j]=1.0; M[j,i]=1.0
        H_basis.append(M)
        M2 = np.zeros((4,4), dtype=complex)
        M2[i,j]= -1j; M2[j,i]= 1j
        H_basis.append(M2)
# now 4 + 6*2 = 16 basis elements
assert len(H_basis)==16

# build matrix mapping h_params to superoperator
# Mmat maps h_params (16) to vec(Kmat) (256)
Mmat = np.zeros((N*N*N*N, 16), dtype=complex)
for idx, Hb in enumerate(H_basis):
    Kmat = -1j*(np.kron(np.eye(4), Hb) - np.kron(Hb.T, np.eye(4)))
    Mmat[:, idx] = Kmat.reshape((N*N*N*N,), order='F')

# target vec (flatten anti-Hermitian part)
bvec = B_ah.reshape((N*N*N*N,), order='F')
# solve least squares for complex Mmat * x = bvec
# stack real/imag parts to solve real least squares
A_real = np.vstack([np.hstack([Mmat.real, -Mmat.imag]), np.hstack([Mmat.imag, Mmat.real])])
b_real = np.hstack([bvec.real, bvec.imag])
sol, *_ = np.linalg.lstsq(A_real, b_real, rcond=None)

x_complex = sol[:16] + 1j * sol[16:]
# reconstruct h
h_est = sum(x_complex[i]*H_basis[i] for i in range(16))
# make Hermitian
h_est = 0.5*(h_est + h_est.conj().T)
print('h_est hermitian? max imag part:', np.max(np.abs(h_est.imag)))

# compute norms of Hermitian and anti-Hermitian parts of B
B_h = 0.5 * (B + B.conj().T)
B_ah = 0.5 * (B - B.conj().T)
print('norm(B) frob:', np.linalg.norm(B,ord='fro'))
print('norm Hermitian part frob:', np.linalg.norm(B_h,ord='fro'))
print('norm Anti-Hermitian part frob:', np.linalg.norm(B_ah,ord='fro'))

# compute correction superoperator (unitary-targeting)
Kcorr = -1j*(np.kron(np.eye(4), h_est) - np.kron(h_est.T, np.eye(4)))
# compute corrected A
A_corr = A_circ - Kcorr
print('norm A_corr - A_target frob (should be smaller if correction worked):', np.linalg.norm(A_corr - A_target, ord='fro'))

# build unitary correction U = exp(-i h_est * dt/2)
U_corr = scipy.linalg.expm(-1j * h_est * (dt/2.0))
Ucorr_super = np.kron(U_corr.conj(), U_corr)
S_circ_half_corrected = Ucorr_super @ S_circ_half @ scipy.linalg.inv(Ucorr_super)
step_err_before = np.max(np.abs(S_circ_half - scipy.linalg.expm(Ld * dt_half)))
step_err_after = np.max(np.abs(S_circ_half_corrected - scipy.linalg.expm(Ld * dt_half)))
print('step_err_before', step_err_before, 'step_err_after', step_err_after)

# quick accumulation test with corrected half-step
U = scipy.linalg.expm(-1j * H * dt)
U_super = np.kron(U.conj(), U)
S_circ_full_corr = S_circ_half_corrected @ U_super @ S_circ_half_corrected
S_classical = scipy.linalg.expm(Ld * dt_half) @ U_super @ scipy.linalg.expm(Ld * dt_half)

# final after n steps
n_steps=150
S_circ_n = np.linalg.matrix_power(S_circ_full_corr, n_steps)
S_class_n = np.linalg.matrix_power(S_classical, n_steps)
# initial vector
psi0 = (basis(2,0)+basis(2,1)).unit()
psi_full = tensor(psi0, basis(2,0))
rho0 = psi_full.proj().full()
vec0 = rho0.reshape((N*N,), order='F')
rho_corr_final = (S_circ_n @ vec0).reshape((N,N), order='F')
rho_class_final = (S_class_n @ vec0).reshape((N,N), order='F')

print('final diff after correction max_elem:', np.max(np.abs(rho_corr_final - rho_class_final)))
