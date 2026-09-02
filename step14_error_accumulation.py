import numpy as np
import scipy.linalg
from qutip import basis, sigmax, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import csv

# Params
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_steps = 150

# use dt from before
dt = T_final / n_steps
print('dt=', dt, 'n_steps=', n_steps)

# Build H and collapse ops
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H = H_q.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N = H.shape[0]

I = np.eye(N, dtype=complex)

def L_D_from_C(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return Ld

# classical maps
dt_half = dt/2.0
L_D_full = L_D_from_C([C0, C1])
P_D_half = scipy.linalg.expm(L_D_full * dt_half)
U = scipy.linalg.expm(-1j * H * dt)
U_super = np.kron(U.conj(), U)
S_classical = P_D_half @ U_super @ P_D_half

# circuit map
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

gamma0_half = 1 - np.exp(-gamma0 * dt_half)
gamma1_half = 1 - np.exp(-gamma1 * dt_half)
K0 = build_kraus_from_dilation(gamma0_half)
K1 = build_kraus_from_dilation(gamma1_half)
S0 = sum(np.kron(K.conj(), K) for K in K0)
S1 = sum(np.kron(K.conj(), K) for K in K1)
S_circ_half = S1 @ S0
S_circ_full = S_circ_half @ U_super @ S_circ_half

# difference operator
D = S_circ_full - S_classical

# initial state vectorize
# choose same initial state as before: |+>|0>
from qutip import basis
psi0 = tensor((basis(2,0)+basis(2,1)).unit(), basis(2,0))
rho0 = psi0.proj().full()
vec0 = rho0.reshape((N*N,), order='F')

# iterate and record norms
rows = []
vec_class = vec0.copy()
vec_circ = vec0.copy()
print('Stepping and recording differences up to n_steps...')
for k in range(1, n_steps+1):
    vec_class = S_classical @ vec_class
    vec_circ = S_circ_full @ vec_circ
    rho_c = vec_circ.reshape((N,N), order='F')
    rho_cl = vec_class.reshape((N,N), order='F')
    diff = rho_c - rho_cl
    max_elem = np.max(np.abs(diff))
    frob = np.linalg.norm(diff, ord='fro')
    # fidelity between density matrices
    try:
        sqrt_rho = scipy.linalg.sqrtm(rho_c)
        fid = np.real(np.trace(scipy.linalg.sqrtm(sqrt_rho @ rho_cl @ sqrt_rho)))**2
    except Exception:
        fid = np.nan
    rows.append([k, max_elem, frob, fid])
    if k % 25 == 0 or k==1 or k==n_steps:
        print(f'k={k:3d} max={max_elem:.6e} frob={frob:.6e} fid={fid:.6e}')

# save CSV
with open('/home/marvan-mahamood/qsim/accumulation_vs_steps.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step','max_elem_diff','frob_diff','fidelity'])
    writer.writerows(rows)
print('\nSaved /home/marvan-mahamood/qsim/accumulation_vs_steps.csv')
