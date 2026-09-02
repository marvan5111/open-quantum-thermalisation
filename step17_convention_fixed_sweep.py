import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import csv

# params
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_list = [10, 30, 60, 150, 300, 600]

# system
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H = H_q.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N = H.shape[0]
I = np.eye(N, dtype=complex)

# basis of operators E_ij
E = []
for i in range(N):
    for j in range(N):
        M = np.zeros((N,N), dtype=complex)
        M[i,j] = 1.0
        E.append(M)

# helper: build circuit S_circ_half by action on basis (explicitly apply Kraus maps)
def build_S_circ_half_action(dt):
    dt_half = dt/2.0
    gamma0_half = 1 - np.exp(-gamma0 * dt_half)
    gamma1_half = 1 - np.exp(-gamma1 * dt_half)
    # build kraus for qubit0 and qubit1 using swap embedding (as full-system 4x4 matrices)
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
    K0 = build_kraus_full(gamma0_half)
    K1 = build_kraus_full(gamma1_half)
    # action on basis
    Scols = []
    for M in E:
        rho = M
        # apply S0 then S1
        rho_tmp = np.zeros_like(rho)
        for K in K0:
            rho_tmp += K @ rho @ K.conj().T
        rho = rho_tmp
        rho_tmp = np.zeros_like(rho)
        for K in K1:
            rho_tmp += K @ rho @ K.conj().T
        rho = rho_tmp
        # column for superop is vec(rho) with column-major
        Scols.append(rho.reshape((N*N,), order='F'))
    S = np.column_stack(Scols)
    return S

# helper: build classical P_D_half via Liouvillian
def build_P_D_half(dt):
    dt_half = dt/2.0
    # build L_D
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in [C0, C1]:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return scipy.linalg.expm(Ld * dt_half)

rows = []
print('Running convention-fixed sweep...')
for n_steps in n_list:
    dt = T_final / n_steps
    P_D_half = build_P_D_half(dt)
    S_circ_half = build_S_circ_half_action(dt)
    # per-step metrics
    diff_half = S_circ_half - P_D_half
    max_half = np.max(np.abs(diff_half))
    frob_half = np.linalg.norm(diff_half, ord='fro')
    rel_half = frob_half / np.linalg.norm(P_D_half, ord='fro')

    # full Strang
    U = scipy.linalg.expm(-1j * H * dt)
    U_super = np.kron(U.conj(), U)
    S_classical = P_D_half @ U_super @ P_D_half
    S_circ_full = S_circ_half @ U_super @ S_circ_half

    # final states via matrix power
    vec0 = E[0].copy() # Not correct initial state; instead use |+>|0>
    # build correct initial vec
    psi0 = (basis(2,0)+basis(2,1)).unit()
    psi_full = tensor(psi0, basis(2,0))
    rho0 = psi_full.proj().full()
    vec0 = rho0.reshape((N*N,), order='F')

    S_classical_n = np.linalg.matrix_power(S_classical, n_steps)
    S_circ_n = np.linalg.matrix_power(S_circ_full, n_steps)
    rho_class_final = (S_classical_n @ vec0).reshape((N,N), order='F')
    rho_circ_final = (S_circ_n @ vec0).reshape((N,N), order='F')

    # comparisons
    max_circ_vs_class = np.max(np.abs(rho_circ_final - rho_class_final))
    max_circ_vs_qutip = np.max(np.abs(rho_circ_final - rho_class_final))
    print(f'n={n_steps:4d} dt={dt:.5f} per-step max={max_half:.6e} final circ-vs-class max={max_circ_vs_class:.6e}')

    rows.append([n_steps, dt, max_half, frob_half, rel_half, max_circ_vs_class])

with open('/home/marvan-mahamood/qsim/convention_fixed_sweep.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n_steps','dt','max_half','frob_half','rel_half','max_circ_vs_class'])
    writer.writerows(rows)

print('\nSaved /home/marvan-mahamood/qsim/convention_fixed_sweep.csv')
