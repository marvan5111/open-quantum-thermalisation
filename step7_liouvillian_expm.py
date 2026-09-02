import numpy as np
import scipy.linalg
from qutip import basis, sigmax, sigmaz, qeye, tensor, mesolve, Qobj

# Parameters (must match previous runs)
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0

# Build Hamiltonian and collapse operators (QuTiP ordering: q0, q1)
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_qutip = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H_np = H_qutip.full()

# lowering operator
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
C_ops = [C0, C1]

# initial state |00><00|
psi0 = tensor(basis(2,0), basis(2,0))
rho0 = psi0 * psi0.dag()
rho0_np = rho0.full()

N = H_np.shape[0]

# Helper: build Liouvillian superoperator L of size N^2 x N^2 acting on vec(rho) (column stacking)
def build_liouvillian(H, C_list):
    I = np.eye(N, dtype=complex)
    # unitary part: -i[H, rho] -> -i (I ⊗ H - H^T ⊗ I)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    # dissipator
    for C in C_list:
        C = np.array(C, dtype=complex)
        C_dagC = C.conj().T @ C
        # term: C * rho * C^\n        L += np.kron(C.conj(), C)
        # subtract 1/2 {C^\dagger C, rho}
        L += -0.5 * np.kron(I, C_dagC)
        L += -0.5 * np.kron(C_dagC.T, I)
    return L

# Vectorize helper (column-stacking)
def vec(rho):
    return rho.reshape(N*N, order='F')

def mat(v):
    return v.reshape((N, N), order='F')

# Build Liouvillian
L = build_liouvillian(H_np, C_ops)

# Sweep n_values and compare Liouvillian-expm discrete evolution against QuTiP mesolve
n_values = [10, 20, 50, 100, 150, 300, 600]
results = []
for n in n_values:
    dt = T_final / n
    P = scipy.linalg.expm(L * dt)
    # generate snapshots by repeatedly applying P to vec(rho0)
    vec_rho = vec(rho0_np)
    snapshots = [mat(vec_rho).copy()]
    for _ in range(n):
        vec_rho = P @ vec_rho
        snapshots.append(mat(vec_rho).copy())
    snapshots = np.array(snapshots)

    # Compute QuTiP exact on same grid
    tlist = np.linspace(0.0, T_final, n+1)
    result = mesolve(H_qutip, rho0, tlist, [np.sqrt(gamma0)*tensor(sm, I2), np.sqrt(gamma1)*tensor(I2, sm)])
    rho_exact = np.array([state.full() for state in result.states], dtype=complex)

    # Compare snapshots
    max_abs_diff = np.max(np.abs(snapshots - rho_exact))
    final_abs_diff = np.max(np.abs(snapshots[-1] - rho_exact[-1]))
    max_frob = np.max([np.linalg.norm(snapshots[i] - rho_exact[i], ord='fro') for i in range(len(tlist))])

    print(f"n={n:3d} dt={dt:.5f} Liouvillian vs QuTiP: max|elem diff|={max_abs_diff:.3e} final|elem diff|={final_abs_diff:.3e} maxFrob={max_frob:.3e}")
    results.append((n, dt, max_abs_diff, final_abs_diff, max_frob))

# Save results
import csv
with open('/home/marvan-mahamood/qsim/liouvillian_vs_qutip.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n', 'dt', 'max_elem_diff', 'final_elem_diff', 'max_frobenius'])
    w.writerows(results)

# Also compare Trotter (saved rho_trotter_150.npy) vs Liouvillian for n=150 if available
import os
if os.path.exists('/home/marvan-mahamood/qsim/rho_trotter_150.npy'):
    rho_trotter = np.load('/home/marvan-mahamood/qsim/rho_trotter_150.npy')
    # build Liouvillian snapshots for n=150 (we already have it in results)
    dt = T_final / 150
    P = scipy.linalg.expm(L * dt)
    vec_rho = vec(rho0_np)
    snapshots_liou = [mat(vec_rho).copy()]
    for _ in range(150):
        vec_rho = P @ vec_rho
        snapshots_liou.append(mat(vec_rho).copy())
    snapshots_liou = np.array(snapshots_liou)
    # compare to rho_trotter
    max_abs = np.max(np.abs(rho_trotter - snapshots_liou))
    final_abs = np.max(np.abs(rho_trotter[-1] - snapshots_liou[-1]))
    max_frob = np.max([np.linalg.norm(rho_trotter[i] - snapshots_liou[i], ord='fro') for i in range(len(snapshots_liou))])
    print('\nTrotter (n=150) vs Liouvillian-expm (n=150):')
    print(f"max|elem diff|={max_abs:.3e} final|elem diff|={final_abs:.3e} maxFrob={max_frob:.3e}")
    with open('/home/marvan-mahamood/qsim/trotter_vs_liouvillian_150.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerow(['max_elem_diff', max_abs])
        w.writerow(['final_elem_diff', final_abs])
        w.writerow(['max_frobenius', max_frob])

print('\nSaved liouvillian_vs_qutip.csv and trotter_vs_liouvillian_150.csv (if trotter present)')
