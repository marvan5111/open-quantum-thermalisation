import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import csv
from qutip import basis, sigmax, sigmaz, qeye, tensor, mesolve, Qobj

# Parameters
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0

# Build Hamiltonian and collapse operators
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_qutip = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H = H_qutip.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
C_list = [C0, C1]

N = H.shape[0]
I = np.eye(N, dtype=complex)

# Build superoperators
# L = L_H + L_D where L_H = -i (I ⊗ H - H.T ⊗ I)
# and L_D = sum_k (kron(C_k.conj(), C_k) - 0.5 kron(I, C_k^† C_k) - 0.5 kron(C_k^T C_k^*, I))

def build_L_H(H):
    return -1j * (np.kron(I, H) - np.kron(H.T, I))

def build_L_D(C_list):
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in C_list:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return Ld

L_H = build_L_H(H)
L_D = build_L_D(C_list)
L = L_H + L_D

# initial state
psi0 = tensor(basis(2,0), basis(2,0))
rho0 = psi0 * psi0.dag()
rho0_np = rho0.full()
vec0 = rho0_np.reshape(N*N, order='F')

# read trotter errors to overlay if present
trotter_csv = '/home/marvan-mahamood/qsim/trotter_error_sweep.csv'
try:
    trotter_data = []
    with open(trotter_csv, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            n = int(row[0]); dt = float(row[1]); max_elem = float(row[2]); final_elem = float(row[3]); max_frob = float(row[4])
            trotter_data.append((n, dt, max_elem, final_elem, max_frob))
    trotter_data.sort()
except Exception:
    trotter_data = []

# choose n_values from trotter if available else default
if trotter_data:
    n_values = [row[0] for row in trotter_data]
else:
    n_values = [10,20,50,100,150,300,600]

results = []
for n in n_values:
    dt = T_final / n
    # exact discrete propagator
    P_exact = scipy.linalg.expm(L * dt)
    # Strang-split superoperator S(dt) = exp(L_D dt/2) exp(L_H dt) exp(L_D dt/2)
    S = scipy.linalg.expm(L_D * (dt/2.0)) @ scipy.linalg.expm(L_H * dt) @ scipy.linalg.expm(L_D * (dt/2.0))
    # apply repeatedly
    v_exact = vec0.copy()
    v_S = vec0.copy()
    max_diff = 0.0
    max_frob = 0.0
    for _ in range(n):
        v_exact = P_exact @ v_exact
        v_S = S @ v_S
        rho_exact = v_exact.reshape((N,N), order='F')
        rho_S = v_S.reshape((N,N), order='F')
        diff = np.max(np.abs(rho_S - rho_exact))
        frob = np.linalg.norm(rho_S - rho_exact, ord='fro')
        if diff > max_diff:
            max_diff = diff
        if frob > max_frob:
            max_frob = frob
    final_diff = np.max(np.abs(rho_S - rho_exact))
    results.append((n, dt, max_diff, final_diff, max_frob))
    print(f'n={n:4d} dt={dt:.5f} Strang-superop max|diff|={max_diff:.3e} final|diff|={final_diff:.3e} maxFrob={max_frob:.3e}')

# save results
csv_out = '/home/marvan-mahamood/qsim/strang_superop_vs_liouvillian.csv'
with open(csv_out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['n','dt','max_elem_diff','final_elem_diff','max_frobenius'])
    w.writerows(results)
print('\nSaved', csv_out)

# Plot comparison: superop vs trotter (if trotter present)
dts = [r[1] for r in results]
max_elem = [r[2] for r in results]
final_elem = [r[3] for r in results]
max_frob = [r[4] for r in results]

plt.figure(figsize=(8,5))
plt.loglog(dts, final_elem, marker='o', label='Strang-superop final elem diff vs exp(Ldt)')
if trotter_data:
    dt_t = [row[1] for row in trotter_data]
    final_t = [row[3] for row in trotter_data]
    plt.loglog(dt_t, final_t, marker='s', label='Circuit Trotter final elem diff vs QuTiP (from file)')
plt.gca().invert_xaxis()
plt.xlabel('dt')
plt.ylabel('final elementwise error')
plt.title('Strang superoperator splitting vs circuit Trotter')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/home/marvan-mahamood/qsim/strang_superop_vs_trotter_plot.png', dpi=150)
print('Saved plot: /home/marvan-mahamood/qsim/strang_superop_vs_trotter_plot.png')
