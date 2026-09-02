import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis
import csv

# params
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_list = [10, 30, 60, 150, 300, 600]

# system operators
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H = H_q.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N = H.shape[0]
I = np.eye(N, dtype=complex)

# function to build P_D_half
def build_P_D_half(dt):
    dt_half = dt/2.0
    Ld = np.zeros((N*N, N*N), dtype=complex)
    for C in [C0, C1]:
        Cd = C.conj().T @ C
        Ld += np.kron(C.conj(), C)
        Ld += -0.5 * np.kron(I, Cd)
        Ld += -0.5 * np.kron(Cd.T, I)
    return scipy.linalg.expm(Ld * dt_half)

# extract Kraus from superoperator P (column-major vec)
def kraus_from_superop(P, tol=1e-12):
    # build Choi: J = sum_{ij} E_ij x S(E_ij)
    d = int(np.sqrt(P.shape[0]))
    J = np.zeros((d*d, d*d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    J[i*d+k, j*d+l] = P[i + j*d, k + l*d]
    # eigendecompose
    vals, vecs = np.linalg.eigh(J)
    kraus = []
    for idx, val in enumerate(vals):
        if val > tol:
            v = vecs[:, idx]
            K = np.sqrt(val) * v.reshape((d, d), order='F')
            kraus.append(K)
    return kraus

# build Stinespring isometry V from Kraus: V = sum_k K_k x |k> (maps H_sys -> H_sys x H_anc)
# then extend V to full unitary on H_sys x H_anc by completing basis

def stinespring_unitary_from_kraus(kraus):
    d = kraus[0].shape[0]
    m = len(kraus)
    V = np.zeros((d*m, d), dtype=complex)
    for k, K in enumerate(kraus):
        V[k*d:(k+1)*d, :] = K
    # Perform QR on V to get orthonormal columns
    Qv, R = np.linalg.qr(V)
    Q = np.zeros((d*m, d*m), dtype=complex)
    Q[:, :d] = Qv[:, :d]
    col = d
    # fill remaining columns by Gram-Schmidt using standard basis
    for i in range(d*m):
        if col >= d*m:
            break
        candidate = np.zeros((d*m,), dtype=complex)
        candidate[i] = 1.0
        for j in range(col):
            proj = np.vdot(Q[:, j], candidate)
            candidate -= proj * Q[:, j]
        norm = np.linalg.norm(candidate)
        if norm > 1e-12:
            Q[:, col] = candidate / norm
            col += 1
    while col < d*m:
        candidate = np.random.randn(d*m) + 1j*np.random.randn(d*m)
        for j in range(col):
            proj = np.vdot(Q[:, j], candidate)
            candidate -= proj * Q[:, j]
        norm = np.linalg.norm(candidate)
        if norm > 1e-12:
            Q[:, col] = candidate / norm
            col += 1
    U = Q
    return U

results = []
for n_steps in n_list:
    dt = T_final / n_steps
    P_D_half = build_P_D_half(dt)
    kraus = kraus_from_superop(P_D_half, tol=1e-14)
    m = len(kraus)
    print('n_steps', n_steps, 'dt', dt, 'kraus_count', m)
    U = stinespring_unitary_from_kraus(kraus)
    d = N
    # build superoperator from U by action on basis
    E = []
    for i in range(d):
        for j in range(d):
            M = np.zeros((d,d), dtype=complex)
            M[i,j] = 1.0
            E.append(M)
    Scols = []
    for M in E:
        rho_joint = np.zeros((d*m, d*m), dtype=complex)
        rho_joint[0:d, 0:d] = M
        out = U @ rho_joint @ U.conj().T
        rho_out = np.zeros((d,d), dtype=complex)
        for k in range(m):
            rho_out += out[k*d:(k+1)*d, k*d:(k+1)*d]
        Scols.append(rho_out.reshape((d*d,), order='F'))
    S_from_U = np.column_stack(Scols)
    maxdiff = np.max(np.abs(S_from_U - P_D_half))
    frob = np.linalg.norm(S_from_U - P_D_half, ord='fro')
    print('  stinespring map maxdiff', maxdiff, 'frob', frob)
    # Compose and compare final
    U_unitary = scipy.linalg.expm(-1j * H * dt)
    U_super = np.kron(U_unitary.conj(), U_unitary)
    S_circ_full = S_from_U @ U_super @ S_from_U
    S_classical = P_D_half @ U_super @ P_D_half
    d2 = d*d
    psi0 = (basis(2,0)+basis(2,1)).unit()
    psi_full = tensor(psi0, basis(2,0))
    rho0 = psi_full.proj().full()
    vec0 = rho0.reshape((d2,), order='F')
    S_circ_n = np.linalg.matrix_power(S_circ_full, n_steps)
    S_class_n = np.linalg.matrix_power(S_classical, n_steps)
    rho_circ_final = (S_circ_n @ vec0).reshape((d,d), order='F')
    rho_class_final = (S_class_n @ vec0).reshape((d,d), order='F')
    max_final = np.max(np.abs(rho_circ_final - rho_class_final))
    results.append([n_steps, dt, m, maxdiff, frob, max_final])

with open('/home/marvan-mahamood/qsim/stinespring_exact_results.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n_steps','dt','ancilla_dim','maxdiff_map','frob_map','max_final_diff'])
    writer.writerows(results)
print('\nSaved /home/marvan-mahamood/qsim/stinespring_exact_results.csv')
