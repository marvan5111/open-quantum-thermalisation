import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis

# params
gamma0, gamma1 = 0.15, 0.10

# build P_D_half for dt=0.04 as test
dt = 0.04

sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N = 4
I = np.eye(N, dtype=complex)

Ld = np.zeros((N*N, N*N), dtype=complex)
for C in [C0, C1]:
    Cd = C.conj().T @ C
    Ld += np.kron(C.conj(), C)
    Ld += -0.5 * np.kron(I, Cd)
    Ld += -0.5 * np.kron(Cd.T, I)
P = scipy.linalg.expm(Ld * (dt/2.0))

# extract kraus from P (as before)
def kraus_from_superop(P, tol=1e-14):
    d = int(np.sqrt(P.shape[0]))
    J = np.zeros((d*d, d*d), dtype=complex)
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    J[i*d+k, j*d+l] = P[i + j*d, k + l*d]
    vals, vecs = np.linalg.eigh(J)
    kraus = []
    for idx, val in enumerate(vals):
        if val > tol:
            v = vecs[:, idx]
            K = np.sqrt(val) * v.reshape((d, d), order='F')
            kraus.append(K)
    return kraus

kraus = kraus_from_superop(P)
print('extracted', len(kraus), 'K')

# transformations to try
transforms = {
    'K': lambda K: K,
    'K_T': lambda K: K.T,
    'K_H': lambda K: K.conj().T,
    'K_conj': lambda K: K.conj(),
    'K_swapaxes': lambda K: K.reshape(K.shape, order='C').T, # same as T
}

placements = ['rows_stack','cols_stack']

best = None

for tname, tf in transforms.items():
    Ks = [tf(K) for K in kraus]
    for placement in placements:
        # build V of shape (d*m, d) for rows_stack or (d, d*m) for cols_stack
        d = N; m = len(Ks)
        if placement == 'rows_stack':
            V = np.zeros((d*m, d), dtype=complex)
            for k,K in enumerate(Ks):
                V[k*d:(k+1)*d, :] = K
            # extend to unitary (QR) as before
            Qv, R = np.linalg.qr(V)
            U = np.zeros((d*m, d*m), dtype=complex)
            U[:, :d] = Qv[:, :d]
            col = d
            for i in range(d*m):
                if col>=d*m: break
                candidate = np.zeros((d*m,), dtype=complex); candidate[i]=1.0
                for j in range(col):
                    proj = np.vdot(U[:, j], candidate); candidate -= proj * U[:, j]
                norm = np.linalg.norm(candidate)
                if norm>1e-12:
                    U[:, col] = candidate/norm; col+=1
            while col<d*m:
                cand = np.random.randn(d*m)+1j*np.random.randn(d*m)
                for j in range(col): cand -= np.vdot(U[:, j], cand)*U[:, j]
                norm = np.linalg.norm(cand)
                if norm>1e-12:
                    U[:, col] = cand/norm; col+=1
            # apply U to |0><0| x M basis as earlier to build S
            E = []
            for i in range(d):
                for j in range(d):
                    M = np.zeros((d,d), dtype=complex); M[i,j]=1.0; E.append(M)
            Scols=[]
            for M in E:
                rho_joint = np.zeros((d*m,d*m), dtype=complex)
                # ancilla |0><0| placed in block 0; joint indexing ancilla-major
                rho_joint[0:d,0:d] = M
                out = U @ rho_joint @ U.conj().T
                rho_out = np.zeros((d,d), dtype=complex)
                for k in range(m): rho_out += out[k*d:(k+1)*d, k*d:(k+1)*d]
                Scols.append(rho_out.reshape((d*d,), order='F'))
            S = np.column_stack(Scols)
        else: # cols_stack: maybe ancilla is last index, joint ordering system-major
            V = np.zeros((d*m, d), dtype=complex)
            for k,K in enumerate(Ks):
                # place K into columns blocks
                V[:, k*d:(k+1)*d] = np.kron(np.eye(m)[k:k+1].T, K)[:,:d]
                # this is awkward, skip
            # alternative: build U differently
            continue
        maxdiff = np.max(np.abs(S - P))
        print('transform',tname,'placement',placement,'maxdiff',maxdiff)
        if best is None or maxdiff < best[0]:
            best = (maxdiff, tname, placement)

print('\nBest mapping:', best)
