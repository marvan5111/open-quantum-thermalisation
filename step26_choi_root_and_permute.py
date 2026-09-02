import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis

# params
gamma0,gamma1=0.15,0.10
dt=0.04
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N=4; I=np.eye(N,dtype=complex)
Ld=np.zeros((N*N,N*N),dtype=complex)
for C in [C0,C1]:
    Cd=C.conj().T@C
    Ld+=np.kron(C.conj(),C)
    Ld+=-0.5*np.kron(I,Cd)
    Ld+=-0.5*np.kron(Cd.T,I)
P=scipy.linalg.expm(Ld*(dt/2.0))
# Build Choi J with our convention
d=int(np.sqrt(P.shape[0]))
J=np.zeros((d*d,d*d),dtype=complex)
for i in range(d):
    for j in range(d):
        for k in range(d):
            for l in range(d):
                J[i*d+k,j*d+l]=P[i + j*d, k + l*d]
# take sqrt of J: J = V V^dag
vals, vecs = np.linalg.eigh(J)
# build V_ch = sum sqrt(val) |v>
Vch = np.zeros_like(J)
for idx, val in enumerate(vals):
    if val>1e-16:
        Vch[:, idx] = np.sqrt(val) * vecs[:, idx]
# Now interpret Vch as mapping from ancilla-system to ancilla-system; try reshape permutations
# Vch shape (d*d, d*d). We want an isometry of shape (d*m, d) where m is ancilla dim = number of nonzero eigs
nonzero = [i for i,v in enumerate(vals) if v>1e-16]
m = len(nonzero)
print('nonzero count m=', m)
# take columns corresponding to nonzero eigenvectors
Vcols = [Vch[:, idx] for idx in nonzero]
# each column is length d*d; try reshape into (m,d) x d?? We'll test permutations

def try_reshape_and_test(Vcols, permute_axes):
    # permute_axes is a tuple telling how to reorder indices when reshaping (i1,i2)->(j1,j2)
    d2 = d*d
    # build V matrix of shape (d*m, d) by stacking K matrices
    Klist = []
    for col in Vcols:
        M = col.reshape((d,d), order='F')
        Klist.append(M)
    # enforce completeness
    Mmat = sum(K.conj().T@K for K in Klist)
    inv_sqrt = scipy.linalg.inv(scipy.linalg.sqrtm(Mmat))
    Kp = [K @ inv_sqrt for K in Klist]
    # build S_from_Kp
    S_from_Kp = sum(np.kron(K.conj(),K) for K in Kp)
    err = np.max(np.abs(S_from_Kp - P))
    return err, Kp

err, Kp = try_reshape_and_test(Vcols, None)
print('initial err from Choi-root Kraus reshape:', err)
# If that fails, try permuting vector before reshape
best = (1e9, None)
from itertools import permutations
for perm in permutations([0,1]):
    # try permutations of (i,j) axes when reshaping -> but for square it's same; still try transposes
    Klist_try = [col.reshape((d,d), order='F').T if perm==(1,0) else col.reshape((d,d), order='F') for col in Vcols]
    Mmat=sum(K.conj().T@K for K in Klist_try)
    inv_sqrt=scipy.linalg.inv(scipy.linalg.sqrtm(Mmat))
    Kp_try=[K@inv_sqrt for K in Klist_try]
    S_from_Kp=sum(np.kron(K.conj(),K) for K in Kp_try)
    err_try = np.max(np.abs(S_from_Kp - P))
    if err_try < best[0]: best=(err_try, perm)
print('best perm result:', best)

# If still bad, print a few entries to inspect
print('\nP (sample entries):')
for idx in [0,1,2,3,4]: print(P.flat[idx])
print('\nS_from_Kp (sample entries):')
S_from_Kp = sum(np.kron(K.conj(),K) for K in Kp)
for idx in [0,1,2,3,4]: print(S_from_Kp.flat[idx])
