import numpy as np
import scipy.linalg
from qutip import sigmax, sigmaz, qeye, tensor, Qobj, basis
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# params
gamma0, gamma1 = 0.15, 0.10

dt = 0.04

# build K full function
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

# compute
dt_half = dt/2
g0 = 1 - np.exp(-gamma0 * dt_half)
g1 = 1 - np.exp(-gamma1 * dt_half)
K0 = build_kraus_full(g0)
K1 = build_kraus_full(g1)
print('K0[0] norm, max, min', np.linalg.norm(K0[0]), np.max(np.abs(K0[0])), np.min(np.abs(K0[0])))
print('K0[1] norm, max, min', np.linalg.norm(K0[1]), np.max(np.abs(K0[1])), np.min(np.abs(K0[1])))
print('K1[0] norm', np.linalg.norm(K1[0]))

# apply to rho = |0><0| (two-qubit)
rho = np.zeros((4,4), complex)
rho[0,0]=1.0
out = np.zeros_like(rho)
for K in K0:
    out += K @ rho @ K.conj().T
print('after K0 trace', np.trace(out), 'max abs elem', np.max(np.abs(out)))
# then K1
out2 = np.zeros_like(out)
for K in K1:
    out2 += K @ out @ K.conj().T
print('after K1 trace', np.trace(out2), 'max abs elem', np.max(np.abs(out2)))

# build P_D_half via L
from qutip import sigmax, sigmaz, qeye, tensor, Qobj
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2).full()
C1 = np.sqrt(gamma1) * tensor(I2, sm).full()
N=4; I=np.eye(N,dtype=complex)
Ld = np.zeros((N*N,N*N),dtype=complex)
for C in [C0,C1]:
    Cd = C.conj().T @ C
    Ld += np.kron(C.conj(), C)
    Ld += -0.5 * np.kron(I, Cd)
    Ld += -0.5 * np.kron(Cd.T, I)
P = scipy.linalg.expm(Ld * dt_half)
# apply P to vec(rho)
vec = rho.reshape((N*N,), order='F')
outP = P @ vec
outPmat = outP.reshape((N,N), order='F')
print('P applied trace', np.trace(outPmat), 'max abs elem', np.max(np.abs(outPmat)))

# compare elementwise
print('max |out2 - outP|', np.max(np.abs(out2 - outPmat)))
