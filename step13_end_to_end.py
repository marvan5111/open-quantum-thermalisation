import numpy as np
import scipy.linalg
from qutip import mesolve, basis, sigmax, sigmay, sigmaz, qeye, tensor, Qobj
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Parameters (match earlier)
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_steps = 150

dt = T_final / n_steps
print('Running end-to-end with n_steps=', n_steps, 'dt=', dt)

# Build system operators
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H_q = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
H = H_q.full()
sm = Qobj(np.array([[0,0],[1,0]], dtype=complex))
C0 = np.sqrt(gamma0) * tensor(sm, I2)
C1 = np.sqrt(gamma1) * tensor(I2, sm)

# initial state: choose a nontrivial pure state for comparison (e.g., |+>|0>)
psi0 = tensor((basis(2,0)+basis(2,1)).unit(), basis(2,0))
rho0 = psi0.proj()

# QuTiP mesolve (Lindblad) to get exact final state
H_qobj = H_q
c_ops = [C0, C1]
result = mesolve(H_qobj, psi0, [T_final], c_ops=c_ops)
psi_final = result.states[-1]
rho_qutip = psi_final.full()

# Build Trotter circuit simulation using corrected embedding
# helper: build full-system Kraus set for a single-qubit amplitude damping dilation (ancilla+swap embedding)
def build_kraus_full(gamma_step):
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    qc3 = QuantumCircuit(3)
    # unconditional swap so target is adjacent to ancilla
    qc3.swap(1,2)
    qc3.cry(theta, 0, 1)
    qc3.cx(1, 0)
    qc3.swap(1,2)
    U8 = Operator(qc3).data
    # extract K_k = <k_anc| U |0_anc> mapping to 4x4 full-system Kraus
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

# precompute unitary
U = scipy.linalg.expm(-1j * H * dt)

# simulate
rho = rho0.full()
for step in range(n_steps):
    # half-step dissipator
    gamma0_half = 1 - np.exp(-gamma0 * (dt/2.0))
    gamma1_half = 1 - np.exp(-gamma1 * (dt/2.0))
    K0 = build_kraus_full(gamma0_half)
    # apply K0 to rho: rho = sum_k K0_k rho K0_k^
    rho_temp = np.zeros_like(rho)
    for K in K0:
        rho_temp += K @ rho @ K.conj().T
    rho = rho_temp
    K1 = build_kraus_full(gamma1_half)
    rho_temp = np.zeros_like(rho)
    for K in K1:
        rho_temp += K @ rho @ K.conj().T
    rho = rho_temp

    # unitary step
    rho = U @ rho @ U.conj().T

    # second half-step
    rho_temp = np.zeros_like(rho)
    for K in K0:
        rho_temp += K @ rho @ K.conj().T
    rho = rho_temp
    rho_temp = np.zeros_like(rho)
    for K in K1:
        rho_temp += K @ rho @ K.conj().T
    rho = rho_temp

# compare final states
rho_trot = rho
max_elem = np.max(np.abs(rho_trot - rho_qutip))
frob = np.linalg.norm(rho_trot - rho_qutip, ord='fro')
rel_frob = frob / np.linalg.norm(rho_qutip, ord='fro')
trace_diff = np.abs(np.trace(rho_trot) - np.trace(rho_qutip))

# fidelity
sqrt_rho = scipy.linalg.sqrtm(rho_trot)
try:
    fid_mat = scipy.linalg.sqrtm(sqrt_rho @ rho_qutip @ sqrt_rho)
    fidelity = np.real_if_close(np.trace(fid_mat))**2
except Exception:
    # fallback to trace(sqrt(sqrt(rho) sigma sqrt(rho)))^2 computed numerically
    fidelity = np.real(np.trace(scipy.linalg.sqrtm(scipy.linalg.sqrtm(rho_trot) @ rho_qutip @ scipy.linalg.sqrtm(rho_trot))))

print('\nEnd-to-end comparison (n_steps={}):'.format(n_steps))
print('max elementwise diff:', max_elem)
print('Frobenius norm diff:', frob)
print('relative Frobenius:', rel_frob)
print('trace difference:', trace_diff)
print('state fidelity (approx):', fidelity)

# Save outputs
np.save('/home/marvan-mahamood/qsim/rho_trot_end2end.npy', rho_trot)
np.save('/home/marvan-mahamood/qsim/rho_qutip_end2end.npy', rho_qutip)
print('\nSaved rho_trot_end2end.npy and rho_qutip_end2end.npy')
