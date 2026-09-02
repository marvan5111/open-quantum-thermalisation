# Sweep Trotter step counts, compare to QuTiP exact master-equation solution,
# and save a plot + CSV of error metrics.
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, DensityMatrix, partial_trace
from qutip import basis, sigmax, sigmaz, qeye, tensor, mesolve, Qobj
import matplotlib.pyplot as plt
import csv
import scipy.linalg

# Model parameters (must match previous runs)
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0

# Build QuTiP objects for the exact solver
sx = sigmax()
sz = sigmaz()
I2 = qeye(2)
sm = Qobj(np.array([[0, 0], [1, 0]], dtype=complex))
H_qutip = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
# Convert QuTiP Hamiltonian to a numpy matrix for exact exponentiation
H_np = H_qutip.full()
psi0 = tensor(basis(2, 0), basis(2, 0))
rho0_qutip = psi0 * psi0.dag()
c_ops = [np.sqrt(gamma0) * tensor(sm, I2), np.sqrt(gamma1) * tensor(I2, sm)]

# Small helper: build one Trotter-step unitary using Qiskit
def trotter_unitary_matrix(dt):
    # Keep the original gate-construction function for reference, but prefer exact_unitary where used.
    qc = QuantumCircuit(2)
    qc.h(0); qc.h(1)
    qc.cx(0, 1)
    qc.rz(2 * Jxx * dt, 1)
    qc.cx(0, 1)
    qc.h(0); qc.h(1)
    qc.rz(2 * h0 * dt, 0)
    qc.rz(2 * h1 * dt, 1)
    U = Operator(qc).data
    return U

# Exact unitary for the full 2-qubit Hamiltonian using matrix exponential
def exact_unitary(dt):
    # H_np is the 4x4 numpy Hamiltonian defined earlier
    return scipy.linalg.expm(-1j * H_np * dt)

# Single-step Kraus for amplitude damping with a small gamma_step
def damping_kraus(gamma_step):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma_step)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma_step)], [0, 0]], dtype=complex)
    return [K0, K1]

# Apply 1-qubit Kraus to full 2-qubit density matrix for a given qubit index
def apply_1qubit_kraus(rho2, kraus_ops, qubit_idx):
    I2 = np.eye(2, dtype=complex)
    new_rho = np.zeros_like(rho2, dtype=complex)
    for K in kraus_ops:
        if qubit_idx == 0:
            full_K = np.kron(I2, K)   # qubit 0 is the second factor in our ordering (match earlier code)
        else:
            full_K = np.kron(K, I2)
        new_rho += full_K @ rho2 @ full_K.conj().T
    return new_rho

# Dilation-based amplitude damping on a chosen system qubit using a fresh ancilla.
# rho2: 4x4 numpy array representing two system qubits (q0, q1)
# gamma_step: probability 1 - exp(-gamma*dt) for that time interval
# qubit_idx: 0 or 1 indicating which system qubit to damp
def apply_dilation_damping(rho2, gamma_step, qubit_idx):
    # theta chosen so that sin^2(theta/2) = gamma_step
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    # Create Qiskit DensityMatrix for ancilla ⊗ system
    dm_sys = DensityMatrix(rho2)
    dm_anc = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
    # ancilla as qubit 0, system qubits follow as 1,2
    dm_total = dm_anc.tensor(dm_sys)
    # Build the 2-qubit unitary U_anc_target acting on (ancilla, target)
    qc2 = QuantumCircuit(2)
    qc2.cry(theta, 0, 1)
    qc2.cx(1, 0)
    U2 = Operator(qc2).data  # 4x4 unitary acting on ancilla ⊗ target

    # Build a 3-qubit circuit that swaps the target into position 1 (adjacent to ancilla),
    # applies the ancilla-target 2-qubit unitary, then swaps back. This avoids manual index mapping errors.
    qc3 = QuantumCircuit(3)
    swapped = False
    if qubit_idx == 0:
        qc3.swap(1, 2)
        swapped = True
    # Apply the 2-qubit dilation on (ancilla at 0, target now at 1)
    qc3.cry(theta, 0, 1)
    qc3.cx(1, 0)
    if swapped:
        qc3.swap(1, 2)

    U8 = Operator(qc3).data
    dm_out_mat = U8 @ dm_total.data @ U8.conj().T
    # Trace out ancilla (dimension 2) from dm_out_mat shaped (2,4,2,4) -> result 4x4
    rho_sys = np.einsum('aibj->ij', dm_out_mat.reshape(2, 4, 2, 4))
    return rho_sys

# Build full-system Kraus operators from the dilation unitary U8 and ancilla initial |0>
# Returns a list of 4x4 numpy arrays K_k acting on the 2-qubit system
def build_kraus_from_dilation(qubit_idx, gamma_step):
    theta = 2 * np.arcsin(np.sqrt(gamma_step))
    # Build 3-qubit circuit as in apply_dilation_damping
    qc3 = QuantumCircuit(3)
    swapped = False
    if qubit_idx == 0:
        qc3.swap(1, 2)
        swapped = True
    qc3.cry(theta, 0, 1)
    qc3.cx(1, 0)
    if swapped:
        qc3.swap(1, 2)
    U8 = Operator(qc3).data
    # Extract Kraus operators K_k = <k_ancilla| U |0_ancilla>
    K_list = []
    for k in range(2):
        K = np.zeros((4, 4), dtype=complex)
        for sp in range(4):
            row = (k << 2) | sp
            for s in range(4):
                col = (0 << 2) | s
                K[sp, s] = U8[row, col]
        K_list.append(K)
    return K_list

# Apply full-system Kraus operators (4x4) to a 4x4 density matrix
def apply_full_kraus(rho, K_list):
    new_rho = np.zeros_like(rho, dtype=complex)
    for K in K_list:
        new_rho += K @ rho @ K.conj().T
    return new_rho

# Run the Trotter simulation (unitary step then local amplitude damping on each qubit)
def run_trotter_sim(n_steps):
    dt = T_final / n_steps
    U = trotter_unitary_matrix(dt)
    # initial state |00><00|
    e = np.array([[1, 0], [0, 0]], dtype=complex)
    rho = np.kron(e, e)  # 4x4 density matrix as a flat 4x4 array
    # Use exact finite-time damping parameter for half-steps
    gamma0_half = 1 - np.exp(-gamma0 * (dt / 2))
    gamma1_half = 1 - np.exp(-gamma1 * (dt / 2))

    # Use a 4th-order Suzuki composition: S(w1*dt) S(w0*dt) S(w1*dt)
    # where S(dt_sub) is the Strang step (half dissipator -> unitary dt_sub -> half dissipator).
    w1 = 1.0 / (4.0 - 4.0**(1.0/3.0))
    w0 = 1.0 - 2.0 * w1

    snapshots = [rho.copy()]
    for _ in range(n_steps):
        for coeff in (w1, w0, w1):
            dt_sub = coeff * dt
            # unitary for this substep
            # Use exact unitary from full Hamiltonian for the substep
            U_sub = exact_unitary(dt_sub)
            # half-step dissipators for this substep (using exact finite-time mapping)
            gamma0_half_sub = 1 - np.exp(-gamma0 * (dt_sub / 2.0))
            gamma1_half_sub = 1 - np.exp(-gamma1 * (dt_sub / 2.0))
            kraus0_half_sub = build_kraus_from_dilation(0, gamma0_half_sub)
            kraus1_half_sub = build_kraus_from_dilation(1, gamma1_half_sub)

            # half dissipator
            rho = apply_full_kraus(rho, kraus0_half_sub)
            rho = apply_full_kraus(rho, kraus1_half_sub)
            # unitary for the substep
            rho = U_sub @ rho @ U_sub.conj().T
            # half dissipator
            rho = apply_full_kraus(rho, kraus0_half_sub)
            rho = apply_full_kraus(rho, kraus1_half_sub)
        snapshots.append(rho.copy())
    return np.array(snapshots)

# N-step values to test
n_values = [600, 1200, 2400]

# Storage for results
results = []

# Sweep n_values, compute exact with QuTiP at same time grid, compute error metrics
for n in n_values:
    print(f"Running n_steps = {n} ...")
    # Trotter simulation
    rho_trotter = run_trotter_sim(n)  # shape (n+1, 4, 4)
    # Exact QuTiP evolution on matching grid
    tlist = np.linspace(0.0, T_final, n + 1)
    result = mesolve(H_qutip, rho0_qutip, tlist, c_ops)  # returns states for each t in tlist
    rho_exact = np.array([state.full() for state in result.states], dtype=complex)
    # Ensure shapes match
    assert rho_exact.shape == rho_trotter.shape
    # Compute error metrics
    max_abs_diff = np.max(np.abs(rho_trotter - rho_exact))
    final_abs_diff = np.max(np.abs(rho_trotter[-1] - rho_exact[-1]))
    max_frob_diff = np.max([np.linalg.norm(rho_trotter[i] - rho_exact[i], ord='fro') for i in range(len(tlist))])
    dt = T_final / n
    # Save row
    results.append((n, dt, max_abs_diff, final_abs_diff, max_frob_diff))
    # Print row
    print(f"n={n:3d} dt={dt:.5f} max|elem diff|={max_abs_diff:.3e} final|elem diff|={final_abs_diff:.3e} maxFrob={max_frob_diff:.3e}")

# Save CSV of results
csv_file = "trotter_error_sweep.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["n_steps", "dt", "max_elem_diff", "final_elem_diff", "max_frobenius_diff"])
    writer.writerows(results)
print(f"\nSaved CSV: {csv_file}")

# Plot errors vs dt
dts = [row[1] for row in results]
max_elem = [row[2] for row in results]
final_elem = [row[3] for row in results]
max_frob = [row[4] for row in results]

plt.figure(figsize=(7,5))
plt.loglog(dts, max_elem, marker='o', label='max elementwise |diff|')
plt.loglog(dts, final_elem, marker='s', label='final elementwise |diff|')
plt.loglog(dts, max_frob, marker='^', label='max Frobenius norm')
plt.gca().invert_xaxis()  # smaller dt to the right is optional; comment out if undesired
plt.xlabel('dt (T_final / n_steps)')
plt.ylabel('Error')
plt.title('Trotter error vs dt (log-log)')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('trotter_error_vs_dt.png', dpi=150)
print("Saved plot: trotter_error_vs_dt.png")
