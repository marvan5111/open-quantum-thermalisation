import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace

# Must match Stage 1 QuTiP parameters exactly
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0

def trotter_unitary(dt):
    qc = QuantumCircuit(2)
    qc.h(0); qc.h(1)
    qc.cx(0, 1)
    qc.rz(2 * Jxx * dt, 1)
    qc.cx(0, 1)
    qc.h(0); qc.h(1)
    qc.rz(2 * h0 * dt, 0)
    qc.rz(2 * h1 * dt, 1)
    return qc

def damping_step_kraus(gamma_step):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma_step)]])
    K1 = np.array([[0, np.sqrt(gamma_step)], [0, 0]])
    return [K0, K1]

def apply_1qubit_kraus(rho2, kraus_ops, qubit_idx):
    I2 = np.eye(2)
    new_rho = np.zeros_like(rho2, dtype=complex)
    for K in kraus_ops:
        full_K = np.kron(I2, K) if qubit_idx == 0 else np.kron(K, I2)
        new_rho += full_K @ rho2 @ full_K.conj().T
    return new_rho

def run_trotter_sim(n_steps):
    dt = T_final / n_steps
    U_step = Operator(trotter_unitary(dt)).data

    e = np.array([[1, 0], [0, 0]], dtype=complex)
    rho = DensityMatrix(e).tensor(DensityMatrix(e)).data

    # Strang splitting with dilation-based damping: apply half dissipator (via ancilla), unitary, then half dissipator
    gamma0_half = 1 - np.exp(-gamma0 * (dt / 2))
    gamma1_half = 1 - np.exp(-gamma1 * (dt / 2))

    def apply_dilation_damping(rho2, gamma_step, qubit_idx):
        theta = 2 * np.arcsin(np.sqrt(gamma_step))
        dm_sys = DensityMatrix(rho2)
        dm_anc = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
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
        # If target is q0 (rightmost), swap q0 and q1 to bring target to position 1
        if qubit_idx == 0:
            qc3.swap(1, 2)
            swapped = True
        # Apply the 2-qubit dilation on (ancilla at 0, target now at 1)
        qc3.cry(theta, 0, 1)
        qc3.cx(1, 0)
        # swap back if we swapped earlier
        if swapped:
            qc3.swap(1, 2)

        # Build full 8x8 unitary and apply
        U8 = Operator(qc3).data
        dm_out_mat = U8 @ dm_total.data @ U8.conj().T
        # Trace out ancilla (dimension 2) from dm_out_mat shaped (2,4,2,4) -> result 4x4
        rho_sys = np.einsum('aibj->ij', dm_out_mat.reshape(2, 4, 2, 4))
        return rho_sys

    snapshots = [rho.copy()]
    for _ in range(n_steps):
        # half dissipator via dilation per qubit
        rho = apply_dilation_damping(rho, gamma0_half, 0)
        rho = apply_dilation_damping(rho, gamma1_half, 1)
        # full unitary
        rho = U_step @ rho @ U_step.conj().T
        # half dissipator via dilation per qubit
        rho = apply_dilation_damping(rho, gamma0_half, 0)
        rho = apply_dilation_damping(rho, gamma1_half, 1)
        snapshots.append(rho.copy())

    return np.array(snapshots)

if __name__ == "__main__":
    n_steps = 150
    print(f"Running Trotter simulation: n_steps={n_steps}, dt={T_final/n_steps:.4f}")
    rho_trotter = run_trotter_sim(n_steps)
    np.save("rho_trotter_150.npy", rho_trotter)
    print("Trotter run shape:", rho_trotter.shape)
    print("Final trace:", np.trace(rho_trotter[-1]).real)
    print("Hermitian check (max asymmetry):",
          np.max(np.abs(rho_trotter[-1] - rho_trotter[-1].conj().T)))
