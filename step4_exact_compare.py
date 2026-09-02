# Compare the Trotterized open-system simulation against the exact QuTiP solution.
# This script matches the parameters used in step3_trotter_circuit.py exactly.
import numpy as np
from qutip import basis, sigmax, sigmaz, qeye, tensor, mesolve, Qobj

# Physical parameters must match the Trotter simulation.
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0
n_steps = 150

# Basic operators used in the two-qubit model.
sx = sigmax()
sz = sigmaz()
I2 = qeye(2)

# Lowering operator for amplitude damping: |1><0|.
sm = Qobj(np.array([[0, 0], [1, 0]], dtype=complex))

# Hamiltonian for the two-qubit system:
# H = Jxx * X⊗X + h0 * Z⊗I + h1 * I⊗Z
H = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)

# Initial state: |00><00|
psi0 = tensor(basis(2, 0), basis(2, 0))
rho0 = psi0 * psi0.dag()

# Lindblad collapse operators for amplitude damping on qubit 0 and qubit 1.
c_ops = [
    np.sqrt(gamma0) * tensor(sm, I2),
    np.sqrt(gamma1) * tensor(I2, sm),
]

# Time grid aligned with the Trotter snapshots.
tlist = np.linspace(0.0, T_final, n_steps + 1)

# Exact master-equation evolution.
result = mesolve(H, rho0, tlist, c_ops)
rho_exact = np.array([state.full() for state in result.states], dtype=complex)

# Load the Trotterized density matrices saved earlier.
rho_trotter = np.load("rho_trotter_150.npy")

# Error metrics to quantify the Trotter approximation.
max_abs_diff = np.max(np.abs(rho_trotter - rho_exact))
final_abs_diff = np.max(np.abs(rho_trotter[-1] - rho_exact[-1]))
max_frob_diff = np.max([
    np.linalg.norm(rho_trotter[i] - rho_exact[i], ord="fro")
    for i in range(len(tlist))
])

# Print summary statistics.
print(f"Exact trace at final time: {np.trace(rho_exact[-1]).real:.12f}")
print(f"Trotter trace at final time: {np.trace(rho_trotter[-1]).real:.12f}")
print(f"Maximum elementwise |diff| over all steps: {max_abs_diff:.3e}")
print(f"Final elementwise |diff|: {final_abs_diff:.3e}")
print(f"Maximum Frobenius norm diff over time: {max_frob_diff:.3e}")
