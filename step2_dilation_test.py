import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace

def amplitude_damping_kraus(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    return [K0, K1]

def apply_kraus(rho, kraus_ops):
    return sum(K @ rho @ K.conj().T for K in kraus_ops)

def run_dilation_test(gamma, rho_system):
    theta = 2 * np.arcsin(np.sqrt(gamma))
    dm_sys = DensityMatrix(rho_system)
    dm_anc = DensityMatrix(np.array([[1, 0], [0, 0]]))
    dm2 = dm_anc.tensor(dm_sys)

    qc = QuantumCircuit(2)
    qc.cry(theta, 0, 1)
    qc.cx(1, 0)

    dm2_out = dm2.evolve(qc)
    rho_out = partial_trace(dm2_out, [1]).data
    return rho_out

test_states = {
    "|0><0|": np.array([[1, 0], [0, 0]], dtype=complex),
    "|1><1|": np.array([[0, 0], [0, 1]], dtype=complex),
    "|+><+|": 0.5 * np.array([[1, 1], [1, 1]], dtype=complex),
    "mixed":  np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex),
}

all_pass = True
for gamma in [0.1, 0.3, 0.5, 0.9]:
    kraus = amplitude_damping_kraus(gamma)
    for name, rho in test_states.items():
        rho_analytical = apply_kraus(rho, kraus)
        rho_circuit = run_dilation_test(gamma, rho)
        diff = np.max(np.abs(rho_analytical - rho_circuit))
        status = "OK" if diff < 1e-9 else "MISMATCH"
        if diff >= 1e-9:
            all_pass = False
        print(f"gamma={gamma:.1f} state={name:10s} max|diff|={diff:.2e}  {status}")

print("\nALL TESTS PASSED" if all_pass else "\nBUG FOUND")
