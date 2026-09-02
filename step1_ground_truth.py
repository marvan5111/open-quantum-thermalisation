import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pauli operators and identity
sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
sm = qt.sigmam()
I2 = qt.qeye(2)

def op_on(op, qubit, n=2):
    ops = [I2] * n
    ops[qubit] = op
    return qt.tensor(ops)

# Hamiltonian: coupling + local fields
Jxx = 1.0
h0, h1 = 0.7, 1.1
H = (Jxx * op_on(sx, 0) * op_on(sx, 1)
     + h0 * op_on(sz, 0)
     + h1 * op_on(sz, 1))

# Damping (energy relaxation) on each qubit
gamma0, gamma1 = 0.15, 0.10
c_ops = [np.sqrt(gamma0) * op_on(sm, 0), np.sqrt(gamma1) * op_on(sm, 1)]

# Start both qubits excited
psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
rho0 = psi0 * psi0.dag()

tlist = np.linspace(0, 6, 300)
result = qt.mesolve(H, rho0, tlist, c_ops, e_ops=[])




purity = [(s * s).tr().real for s in result.states]
entropy = [qt.entropy_vn(s) for s in result.states]

print("Final purity:", purity[-1])
print("Final entropy:", entropy[-1])

plt.plot(tlist, purity, label="purity")
plt.plot(tlist, entropy, label="entropy")
plt.legend()
plt.savefig("ground_truth.png", dpi=130)
print("Saved ground_truth.png")



