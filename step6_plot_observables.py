import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmaz, qeye, tensor, mesolve

# Parameters (must match previous runs)
Jxx = 1.0
h0, h1 = 0.7, 1.1
gamma0, gamma1 = 0.15, 0.10
T_final = 6.0

# Load Trotter result
rho_trotter = np.load('/home/marvan-mahamood/qsim/rho_trotter_150.npy')  # shape (n+1, 4, 4)
n_points = rho_trotter.shape[0]

# Build exact QuTiP evolution on same grid
n_steps = n_points - 1
tlist = np.linspace(0.0, T_final, n_points)
from qutip import sigmax, sigmaz, sigmay, qeye, tensor
sx = sigmax(); sz = sigmaz(); I2 = qeye(2)
H = Jxx * tensor(sx, sx) + h0 * tensor(sz, I2) + h1 * tensor(I2, sz)
psi0 = tensor(basis(2,0), basis(2,0))
rho0 = psi0 * psi0.dag()
from qutip import Qobj
c_ops = [np.sqrt(gamma0) * tensor(Qobj(np.array([[0,0],[1,0]], dtype=complex)), I2),
         np.sqrt(gamma1) * tensor(I2, Qobj(np.array([[0,0],[1,0]], dtype=complex)))]
result = mesolve(H, rho0, tlist, c_ops)
rho_exact = np.array([state.full() for state in result.states], dtype=complex)

# Pauli-Z operator (numpy) and two placements
sz_np = np.array([[1,0],[0,-1]], dtype=complex)
I2_np = np.eye(2, dtype=complex)
op_z_q0 = np.kron(sz_np, I2_np)  # Z on first factor
op_z_q1 = np.kron(I2_np, sz_np)  # Z on second factor

# Compute exact expectations (using QuTiP ordering: tensor(sz, I) is Z on qubit0)
exp_exact_z0 = np.real([np.trace(op_z_q0 @ rho_exact[i]) for i in range(n_points)])
exp_exact_z1 = np.real([np.trace(op_z_q1 @ rho_exact[i]) for i in range(n_points)])

# Compute trotter expectations for both possible conventions
exp_trot_z_if = np.real([np.trace(op_z_q0 @ rho_trotter[i]) for i in range(n_points)])
exp_trot_z_ir = np.real([np.trace(op_z_q1 @ rho_trotter[i]) for i in range(n_points)])
# The other mapping (swapped)
exp_trot_z_if_swapped = np.real([np.trace(op_z_q1 @ rho_trotter[i]) for i in range(n_points)])
exp_trot_z_ir_swapped = np.real([np.trace(op_z_q0 @ rho_trotter[i]) for i in range(n_points)])

# Decide which mapping matches exact better for z0
err_map_A = np.mean(np.abs(exp_trot_z_if - exp_exact_z0))
err_map_B = np.mean(np.abs(exp_trot_z_if_swapped - exp_exact_z0))
if err_map_A <= err_map_B:
    mapping = 'same'
    exp_trot_z0 = exp_trot_z_if
    exp_trot_z1 = exp_trot_z_ir
else:
    mapping = 'swapped'
    exp_trot_z0 = exp_trot_z_if_swapped
    exp_trot_z1 = exp_trot_z_ir_swapped

# Save CSV
import csv
csv_file = '/home/marvan-mahamood/qsim/observables_z_trotter_vs_qutip.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t', 'z0_exact', 'z0_trotter', 'z1_exact', 'z1_trotter'])
    for i, t in enumerate(tlist):
        writer.writerow([t, exp_exact_z0[i], exp_trot_z0[i], exp_exact_z1[i], exp_trot_z1[i]])

# Plot
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(tlist, exp_exact_z0, label='Exact ⟨Z0⟩', lw=2)
plt.plot(tlist, exp_trot_z0, '--', label='Trotter ⟨Z0⟩')
plt.xlabel('t'); plt.ylabel('⟨Z0⟩'); plt.title('Z0 vs time'); plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
plt.plot(tlist, exp_exact_z1, label='Exact ⟨Z1⟩', lw=2)
plt.plot(tlist, exp_trot_z1, '--', label='Trotter ⟨Z1⟩')
plt.xlabel('t'); plt.ylabel('⟨Z1⟩'); plt.title('Z1 vs time'); plt.legend(); plt.grid(True)

plt.suptitle(f'Observable comparison (mapping={mapping})')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('/home/marvan-mahamood/qsim/observables_trotter_vs_qutip.png', dpi=150)
print('Saved plot: /home/marvan-mahamood/qsim/observables_trotter_vs_qutip.png')
print('Saved CSV:', csv_file)
# Print brief summary
print('Mapping chosen:', mapping)
print('Mean absolute Z0 error:', np.mean(np.abs(exp_trot_z0 - exp_exact_z0)))
print('Mean absolute Z1 error:', np.mean(np.abs(exp_trot_z1 - exp_exact_z1)))
