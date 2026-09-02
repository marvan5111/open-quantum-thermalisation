import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import log

csv_path = '/home/marvan-mahamood/qsim/exact_kraus_convergence.csv'
out_prefix = '/home/marvan-mahamood/qsim/fig_exact_kraus'

rows = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({k: float(v) for k,v in r.items()})

# Sort by dt ascending
rows = sorted(rows, key=lambda r: r['dt'])

dt = np.array([r['dt'] for r in rows])
per_step = np.array([r['per_step_error'] for r in rows])
final_max = np.array([r['final_max_diff'] for r in rows])
final_frob = np.array([r['final_frob_diff'] for r in rows])

# Plot final_max vs dt (log-log)
plt.figure(figsize=(6,4))
plt.loglog(dt, final_max, 'o-', label='final max diff')
plt.loglog(dt, final_frob, 's--', label='final frob diff')
plt.loglog(dt, per_step, 'x-.', label='per-step err')
plt.gca().invert_xaxis()
plt.xlabel('dt')
plt.ylabel('Error (log scale)')
plt.title('Exact-Kraus Convergence')
plt.grid(True, which='both', ls=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(out_prefix + '_errors_vs_dt.png', dpi=200)

# Show orders: fit power law final_max = c * dt^p
log_dt = np.log(dt)
log_final = np.log(final_max)
coeffs = np.polyfit(log_dt, log_final, 1)
exp_p = coeffs[0]
prefactor = np.exp(coeffs[1])

# Richardson extrapolation (p=2). Use pairs where ratio known or compute r.
p_order = 2.0
extrap_results = []
for i in range(len(dt)-1):
    dt1 = dt[i+1]  # smaller
    dt0 = dt[i]
    r = dt0/dt1
    E0 = final_max[i]
    E1 = final_max[i+1]
    # generalized Richardson: true ≈ (r^p * E1 - E0) / (r**p - 1)
    if r**p_order - 1 != 0:
        E_ex = (r**p_order * E1 - E0) / (r**p_order - 1)
    else:
        E_ex = np.nan
    extrap_results.append((dt0, dt1, r, E0, E1, E_ex))

# Save a short report
with open('exact_kraus_report.txt', 'w') as fo:
    fo.write('Exact Kraus convergence report\n')
    fo.write('=================================\n')
    fo.write(f'data file: {csv_path}\n')
    fo.write('\nData points (dt, per-step, final-max, final-frob):\n')
    for r in rows:
        fo.write(f"dt={r['dt']:g} per-step={r['per_step_error']:.3e} final-max={r['final_max_diff']:.3e} final-frob={r['final_frob_diff']:.3e}\n")
    fo.write('\nPower-law fit (final_max ~ c * dt^p) [log-log polyfit]:\n')
    fo.write(f'  fitted exponent p ≈ {exp_p:.3f} (expect ~2 for Strang)\n')
    fo.write(f'  prefactor c ≈ {prefactor:.3e}\n')

    fo.write('\nRichardson extrapolation (p=2) using consecutive pairs:\n')
    fo.write('dt0  dt1   r    E(dt0)       E(dt1)       Extrapolated\n')
    for (dt0, dt1, r, E0, E1, E_ex) in extrap_results:
        fo.write(f'{dt0:.5g} {dt1:.5g} {r:.3g} {E0:.3e} {E1:.3e} {E_ex:.3e}\n')
    fo.write('\nNotes:\n- Per-step errors are at machine precision (~1e-15).\n- Final errors are extremely small (1e-15..1e-13) likely due to floating-point accumulation.\n')

# Additional plot: Richardson extrapolated vs original final_max
# Use extrapolation from the smallest pair
if len(extrap_results) > 0:
    dt0, dt1, r, E0, E1, E_ex = extrap_results[-1]
    # bar plot of E(dt1), E(dt0), E_ex
    labels = [f'E({dt1})', f'E({dt0})', 'Extrapolated']
    vals = [E1, E0, E_ex]
    plt.figure(figsize=(5,3))
    plt.bar(labels, vals, color=['C0','C1','C2'])
    plt.yscale('log')
    plt.ylabel('Error (log)')
    plt.title('Richardson extrapolation (p=2)')
    plt.tight_layout()
    plt.savefig(out_prefix + '_richardson.png', dpi=200)

print('Saved figures:', out_prefix + '_errors_vs_dt.png', out_prefix + '_richardson.png')
print('Report saved: exact_kraus_report.txt')
