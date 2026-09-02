import csv
import numpy as np
import matplotlib.pyplot as plt

csv1 = '/home/marvan-mahamood/qsim/liouvillian_vs_qutip.csv'
csv2 = '/home/marvan-mahamood/qsim/trotter_vs_liouvillian_150.csv'

# Read liouvillian_vs_qutip.csv
n = []
dt = []
max_elem = []
final_elem = []
max_frob = []
with open(csv1, newline='') as f:
    r = csv.reader(f)
    header = next(r)
    for row in r:
        ni = int(row[0]); dti = float(row[1]); me = float(row[2]); fe = float(row[3]); mf = float(row[4])
        n.append(ni); dt.append(dti); max_elem.append(me); final_elem.append(fe); max_frob.append(mf)

print('Liouvillian vs QuTiP:')
print(' n  dt        max_elem_diff   final_elem_diff   max_frobenius')
for ni,dti,me,fe,mf in zip(n,dt,max_elem,final_elem,max_frob):
    print(f'{ni:3d} {dti:.5f} {me:14.6e} {fe:16.6e} {mf:14.6e}')

# Plot
plt.figure(figsize=(7,5))
plt.loglog(dt, max_elem, marker='o', label='max elementwise |diff|')
plt.loglog(dt, final_elem, marker='s', label='final elementwise |diff|')
plt.loglog(dt, max_frob, marker='^', label='max Frobenius')
plt.gca().invert_xaxis()
plt.xlabel('dt')
plt.ylabel('Error')
plt.title('Liouvillian-expm vs QuTiP (discrete propagator correctness)')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/home/marvan-mahamood/qsim/liouvillian_vs_qutip_plot.png', dpi=150)
print('\nSaved plot: /home/marvan-mahamood/qsim/liouvillian_vs_qutip_plot.png')

# Read trotter_vs_liouvillian_150.csv if exists
import os
if os.path.exists(csv2):
    print('\nTrotter vs Liouvillian for n=150:')
    with open(csv2, newline='') as f:
        r = csv.reader(f)
        rows = list(r)
        for row in rows:
            print(', '.join(row))
    # Also print a small summary if values present
    try:
        vals = {rows[i][0]: float(rows[i][1]) for i in range(1, len(rows))}
        print('\nSummary (n=150):')
        for k,v in vals.items():
            print(f'{k}: {v:.6e}')
    except Exception:
        pass
else:
    print('\nNo trotter_vs_liouvillian_150.csv found.')

