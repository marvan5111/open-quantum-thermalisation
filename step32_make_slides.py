import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import textwrap

out_pdf = '/home/marvan-mahamood/qsim/qsim_presentation.pdf'
fig1 = '/home/marvan-mahamood/qsim/fig_exact_kraus_errors_vs_dt.png'
fig2 = '/home/marvan-mahamood/qsim/fig_exact_kraus_richardson.png'
report = '/home/marvan-mahamood/qsim/exact_kraus_report.txt'

# Read report text
with open(report, 'r') as f:
    report_text = f.read()

pp = PdfPages(out_pdf)

# Slide helpers
def add_text_slide(title, lines, fontsize=18):
    fig = plt.figure(figsize=(11,8.5))
    plt.axis('off')
    plt.title(title, fontsize=28, pad=20)
    wrapped = []
    for ln in lines:
        wrapped += textwrap.wrap(ln, width=90)
    plt.text(0.05, 0.9, '\n'.join(wrapped), fontsize=14, va='top')
    pp.savefig(fig)
    plt.close(fig)

def add_image_slide(title, image_path, caption=None):
    fig = plt.figure(figsize=(11,8.5))
    plt.axis('off')
    plt.title(title, fontsize=22, pad=12)
    try:
        img = Image.open(image_path)
        # fit image into a box
        plt.imshow(img)
        plt.gca().set_axis_off()
        if caption:
            plt.figtext(0.5, 0.03, caption, ha='center', fontsize=10)
    except Exception as e:
        plt.text(0.05, 0.5, f'Could not load image: {image_path}\n{e}', fontsize=12)
    pp.savefig(fig)
    plt.close(fig)

# Title slide
add_text_slide('qsim: Trotterized Open-System Simulation\nExact Kraus Convergence', [
    'Author: marvan-mahamood',
    'Date: 2026-09-02',
    '',
    'Overview: demonstration that exact Kraus extraction removes circuit-level O(dt) bias and yields machine-precision per-step maps.'
])

# Agenda
add_text_slide('Agenda', [
    '1. Problem and goals',
    '2. Method: Strang splitting + ancilla dilation',
    '3. Debugging summary and fixes',
    '4. Exact Kraus extraction results',
    '5. Convergence plots and Richardson extrapolation',
    '6. Conclusions and next steps'
])

# Problem + method
add_text_slide('Problem & Method', [
    'Goal: simulate Lindblad dynamics for 2 qubits using Trotterized circuits with ancilla-based amplitude damping dilation.',
    'Method highlights:',
    '- Strang symmetric splitting: exp(L_D dt/2) exp(L_H dt) exp(L_D dt/2)',
    '- Ancilla dilation for amplitude damping per qubit',
    '- Kraus extraction from Choi root to obtain exact per-step maps',
    '',
    'Validation: compare circuit Trotter outputs to classical QuTiP ground truth.'
])

# Debugging summary
add_text_slide('Debugging summary (key fixes)', [
    '- Fixed conditional swap embedding; use unconditional swap for consistency.',
    '- Use exact finite-time parameter: gamma_step = 1 - exp(-gamma*dt).',
    '- Resolved Choi ↔ Kraus reshape convention via transpose before normalization.',
    '- Enforced Kraus completeness with polar/sqrt normalization to get S_from_K ≈ P to 1e-15.'
])

# Insert convergence figure
add_image_slide('Convergence: Errors vs dt (log-log)', fig1, caption='Per-step error at machine precision; final errors ~1e-15..1e-13')

# Insert Richardson figure
add_image_slide('Richardson extrapolation (p=2)', fig2, caption='Extrapolated error and original errors (log scale)')

# Report excerpt slide
add_text_slide('Report excerpt', report_text.split('\n')[:20])

# Conclusions
add_text_slide('Conclusions & Next steps', [
    '- Exact-Kraus approach removes structural O(dt) per-step bias (per-step errors ~1e-15).',
    '- Final-state differences are at floating-point noise levels; method is validated.',
    '- Next: (recommended) Richardson extrapolation on observables to produce publication-ready estimates; (optional) synthesize exact Kraus into implementable circuits.',
    '- I can prepare slides as PDF (done) and a short markdown write-up if desired.'
])

pp.close()
print('Wrote', out_pdf)
