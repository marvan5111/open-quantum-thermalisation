# qsim — Trotterized Open-System Simulation (Exact Kraus)

This repository contains a research prototype for Trotterized open-system quantum simulation with an emphasis on extracting exact per-step Kraus maps from a Stinespring/Choi representation and validating convergence against classical Liouvillian evolution (QuTiP).

Contents
- A sequence of numbered scripts (step1_*.py … step32_*.py) that record an experimental workflow: building a QuTiP ground truth, implementing Trotterized circuits, diagnosing representation bugs, extracting exact Kraus maps, running convergence sweeps, and producing plots and a PDF presentation.
- Key outputs (generated, excluded via .gitignore): .npy, .csv, .png, .pdf files such as `exact_kraus_convergence.csv`, `fig_exact_kraus_errors_vs_dt.png`, and `qsim_presentation.pdf`.

Highlights / Findings
- The project demonstrates that an exact Kraus extraction (from the Choi root) can remove structural O(dt) bias from per-step circuit maps when the Choi-to-Kraus reshape/transpose convention and Kraus completeness are handled correctly.
- A debugging journey (see step26_choi_root_and_permute.py and related steps) found and fixed a reshape/transpose convention mismatch that prevented recovered Kraus operators from reproducing the intended superoperator. After fixing index conventions and renormalizing the Kraus set (enforce sum K†K ≈ I), per-step errors collapsed to machine precision and final-state differences fell to floating-point noise levels.

Requirements
- Python 3.8+ (recommended)
- numpy, scipy, matplotlib, Pillow, qutip

Suggested environment setup

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt    # create this file if you want reproducible installs

Quick run (small example)
- Generate the QuTiP ground truth: python3 step1_ground_truth.py
- Run the exact Kraus sweep: python3 step30_exact_kraus_sweep.py
- Produce plots & report: python3 step31_plots_and_report.py
- Build slides: python3 step32_make_slides.py

Notes
- Large generated artifacts (plots, numpy arrays, CSVs) are intentionally gitignored. Commit only code, scripts, and small configuration files.
- This repository is organized as an experimental notebook-in-code: the numbered steps are chronological experiments and debugging records; the README summarizes the narrative for readers.

If you want, I can:
- produce a compact `requirements.txt` from the used imports,
- add a small CONTRIBUTING.md describing how to reproduce results, or
- push the repository to GitHub (I will not perform pushes with embedded tokens; I can show the exact commands to run locally).

Author: marvan-mahamood
