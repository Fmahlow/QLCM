# Quantum Diffusion

Research code and manuscript workspace for a simulator-based hybrid quantum
diffusion model for synthetic image generation.

Current focus:

- MNIST and Fashion-MNIST first.
- Hybrid quantum-classical latent diffusion or score-based generation.
- Journal-style manuscript targeting quantum machine learning venues.

See `docs/phase1_scope/research_plan.md` for the initial research plan.

Status: the full planned run matrix (`experiments/run_matrix.csv`, 26 runs)
has been executed once on MNIST, Fashion-MNIST, and a synthetic
Bars-and-Stripes smoke dataset, with the primary MNIST classical-vs-QLCM-R
comparison repeated across 3 seeds. Results, including a negative finding for
the residual quantum placement and a failure mode for the direct quantum-head
placement, are written up in `paper/main.tex` (Section "Results and
Analysis"). See `results/summary.csv` for the raw aggregated metrics and
`experiments/execution_checklist.md` for the evidence trail.

Remaining work before submission: re-run the few-step sampler after fixing
the noise-injection issue identified in `src/qlcm/sampling.py`, repeat the
ablation grid (currently single-seed) across multiple seeds, and extend
qubit count to 8 if compute allows.

- Experiment matrix: `experiments/run_matrix.csv`
- Execution checklist: `experiments/execution_checklist.md`
- QLCM modules: `src/qlcm/`
- Training scripts: `scripts/train_autoencoder.py`, `scripts/train_teacher.py`,
  `scripts/train_qlcm.py`
- Evaluation script: `scripts/evaluate_qlcm.py`
- Results aggregation: `scripts/aggregate_results.py` -> `results/summary.csv`
- Manuscript: `paper/main.tex`
