# Quantum Diffusion

Research code and manuscript workspace for a simulator-based hybrid quantum
diffusion model for synthetic image generation.

Current focus:

- MNIST and Fashion-MNIST first.
- Hybrid quantum-classical latent diffusion or score-based generation.
- Journal-style manuscript targeting quantum machine learning venues.

See `docs/phase1_scope/research_plan.md` for the initial research plan.

Prepared but not currently running:

- Experiment matrix: `experiments/run_matrix.csv`
- Execution checklist: `experiments/execution_checklist.md`
- QLCM modules: `src/qlcm/`
- Training scripts: `scripts/train_autoencoder.py`, `scripts/train_teacher.py`,
  `scripts/train_qlcm.py`
- Evaluation script: `scripts/evaluate_qlcm.py`
- Manuscript: `paper/main.tex`
