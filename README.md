# Quantum Diffusion

Research code and manuscript workspace for a simulator-based hybrid quantum
diffusion model for synthetic image generation.

Current focus:

- MNIST and Fashion-MNIST first.
- Hybrid quantum-classical latent diffusion or score-based generation.
- Journal-style manuscript targeting quantum machine learning venues.

See `docs/phase1_scope/research_plan.md` for the initial research plan.

Status: the full MNIST placement/qubit/depth ablation grid and the
Fashion-MNIST classical/QLCM-R comparison have been run with genuine
consistency distillation (teacher-driven, not training from data alone) at 3
seeds per configuration (30 training runs, 50 evaluation rows), after fixing
two bugs that were silently invalidating earlier results: `train_qlcm.py`
never loaded the trained teacher, and the multistep sampler re-noised with an
arbitrary perturbation instead of the trained schedule. See
`experiments/execution_checklist.md` for the full bug list and fix log. With
both fixes and a stronger teacher (150 epochs), few-step (1/2/4-step)
sampling quality is stable rather than degrading. Results, including a
negative finding for the residual quantum placement and a well-replicated
failure mode for the direct quantum-head placement, are written up in
`paper/main.tex` (Section "Results and Analysis"). See
`results/summary.csv` for the raw aggregated metrics.

Remaining work before submission: diagnose the Fashion-MNIST mode collapse
(persisted after all fixes above), test an EMA target network for
consistency distillation, and extend qubit count to 8 if compute allows. The
manuscript's TikZ architecture figure and LaTeX have been checked for brace
balance and matched environments but not compiled with a LaTeX toolchain
(none installed in this environment) — compile once before submission to
catch any remaining syntax issues.

- Experiment matrix: `experiments/run_matrix.csv`
- Execution checklist: `experiments/execution_checklist.md`
- QLCM modules: `src/qlcm/`
- Training scripts: `scripts/train_autoencoder.py`, `scripts/train_teacher.py`,
  `scripts/train_qlcm.py`
- Evaluation script: `scripts/evaluate_qlcm.py`
- Results aggregation: `scripts/aggregate_results.py` -> `results/summary.csv`
- Manuscript: `paper/main.tex`
