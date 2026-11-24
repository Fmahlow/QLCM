# QLCM v1 (archived)

This is the first iteration of the QLCM research project, archived as-is on
2026-09-15. It grew experiment-by-experiment in response to follow-up
questions rather than from an upfront, pre-registered design, which produced
some real methodological inconsistencies (most notably: the quantum
diffusion baselines added late in the process used a different, ad hoc
training-epoch budget than the original consistency students, chosen by
analogy rather than by a controlled comparison). v2 (repo root) starts over
with an explicit research design agreed on before running experiments.

Nothing here is deleted or considered wrong outright — the core MNIST/Fashion
-MNIST classical-vs-QLCM-R/H/T comparison (3 seeds, parameter-matched,
documented bug fixes) is a reasonable reference point, and the two follow-up
quantum-diffusion-baseline experiments are honestly-reported negative/mixed
results. Treat this folder as a source of lessons learned and possibly
reusable code, not as the paper to submit.

---

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

A follow-up experiment (`scripts/train_quantum_teacher.py` +
`scripts/evaluate_quantum_diffusion.py`, see execution log below) tested
whether QLCM's few-step sampling is faster than a same-architecture quantum
diffusion baseline, since the existing classical-vs-QLCM-R comparison uses
the same step counts on both sides and so cannot support a speed claim. The
quantum diffusion baseline — the same 4-qubit/depth-2 encoder/circuit/decoder
module trained as a standard multi-step noise predictor instead of a
consistency student — never learned a useful score function (training loss
plateaus at 0.954 noise MSE, barely below the 1.0 of a trivial zero predictor
vs. the classical teacher's ≈0.85) and its sample quality is flat from 1 to
50 sampling steps. This is a negative result for the "faster than quantum
diffusion" framing, not a confirmation of it: the comparison requires a
working many-step quantum diffusion baseline to be faster *than*, and this
one does not train competitively at this qubit/depth budget. See
`paper/main.tex` Section "Quantum Diffusion Baseline" for the full writeup.

Since that baseline couldn't learn anything, it couldn't demonstrate a real
speed disadvantage relative to QLCM. So we separately reproduced an actual
published quantum diffusion architecture — Q-Dense from Koelle et al. 2024,
"Quantum Denoising Diffusion Models" (arXiv:2401.07049) — faithfully:
amplitude-embedded 8x8 MNIST digits 0/1, ancilla-guided, 47 dense layers (987
params, matching their reported ~1000), τ=10 diffusion steps, trained to
predict the previous, less-noisy step directly (`src/qlcm/qdense.py`,
`scripts/train_qdense_baseline.py`, `scripts/evaluate_qdense_baseline.py`).
An initial bug in the output-rescaling step caused a geometric collapse to
blank white images under repeated iteration (caught by looking at the actual
sample grids, not just the FID-like number, which looked superficially
plausible); after fixing it, this reproduction does learn a real, 3-seed
-replicated denoising signal and shows a genuine FID-like-vs-NFE curve that
improves from 1 step to a minimum around 4 steps, then degrades through the
full trained 10 steps (3-seed mean±std FID-like: 602.9±7.8 at 1 step,
573.2±16.8 at 4 steps, 626.1±10.3 at 10 steps) — plausibly because the same
fixed-weight circuit is reapplied at every step with no timestep
conditioning, so iterating it drifts toward its own fixed point rather than
continuing to refine the image. Even at its best step count it remains
substantially worse than QLCM-R (359.4±3.0) and the classical baseline
(346.7±16.0). This is suggestive (not conclusive — different dataset scope,
one reproduced architecture, our own interpretation of an underspecified
noise schedule, **and an unresolved epoch-budget mismatch: 150 epochs for
Q-Dense vs. 50 for QLCM-R, chosen by analogy to different reference points in
the codebase rather than by a controlled matched-budget test**) evidence that
QLCM's few-step sampling is not simply failing to be worse than many-step
quantum diffusion by default. See `paper/main.tex` Section "A
Literature-Faithful Quantum Diffusion Baseline: Reproducing Q-Dense" for the
full writeup.

Remaining work before submission (superseded by the v2 restart, kept for
reference): diagnose the Fashion-MNIST mode collapse (persisted after all
fixes above), test an EMA target network for consistency distillation,
extend qubit count to 8 if compute allows, retry the quantum diffusion
baseline with higher circuit capacity, reproduce Q-Dense at QLCM's own
28x28-latent/10-class scale for a directly matched comparison, and resolve
the epoch-budget mismatch noted above. The manuscript's TikZ architecture
figure and LaTeX have been checked for brace balance and matched environments
but not compiled with a LaTeX toolchain (none installed in this environment).

- Experiment matrix: `experiments/run_matrix.csv`
- Execution checklist: `experiments/execution_checklist.md`
- QLCM modules: `src/qlcm/`
- Training scripts: `scripts/train_autoencoder.py`, `scripts/train_teacher.py`,
  `scripts/train_qlcm.py`, `scripts/train_quantum_teacher.py`,
  `scripts/train_qdense_baseline.py`
- Evaluation scripts: `scripts/evaluate_qlcm.py`, `scripts/evaluate_quantum_diffusion.py`,
  `scripts/evaluate_qdense_baseline.py`
- Results aggregation: `scripts/aggregate_results.py` -> `results/summary.csv`
- Manuscript: `paper/main.tex`
