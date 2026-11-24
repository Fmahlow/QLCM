# Execution Checklist

Nothing in this checklist should be started on a low-power machine unless the
expected runtime is acceptable. The current workspace is prepared for execution
but no training pipeline is running.

## Order of Execution

1. Smoke dataset autoencoder:
   `python3 scripts/train_autoencoder.py --config configs/smoke.yaml`

2. Smoke QLCM dry run only:
   `python3 scripts/train_qlcm.py --config configs/smoke.yaml --placement residual --autoencoder results/bars_stripes/autoencoder.pt --dry-run`

3. MNIST autoencoder:
   `python3 scripts/train_autoencoder.py --config configs/mnist.yaml`

4. MNIST latent diffusion teacher:
   `python3 scripts/train_teacher.py --config configs/mnist.yaml --autoencoder results/mnist/autoencoder.pt`

5. MNIST primary classical consistency baseline:
   `python3 scripts/train_qlcm.py --config configs/mnist.yaml --placement classical --autoencoder results/mnist/autoencoder.pt --run-id MNIST-CC-001 --sampling-steps 1`

6. MNIST primary QLCM-R:
   `python3 scripts/train_qlcm.py --config configs/mnist.yaml --placement residual --qubits 4 --depth 2 --autoencoder results/mnist/autoencoder.pt --run-id MNIST-QR-4-2-001 --sampling-steps 1`

7. Evaluate trained students:
   `python3 scripts/evaluate_qlcm.py --config configs/mnist.yaml --placement residual --autoencoder results/mnist/autoencoder.pt --student results/mnist/student_MNIST-QR-4-2-001.pt --steps 1`

8. Expand to the remaining matrix:
   `python3 scripts/generate_run_commands.py`

## Stop Conditions

- If autoencoder reconstruction is poor, do not run QLCM yet.
- If the teacher loss diverges, reduce learning rate before distillation.
- If QLCM-H collapses, prioritize QLCM-R.
- Treat 8-qubit runs as optional on CPU.

## Evidence Required Before Updating Paper Results

- Metrics JSON for every reported run.
- Sample grid for every reported run.
- Parameter counts.
- Runtime measurements.
- At least three seeds for the primary MNIST comparison.

## Execution Log (2026-06-30)

The full 26-run matrix in `experiments/run_matrix.csv` was executed on an
NVIDIA RTX 2080 Ti (`device: cuda` in `configs/mnist.yaml` and
`configs/fashion_mnist.yaml`, changed from the originally committed `cpu`).
All metrics JSON files, sample grids, and the aggregated table are under
`results/` and `results/summary.csv`.

Code issues found and fixed during this run, before trusting any numbers:

1. `scripts/train_qlcm.py`, `scripts/train_autoencoder.py`,
   `scripts/train_teacher.py` never called `torch.manual_seed`; the `seed`
   config field was read but unused. Added `--seed` CLI flag and seeding
   before model construction/data loading. All reported runs were trained
   after this fix.
2. `scripts/evaluate_qlcm.py` wrote `metrics_{placement}_{steps}step.json`
   and `samples_{placement}_{steps}step.png`, which collide across seeds and
   run IDs (e.g. all three classical seeds would overwrite the same file).
   Added `--run-id` tagging so every run's metrics and sample grid persist
   independently.
3. `scripts/evaluate_qlcm.py` did not accept `--qubits`/`--depth`, so it
   silently rebuilt ablated quantum students with the config's default
   architecture instead of the architecture they were trained with, causing
   `state_dict` shape-mismatch errors for any non-default qubit/depth run.
   Fixed by adding the flags and threading them into model construction.
4. `scripts/generate_run_commands.py` did not emit `--seed`, so the
   generated commands for `MNIST-QR-SEED1/2` and `MNIST-CC-SEED1/2` would
   have silently trained at the default seed. Fixed.
5. No FID/KID/recognizability/diversity metric existed (`src/metrics.py`
   only had parameter count and timing), despite being required by
   `experiments/experiment_plan.md` Phase 4. Added `src/eval_metrics.py`
   (classifier-feature FID-like, classifier confidence, label entropy) using
   a small CNN classifier trained per dataset; this is explicitly documented
   in the paper as not comparable to Inception-based FID.
6. The classifier-feature Frechet distance was unstable (NaN) with the
   original default of 64 generated samples against a 64-dimensional feature
   space (near-singular covariance). Fixed by raising the default sample
   count to 512 and adding small-epsilon covariance regularization in
   `frechet_distance`.
7. The multi-step sampler in `src/qlcm/sampling.py` was found to degrade
   FID-like scores as step count increases (1/2/4-step), traced to
   unconditioned noise injection between student evaluations with no
   denoising correction term. Not fixed in this pass; documented as an open
   issue in the paper (Section 5.3) and `README.md`.

All 26 run-matrix rows are marked `done`. The primary MNIST
classical-vs-QLCM-R comparison has seeds 0/1/2; all other rows are seed 0
only.

## Execution Log (2026-07-01): sampler fix, CD fix, stronger teacher, full 3-seed grid

Follow-up campaign (`scripts/run_full_campaign.sh`) addressing item 7 above
and one further issue found while diagnosing it. Results superseded issue 7
and the single-seed ablation caveat from the 2026-06-30 log; prior artifacts
are archived under `results/_archive_v1/` (pre-sampler-fix) and
`results/_archive_v2_ct_fixed_sampler/` (sampler fixed, CD bug not yet
found).

1. Fixed `src/qlcm/sampling.py` per issue 7 above: multistep sampling now
   re-noises via `schedule.add_noise` (the same forward process used in
   training) instead of an arbitrary small perturbation, following Song et
   al. (2023) Algorithm 1. Alone, this did **not** fix the few-step
   degradation (FID-like still worsened 15-30% from 1-step to 4-step in the
   `_archive_v2` results), which prompted further investigation.
2. Root cause of the remaining degradation: `scripts/train_qlcm.py` never
   loaded the trained teacher (`teacher.pt`) at all. The consistency loss
   compared `student(z_t, t)` against `student(z_s, s)` where `z_s` was
   obtained by re-noising the same ground-truth `z0` independently at level
   `s`, not by following the teacher's trajectory from `z_t`. This is
   consistency *training* (CT), not the consistency *distillation* (CD) the
   paper's Method section describes. Fixed by adding `LatentNoiseSchedule.ddim_step`
   (`src/qlcm/teacher.py`) and using it in `combined_qlcm_loss`
   (`src/qlcm/losses.py`) to derive `z_s` from a real teacher-predicted DDIM
   step starting at `z_t`; `train_qlcm.py` now requires `--teacher` and loads
   it frozen. This is the fix that stabilized few-step sampling; the sampler
   fix in isolation was necessary but not sufficient.
3. Strengthened the latent diffusion teacher: `epochs.teacher` raised from 50
   to 150 in `configs/mnist.yaml` and `configs/fashion_mnist.yaml`. Noise MSE
   plateaus at ≈0.85 (MNIST) / ≈0.76 (Fashion-MNIST), vs. still-falling ≈0.88
   / ≈0.79 at 50 epochs.
4. Re-ran the full MNIST placement × qubit × depth grid (8 architectures:
   classical, head/residual at qubits∈{4,6}×depth∈{1,2} restricted to the
   matrix's original cells, time at 4×2) at seeds 0, 1, 2 — 24 training runs
   — plus Fashion-MNIST classical and QLCM-R (4 qubits, depth 2) at seeds 0,
   1, 2 — 6 more runs. All 30 runs succeeded (0 failures). Every run was
   evaluated at 1-step; seed-0 runs were additionally evaluated at 2-step and
   4-step. Aggregated to `results/summary.csv` (50 rows) via
   `scripts/aggregate_results.py`.
5. Regenerated sample-grid figures with `--grid-samples 64` (added to
   `scripts/evaluate_qlcm.py`) instead of rendering all 512 evaluated
   samples, keeping the metrics sample count (512) unchanged for FID-like
   stability. Copied into `figures/`.

Headline change from this log: few-step sampling (1/2/4-step) is now
essentially flat instead of degrading, once training is genuine consistency
distillation. The QLCM-R-vs-classical and QLCM-H-failure findings from the
2026-06-30 log replicate under the corrected training procedure and now have
3-seed statistics for every MNIST ablation cell and the Fashion-MNIST primary
comparison (previously seed-0-only). Fashion-MNIST mode collapse toward a
single class persisted after all three fixes and is not yet explained.

## Execution Log (2026-09-15): quantum diffusion baseline (same-architecture speed test)

Motivation: the classical-vs-QLCM-R comparison above uses the same sampler
and step counts on both sides, so it cannot support a claim that QLCM is
*faster* than anything — only that it is not better at equal step count. The
speed claim that would matter (quantum consistency needs fewer steps than
existing quantum diffusion models) requires comparing against an iterative,
many-step quantum denoiser. Rather than rely on cross-paper NFE counts
against `kolle2024qddm`/`zhang2023quddpm`/`kwun2024mixedquddpm`, we built a
same-architecture internal baseline: `QuantumLatentTeacherScoreNet`
(`src/qlcm/teacher.py`), the same encoder/quantum-circuit/decoder module used
inside QLCM-R and QLCM-H (4 qubits, depth 2, 44,584 parameters), trained via
`scripts/train_quantum_teacher.py` with the standard DDPM noise-prediction
loss for 150 epochs (matching the classical teacher's budget) instead of
consistency distillation. `src/qlcm/sampling.py:sample_diffusion` samples it
with deterministic DDIM (reusing `LatentNoiseSchedule.ddim_step`) at a
configurable step count; `scripts/evaluate_quantum_diffusion.py` runs this
and computes the same classifier-feature metrics used everywhere else.

Ran on MNIST, 4 qubits/depth 2, 3 seeds:

1. Training loss plateaus at 0.954±0.001 noise MSE (last-10-epoch average)
   across all 3 seeds — essentially the 1.0 MSE of a trivial zero-noise
   predictor (noise is unit-variance), and far above the classical teacher's
   documented ≈0.85 plateau, despite the encoder/decoder wrapper (44,584
   params) being the same order of magnitude as the classical teacher's
   90,640; only the 16-parameter quantum circuit itself is qualitatively
   under-capacity.
2. Evaluated seed 0 at 1/2/4/8/16/32/50 DDIM steps: FID-like is flat
   (338.3 -> 339.5 -> 341.2 -> 342.3 -> 342.6 -> 342.8 -> 342.9), with no
   sign of the usual FID-vs-NFE improvement curve. Confirmed across all 3
   seeds at the two extremes: 1-step 337.9±1.5 vs. 50-step 345.0±4.7
   (3-seed mean±std), statistically indistinguishable.
3. Interpretation: this is a negative result for the intended "QLCM needs
   fewer steps than quantum diffusion" framing, not a confirmation of it. A
   score net that never learns a useful denoising direction cannot show a
   step-count-dependent quality curve to be faster than. The flatness is a
   symptom of under-capacity, not evidence that few steps suffice at this
   circuit size. This is mechanistically consistent with the already-reported
   QLCM-H failure (Section "Results and Analysis" of `paper/main.tex`): the
   same 4-qubit/depth-2 circuit fails whether used as a consistency head or
   as an iterative score net.
4. Scripts and artifacts: `scripts/train_quantum_teacher.py`,
   `scripts/evaluate_quantum_diffusion.py`,
   `scripts/run_quantum_diffusion_campaign.sh` (unattended re-run of the
   above); checkpoints at `results/mnist/quantum_teacher_4_2_seed{0,1,2}.pt`;
   metrics at `results/mnist/metrics_QDIFF-4-2-seed*-*step.json`, folded into
   `results/summary.csv` (`placement == "quantum_diffusion"`); training logs
   at `results/_log_train_quantum_teacher_seed{0,1,2}.txt`.
5. Not done: a parameter-matched or higher-capacity version of this baseline
   (e.g. more qubits/depth, or adding the classical residual backbone used in
   QLCM-R) to test whether a quantum diffusion model can be trained
   competitively at all before concluding anything about its step-count
   scaling. This is listed as future work in `paper/main.tex`'s Conclusion.

## Execution Log (2026-09-15, continued): reproducing Q-Dense (Koelle et al. 2024) as a literature-faithful quantum diffusion baseline

Motivation: the same-architecture ablation above (log entry directly above)
could not learn anything, so it could not demonstrate that QLCM needs fewer
steps than a working many-step quantum diffusion model. The user asked
specifically for "a standard quantum model found in the literature" to
compare against, not another variant of our own circuit. We fetched and read
the full text of Koelle et al. 2024, "Quantum Denoising Diffusion Models"
(arXiv:2401.07049), and reproduced their Q-Dense architecture as described in
Sec. 3.1/3.3/4.4 of that paper:

1. New modules: `src/qlcm/quantum_ops.py` (reusable statevector gate
   primitives: arbitrary single-qubit `Rot` gate, `RX`, and a CNOT
   implemented as an index permutation rather than permute+einsum, since
   CNOT only permutes basis amplitudes) and `src/qlcm/qdense.py`
   (`QDenseDiffusionCircuit`: amplitude-embeds a flattened image into
   `n_image_qubits` qubits, adds one ancilla qubit rotated by
   `class_index * 2*pi/n_classes` for guidance, applies `n_layers` layers of
   per-qubit `Rot` + a CNOT ring, then marginalizes the ancilla out of the
   measured probabilities and rescales to the input's norm). With 6 image
   qubits + 1 ancilla and 47 layers this is 987 parameters, matching the
   paper's reported ~1000 for its guided MNIST-8x8 model. Extended
   `src/datasets.py:build_dataset` with `image_size` (resize) and
   `digit_filter` (restrict to given labels) and a `signed` flag (keep raw
   [0,1] pixel intensities instead of the [-1,1] rescaling used everywhere
   else in this repo, since amplitude embedding and a probability-valued
   readout cannot represent negative values) — all backward compatible
   (default args reproduce the previous behavior).
2. The paper predicts the previous, less-noisy diffusion step directly (not
   noise) precisely because its readout is a non-negative probability
   distribution and noise is signed; we reproduce this in
   `scripts/train_qdense_baseline.py`, which builds an explicit forward
   noising trajectory (`forward_chain` in `src/qlcm/qdense.py`: repeated
   Gaussian noise addition with linearly increasing scale, clamped to
   [0, 1]) since the paper cites Ho et al. (2020) methodology but does not
   give the exact schedule for this direct-prediction loss; this is our
   documented interpretation, not a value from the source paper. Trained on
   MNIST digits 0/1 only, downsampled to 8x8 (matching the paper's own
   scoped guided experiment), 4000 images, batch size 256, tau=10, 150
   epochs, 3 seeds.
3. **Bug caught by looking at actual images, not just the metric**: an
   initial implementation of the paper's output-rescaling step ("scale the
   obtained probabilities using the input data's Euclidean norm") multiplied
   the raw probability vector by the input norm without first renormalizing
   the probability vector's own norm to match. Since probabilities summing
   to 1 over 64 outcomes have a small L2 norm unless sharply peaked, this
   caused the signal to shrink geometrically under the repeated, same-weight
   iteration this architecture requires (norm dropping ~5-8x per step in an
   untrained-weights check). The resulting samples were blank, saturated
   -white 8x8 tiles at every step count from 2 onward, with complete mode
   collapse to one class (512/512) — yet the FID-like number alone
   (~1300-1400) did not obviously flag this as "blank images" rather than
   merely "bad quality," so we caught it only by rendering and inspecting
   `results/qdense/samples_*.png` directly. Fixed in `QDenseDiffusionCircuit.forward`
   by rescaling the probability vector to match the input's norm
   (`image_probs * (norm / probs_norm)`) instead of merely multiplying by
   it, and retrained from scratch.
4. With the fix, training loss (MSE against the ground-truth previous-step
   image) drops from ~0.15 to ~0.10 over 150 epochs and plateaus there
   across all 3 seeds — real, replicated learning, unlike the under-capacity
   ablation in the log above. Sample grids after the fix show visible
   digit-like strokes rather than blank tiles. Evaluated at 1/2/4/6/8/10
   steps (samples=512) per seed: FID-like (classifier-feature) is
   genuinely non-flat and follows the same U-shape on all 3 seeds — best
   quality around 4 steps, worse at 1 step, worse still by 10 steps
   (3-seed mean±std: 602.9±7.8 at 1 step, 573.2±16.8 at 4 steps, 626.1±10.3
   at 10 steps). We interpret the post-4-step degradation as a consequence
   of reusing the identical fixed-weight circuit at every step with no
   timestep conditioning (per the paper's description): iterating a fixed
   nonlinear map repeatedly tends toward its own fixed point rather than
   continuing to refine the image, so the full trained tau is not
   necessarily the best sampling budget for this design.
5. Comparison: at every step count tested (1 through 10), this
   literature-faithful reproduction is substantially worse than QLCM-R
   (359.4±3.0 FID-like) and the classical consistency baseline
   (346.7±16.0), even at its own best-performing step count. This is a more
   meaningful comparison than the same-architecture ablation above, because
   this model does learn a real, replicated denoising signal and still does
   not close the gap with QLCM at any NFE budget tested. Caveats: different
   dataset scope (8x8 pixels, digits 0/1, no autoencoder, vs. the rest of
   this paper's 28x28-to-16-dim-latent, 10-class setup), our own
   interpretation of an unspecified forward-noising schedule, and only one
   reproduced architecture from one paper.
6. Scripts and artifacts: `src/qlcm/quantum_ops.py`, `src/qlcm/qdense.py`,
   `scripts/train_qdense_baseline.py`, `scripts/evaluate_qdense_baseline.py`;
   checkpoints at `results/qdense/qdense_seed{0,1,2}.pt`; metrics at
   `results/qdense/metrics_QDENSE-lit-seed*-*step.json`; training logs at
   `results/_log_train_qdense_seed{0,1,2}.txt`. Full writeup in
   `paper/main.tex` Section "A Literature-Faithful Quantum Diffusion
   Baseline: Reproducing Q-Dense".
7. Not done: reproducing Q-Dense at QLCM's own scale (28x28, 10 classes,
   latent-space via the shared autoencoder) for a directly matched
   comparison instead of the paper's own smaller 8x8/2-class scope;
   reproducing the paper's separate "unitary single-sampling" architecture
   (Sec. 3.4 of arXiv:2401.07049), which composes the per-step unitary into
   one matrix for genuine single-shot generation and is a different
   contribution from the sequential Q-Dense reproduced here. Both are listed
   as future work in `paper/main.tex`'s Conclusion.
