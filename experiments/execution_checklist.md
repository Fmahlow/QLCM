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
