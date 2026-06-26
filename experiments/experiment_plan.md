# Experiment Plan

## Phase 0: Smoke Test

Goal: prove the pipeline runs end to end before expensive experiments.

- Dataset: synthetic 8x8 binary patterns or a 1,000-sample MNIST subset.
- Model: tiny latent score model with a replaceable quantum/classical head.
- Training: 1-5 epochs.
- Outputs: generated sample grid, training loss curve, parameter count.
- Success criterion: no runtime failures; generated samples are nonblank and
  metrics script runs.

Operational runs are enumerated in `experiments/run_matrix.csv`. The matrix is
the source of truth for run identifiers, datasets, model variants, ablations,
seeds, and planned status.

## Phase 1: Classical Baselines

- Autoencoder or deterministic downsampling latent representation.
- Classical latent diffusion score network.
- Parameter-matched classical MLP score head.

## Phase 2: Hybrid Quantum Models

- HQ-LDM-A: quantum score head.
- HQ-LDM-B: quantum timestep-conditioning block.
- Optional HQ-LDM-C: quantum residual correction to a classical score estimate.

## Phase 3: Ablations

- Qubits: 4, 6, 8.
- Circuit depth: 1, 2, 4.
- Encoding: angle encoding first; amplitude encoding only if implementation
  complexity is justified.
- Ansatz: hardware-efficient layers first.
- Datasets: MNIST first, Fashion-MNIST second.
- QLCM placement:
  - `head`: quantum circuit is the primary consistency bottleneck.
  - `residual`: quantum circuit corrects a classical consistency estimate.
  - `time`: quantum circuit produces a time-conditioning signal.
- Sampling steps: 1, 2, 4.
- Seeds: 0 for screening, then 0/1/2 for primary MNIST comparison.

## Phase 4: Evaluation

Quantitative:

- KID or FID-like score.
- Classifier-based sample recognizability.
- Label entropy/diversity.
- Parameter count.
- Training and sampling time.

Qualitative:

- Generated sample grids.
- Reconstruction grids if an autoencoder is used.
- Failure examples.

## Phase 5: Manuscript Evidence Package

Required artifacts before writing the results section:

- Fixed random seeds.
- Config files for every reported run.
- Raw metrics table.
- Generated image grids.
- Training curves.
- Ablation table.
- Hardware/software environment notes.

## No-Compute Preparation Completed

- Config files: `configs/smoke.yaml`, `configs/mnist.yaml`,
  `configs/fashion_mnist.yaml`, `configs/ablation_grid.yaml`.
- Architecture code:
  - `src/qlcm/autoencoder.py`
  - `src/qlcm/teacher.py`
  - `src/qlcm/consistency.py`
  - `src/qlcm/losses.py`
  - `src/qlcm/sampling.py`
- Training entry points:
  - `scripts/train_autoencoder.py`
  - `scripts/train_teacher.py`
  - `scripts/train_qlcm.py`
- Table templates:
  - `tables/results_template.tex`
  - `tables/ablation_template.tex`
