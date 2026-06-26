# Quantum Diffusion for Synthetic Image Generation: Research Plan

## Working Title

Quantum Latent Consistency Models for Low-Resolution Synthetic Image Generation

## Current Stage

Stage 1: Research scoping and methodology design.

The project starts from a simulator-only setting and targets MNIST/Fashion-MNIST
as first benchmarks. The intended manuscript style is a journal article suitable
for venues such as Quantum Machine Intelligence or Quantum Information
Processing.

## Core Research Direction

Develop and evaluate a hybrid quantum-classical consistency model for synthetic
image generation in which a parameterized quantum circuit contributes to a
latent consistency function distilled from a small latent diffusion teacher.

The most defensible initial positioning is not "the first quantum diffusion
model". The literature already contains quantum and hybrid diffusion-related
generative models. The intended contribution should instead be a specific
architectural and experimental claim, such as:

- a compact quantum score head for low-dimensional latent/image representations;
- a quantum-conditioned denoising network where time and noisy image features
  are encoded into a parameterized circuit;
- a controlled ablation showing where the quantum module improves or degrades
  sample quality, diversity, or parameter efficiency;
- a reproducible simulator benchmark with transparent limitations.

## Candidate Primary Research Question

Can a simulator-based hybrid quantum latent consistency model produce
recognizable one-step or few-step MNIST/Fashion-MNIST samples while matching or
improving parameter efficiency relative to classical consistency students?

## Sub-questions

1. Which hybrid placement of the quantum module is most effective: direct
   consistency head, time-conditioning block, or residual consistency correction?
2. Does the quantum module improve sample quality, diversity, or parameter
   efficiency under matched parameter and compute budgets?
3. How sensitive is performance to circuit depth, number of qubits, encoding
   strategy, and measurement scheme?

## Preliminary FINER Assessment

| Criterion | Score | Rationale |
| --- | ---: | --- |
| Feasible | 4/5 | Simulator-only experiments on MNIST/Fashion-MNIST are feasible with small latent/image sizes. |
| Interesting | 4/5 | The question sits at the intersection of diffusion models and quantum machine learning. |
| Novel | 3/5 | Quantum diffusion-related work exists; novelty must be narrowed to a specific architecture and benchmark claim. |
| Ethical | 5/5 | Uses public image datasets and synthetic generation; low direct human-subject risk. |
| Relevant | 4/5 | Relevant to practical limits of near-term quantum generative modeling. |
| Average | 4.0/5 | Good candidate if novelty is validated carefully. |

## Scope Boundaries

### In Scope

- Simulator-only quantum circuits.
- MNIST and Fashion-MNIST as first datasets.
- Low-resolution or compressed representations, likely 8x8, 14x14, or learned
  latent vectors.
- Hybrid quantum-classical diffusion or score-based generative models.
- Comparison against compact classical DDPM/score-network baselines.
- Ablations over circuit depth, qubit count, encoding, ansatz, and quantum module
  placement.

### Out of Scope

- Claims of quantum advantage.
- Hardware execution as a requirement.
- Large-scale natural image generation.
- Text-to-image or conditional prompt generation.
- Unverified claims that the architecture is the first of its kind.

## Candidate Architecture Families

### Primary: Quantum Latent Consistency Model

Train a compact classical latent diffusion teacher and distill its trajectory
into a hybrid quantum consistency student. The student maps noisy latent states
at arbitrary timesteps directly toward a clean latent endpoint, supporting
one-step or few-step generation.

See `docs/phase1_scope/quantum_latent_consistency_model.md`.

### A. Quantum Score Head

A classical encoder maps the noisy image and timestep to a compact feature
vector. A parameterized quantum circuit receives the features via angle or
amplitude encoding and returns expectation values used as the score/noise
prediction head.

Advantages: easiest to implement and compare fairly.

Risk: may be viewed as a small quantum layer attached to a classical model.

### B. Quantum Time-Conditioning Block

The quantum circuit encodes diffusion timestep information and produces a
conditioning vector used by a classical denoiser.

Advantages: directly targets a distinctive part of diffusion modeling.

Risk: contribution may be weaker if the denoising path remains entirely
classical.

### C. Quantum Latent Diffusion

Train a compact autoencoder or use dimensionality reduction, then run diffusion
in a low-dimensional latent space with a quantum score module.

Advantages: more realistic for simulator constraints and likely better sample
quality.

Risk: novelty depends on the exact latent diffusion formulation and comparisons.

## Initial Recommendation

Start with Architecture C as the main path and Architecture A as a fallback:

1. Use a small autoencoder or deterministic downsampling/PCA to obtain a compact
   latent representation.
2. Train a score/noise predictor in latent space.
3. Replace the final or intermediate score block with a parameterized quantum
   circuit.
4. Compare with a parameter-matched classical MLP score network.

This provides a credible reason for the quantum component: simulators cannot
handle full images directly, and latent diffusion makes the quantum bottleneck a
central design constraint rather than an afterthought.

## Experimental Plan

### Datasets

- MNIST.
- Fashion-MNIST.
- Optional sanity dataset: Bars-and-Stripes or synthetic geometric images.

### Baselines

- Compact classical DDPM or score model on the same latent representation.
- Classical MLP with matched parameter count.
- Optional VAE or autoencoder sampling baseline.

### Proposed Models

- HQ-LDM-A: hybrid quantum latent diffusion with quantum score head.
- HQ-LDM-B: hybrid quantum latent diffusion with quantum time-conditioning.
- Classical-LDM: identical pipeline without quantum layer.

### Metrics

- FID or FID-like metric where statistically meaningful.
- KID for small-sample robustness.
- Classifier-based sample recognizability for MNIST/Fashion-MNIST.
- Diversity metrics, such as label entropy from a pretrained classifier.
- Parameter count and wall-clock training/inference time.
- Qualitative grids of generated samples.

### Ablations

- Number of qubits: 4, 6, 8 if feasible.
- Circuit depth: 1, 2, 4.
- Encoding: angle encoding vs amplitude encoding if supported.
- Ansatz: hardware-efficient vs strongly entangling layers.
- Module placement: score head vs time-conditioning.

## Validity Criteria

- All baselines must share the same train/test split and preprocessing.
- Classical baselines must be parameter-matched or compute-matched where
  possible.
- Random seeds must be reported.
- Failed or negative results must be preserved and discussed.
- Claims must distinguish "simulator evidence" from "hardware feasibility".

## Main Risks

1. Literature overlap: closely related quantum diffusion image models may already
   exist.
2. Simulator cost: quantum circuit gradients may make experiments slow.
3. Quality gap: classical baselines may outperform the hybrid model.
4. Metric instability: FID on low-resolution grayscale datasets can be noisy.
5. Novelty dilution: a quantum layer in a classical network may not be enough
   unless its role is tightly argued and ablated.

## Stage 1 Devil's Advocate Check

### Verdict

PASS WITH MAJOR WARNINGS.

### Major Issues

1. The phrase "new in the literature" is not yet supportable. It must become a
   specific architectural novelty claim after systematic source screening.
2. The quantum contribution could be superficial unless the circuit placement is
   theoretically motivated and experimentally ablated.
3. Generation quality may be worse than classical baselines; the paper must be
   designed to survive a negative or mixed result by emphasizing when and why the
   hybrid model behaves differently.

### Strongest Counter-argument

A reviewer may argue that the proposed model is a classical latent diffusion
pipeline with a small quantum layer that adds computational overhead without
demonstrating a principled advantage.

### Response Strategy

Frame the paper around architectural analysis and simulator-grounded evidence,
not quantum advantage. The contribution should be a reproducible benchmark and
specific hybrid design, with honest limitations.

## Immediate Next Steps

1. Build a verified literature matrix for quantum diffusion and quantum
   generative image models.
2. Select the exact architecture after identifying the closest prior work.
3. Scaffold code for datasets, baselines, quantum modules, training, metrics,
   and plots.
4. Run a tiny end-to-end smoke experiment on a synthetic dataset.
5. Scale to MNIST and Fashion-MNIST.
6. Draft the LaTeX paper after the first reproducible result exists.
