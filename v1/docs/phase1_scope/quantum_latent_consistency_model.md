# Quantum Latent Consistency Model: Architecture Pivot

## Motivation

The initial hybrid quantum latent diffusion idea is plausible but architecturally
close to existing quantum denoising diffusion work. A stronger contribution is to
transfer the consistency-model principle from classical diffusion into a
simulator-based hybrid quantum architecture.

Working name:

**QLCM: Quantum Latent Consistency Model**

## Core Claim Candidate

QLCM studies whether a parameterized quantum circuit can serve as a compact
latent consistency function that maps noisy latent states at arbitrary diffusion
times directly toward the clean latent endpoint, enabling one-step or few-step
generation in low-dimensional image manifolds.

This differs from ordinary quantum diffusion models because the student model is
not trained merely to predict per-step noise. It is trained to satisfy
trajectory consistency: predictions from different points on the same reverse
trajectory should agree on the same clean latent target.

## Classical Source Idea

Consistency models learn a function that maps any point along a probability-flow
ODE trajectory to the same data sample. Latent Consistency Models apply this in
the latent space of pretrained latent diffusion models, enabling few-step
generation.

QLCM adapts this principle to quantum generative modeling at small scale:

- train or define a latent representation for low-resolution images;
- train a classical teacher score/diffusion model in that latent space;
- distill the teacher trajectory into a hybrid quantum consistency student;
- generate samples with one or a few student evaluations.

## Proposed Architecture

### Components

1. **Latent encoder/decoder**
   - Small autoencoder or deterministic compression.
   - Input: MNIST/Fashion-MNIST image.
   - Output: compact latent vector.

2. **Classical teacher**
   - A small latent diffusion or score model.
   - Provides probability-flow trajectory pairs for distillation.

3. **Quantum consistency student**
   - Receives noisy latent vector `z_t` and time `t`.
   - Encodes features into a parameterized quantum circuit.
   - Outputs either:
     - clean latent estimate `z_0_hat`, or
     - residual correction to a classical consistency head.

4. **Few-step sampler**
   - One-step: `z_T -> z_0_hat`.
   - Few-step: `z_T -> z_t1 -> z_0_hat`, with optional noise injection.

## Candidate Variants

### QLCM-A: Pure Quantum Consistency Head

Classical feature compressor -> quantum circuit -> clean latent prediction.

Best as a minimal proof of concept.

### QLCM-B: Quantum Residual Consistency

Classical consistency network predicts a baseline clean latent. The quantum
circuit predicts a residual correction.

Best for generation quality and stability.

### QLCM-C: Quantum Time-Conditioned Consistency

The quantum circuit encodes timestep/noise-level information and modulates a
classical latent consistency network.

Best for isolating whether quantum circuits help with trajectory conditioning.

## Losses

### Distillation Loss

For teacher trajectory pair `(z_t, z_s)` with `s < t`, train:

`|| f_theta(z_t, t) - stopgrad(f_theta(z_s, s)) ||^2`

with boundary condition:

`f_theta(z_epsilon, epsilon) = z_epsilon`

### Teacher-Target Loss

When the clean latent `z_0` is known:

`|| f_theta(z_t, t) - z_0 ||^2`

### Hybrid Objective

Use a weighted sum:

`L = L_consistency + lambda * L_target`

## Why This Is More Distinctive

Compared with a quantum DDPM-style model, QLCM changes the learning target:

- from iterative denoising/noise prediction;
- to trajectory-consistent direct latent prediction.

Compared with existing classical LCMs, QLCM changes the function class:

- part of the consistency map is represented by a parameterized quantum circuit;
- experiments test whether this compact circuit is useful under low-dimensional
  latent constraints.

## Main Experimental Questions

1. Can QLCM generate recognizable MNIST/Fashion-MNIST samples in one or few
   steps?
2. Does QLCM outperform or match parameter-matched classical consistency
   students?
3. Does the quantum module help more as a direct consistency head, residual
   corrector, or time-conditioning module?
4. How do qubit count, circuit depth, and encoding affect consistency error and
   sample quality?

## Recommended Pivot

Make QLCM the main contribution and keep hybrid quantum latent diffusion as the
teacher/baseline infrastructure.

The paper's architectural contribution becomes:

> We introduce and evaluate a hybrid quantum latent consistency model that
> distills a latent diffusion teacher into a parameterized quantum consistency
> student for one-step and few-step synthetic image generation.

This is stronger than claiming a generic quantum diffusion architecture because
the consistency objective is a clearer transfer from modern classical diffusion
research that appears underexplored in quantum image generation.
