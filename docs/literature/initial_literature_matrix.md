# Initial Literature Matrix

This file records the first-pass literature constraints for the project. It is
not yet a systematic review.

## Closest Prior Work

| Work | What It Does | Relevance | Risk to Our Novelty Claim |
| --- | --- | --- | --- |
| Kolle et al., "Quantum Denoising Diffusion Models" (2024), https://arxiv.org/abs/2401.07049 | Introduces Q-Dense and QU-Net quantum diffusion architectures for image generation; benchmarks MNIST, Fashion-MNIST, and CIFAR-10; reports FID, SSIM, and PSNR. | Directly overlaps with simulator-based quantum diffusion for image generation. | High. We cannot claim first quantum diffusion image model. |
| Zhang et al., "Generative quantum machine learning via denoising diffusion probabilistic models" (2023/2024), https://arxiv.org/abs/2310.05866 | Introduces QuDDPM for quantum data generation with intermediate training tasks to reduce barren plateau issues. | Important conceptual foundation for quantum diffusion, but focused on quantum data rather than classical images. | Medium. It constrains terminology and theory positioning. |
| Kwun et al., "Mixed-State Quantum Denoising Diffusion Probabilistic Model" (2024), https://arxiv.org/abs/2411.17608 | Proposes mixed-state QuDDPM with depolarizing forward channels, parameterized quantum circuits, projective measurements, and superfidelity costs. | Relevant for quantum-noise-channel diffusion theory, mostly quantum ensemble/state generation. | Medium. Useful contrast for our classical-image focus. |
| Chen et al., "Quantum Generative Models for Image Generation: Insights from MNIST and MedMNIST" (2025), https://arxiv.org/abs/2504.00034 | Hybrid variational quantum model for image generation with quantum-generated noise and noise scheduling; evaluates MNIST and MedMNIST. | Overlaps with hybrid quantum image generation and scheduled noise. | Medium to high. Need inspect details before claiming novelty around noise scheduling. |

## Implications

The contribution should avoid broad novelty claims such as:

- "the first quantum diffusion model";
- "the first quantum diffusion model for image generation";
- "the first hybrid quantum image generator with noise scheduling".

More defensible candidate claims:

- a latent-space hybrid quantum score model with controlled parameter-matched
  baselines;
- a quantum time-conditioning mechanism inside latent diffusion;
- an empirical study isolating where quantum circuit placement matters in
  simulator-based image diffusion;
- a reproducible benchmark protocol for small-data quantum diffusion image
  generation.

## Working Gap Hypothesis

Existing work appears to focus on direct image-space quantum denoising circuits,
quantum U-Net/convolution designs, or quantum-state diffusion. A plausible gap is
to study a latent diffusion formulation where the quantum circuit is a compact
score/time-conditioning module and where the main experimental question is not
"does it beat all classical diffusion models?" but "under matched small-model
constraints, does the quantum module provide useful inductive bias or parameter
efficiency?"

This hypothesis must be validated by deeper source screening before the paper
uses it as a contribution claim.

## Source Notes

- Kolle et al. report two quantum diffusion architectures and evaluate MNIST,
  Fashion-MNIST, and CIFAR-10 with FID, SSIM, and PSNR.
- Zhang et al. frame QuDDPM as denoising diffusion for quantum data rather than
  classical images.
- Kwun et al. remove a high-fidelity scrambling-unitary requirement by using
  mixed-state diffusion with depolarizing channels and parameterized circuits.
- Chen et al. connects hybrid quantum image generation with quantum-generated
  noise and time-step noise scheduling.
