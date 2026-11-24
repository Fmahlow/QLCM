from __future__ import annotations

import math

import torch
from torch import nn

from src.qlcm.quantum_ops import apply_single_qubit, cnot_permutation, rot_gate, rx_gate


class QDenseDiffusionCircuit(nn.Module):
    """Reproduction of the Q-Dense architecture from Koelle et al. 2024,
    "Quantum Denoising Diffusion Models" (arXiv:2401.07049), Sections 3.1 and
    3.3: their guided MNIST-8x8 configuration amplitude-embeds a flattened
    image into 6 qubits, adds one ancilla qubit rotated by the class label
    for guidance (giving 7 qubits total, matching their reported
    configuration), and applies a stack of "dense" (strongly entangling)
    layers -- one arbitrary single-qubit rotation per qubit per layer plus a
    ring of CNOTs -- before reading out joint probabilities of the image
    qubits (ancilla traced out) rescaled by the input's Euclidean norm. With
    47 layers x 7 qubits x 3 rotation angles this reaches 987 trainable
    parameters, matching their reported ~1000-parameter guided model.

    Unlike QuantumScoreHead/QuantumLatentTeacherScoreNet elsewhere in this
    codebase (which predict a signed noise vector), this circuit's readout is
    a probability distribution and therefore non-negative by construction.
    This is why Koelle et al. train it to predict x_{t-1} (a non-negative
    pixel image) directly rather than the signed noise added at each step;
    we follow that choice in scripts/train_qdense_baseline.py.

    The same weights are applied at every diffusion step (Fig. 1 shows a
    single reused circuit iteratively refining the image); the source paper
    does not describe an explicit timestep-conditioning input to this
    circuit, so we do not add one.
    """

    def __init__(self, n_image_qubits: int = 6, n_layers: int = 47, n_classes: int = 2):
        super().__init__()
        self.n_image_qubits = n_image_qubits
        self.n_qubits = n_image_qubits + 1
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, self.n_qubits, 3))
        perms = torch.stack(
            [cnot_permutation(self.n_qubits, wire, (wire + 1) % self.n_qubits) for wire in range(self.n_qubits)]
        )
        self.register_buffer("cnot_perms", perms)

    def forward(self, pixels: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        batch, n_pixels = pixels.shape
        if n_pixels != 2**self.n_image_qubits:
            raise ValueError(f"expected {2 ** self.n_image_qubits} pixels, got {n_pixels}")
        norm = pixels.norm(dim=1, keepdim=True).clamp_min(1e-8)
        amplitudes = (pixels / norm).to(torch.complex64)
        ancilla = torch.zeros(batch, 2, device=pixels.device, dtype=torch.complex64)
        ancilla[:, 0] = 1.0
        # image amplitudes occupy the more-significant bits, ancilla the LSB,
        # so probs.view(batch, 2**n_image_qubits, 2).sum(-1) below marginalizes it.
        state = torch.einsum("bi,bj->bij", amplitudes, ancilla).reshape(batch, -1)

        class_angle = class_idx.to(pixels.dtype) * (2 * math.pi / self.n_classes)
        state = apply_single_qubit(state, rx_gate(class_angle), self.n_qubits - 1, self.n_qubits)

        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                phi, theta, omega = self.weights[layer, wire]
                gate = rot_gate(phi.expand(batch), theta.expand(batch), omega.expand(batch))
                state = apply_single_qubit(state, gate, wire, self.n_qubits)
            for wire in range(self.n_qubits):
                state = state[:, self.cnot_perms[wire]]

        probs = state.real.square() + state.imag.square()
        image_probs = probs.view(batch, 2**self.n_image_qubits, 2).sum(dim=2)
        # Rescale the (small, since probabilities sum to 1 over 2**n_image_qubits
        # outcomes) probability vector back to the input's norm, not merely
        # multiply by it -- multiplying alone leaves the output norm equal to
        # norm(image_probs) * norm(pixels), which is far smaller than
        # norm(pixels) whenever the distribution isn't sharply peaked, causing
        # the signal to collapse geometrically toward zero under the repeated
        # (recurrent, same-weights) application this architecture requires.
        probs_norm = image_probs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return image_probs * (norm / probs_norm)


def forward_chain(x0: torch.Tensor, timesteps: int, sigma_min: float = 0.05, sigma_max: float = 0.6) -> torch.Tensor:
    """Build an explicit forward noising trajectory x0, x1, ..., x_timesteps
    by repeatedly adding Gaussian noise of increasing scale and clamping to
    [0, 1], so every x_t stays a valid non-negative pixel image (required for
    amplitude embedding and for the probability-valued readout). The source
    paper says it follows Ho et al. (2020) DDPM methodology but does not
    give the exact schedule used for their direct x_{t-1}-prediction loss
    (Sec. 4.4); this linear noise-scale schedule is our interpretation,
    documented as such in experiments/execution_checklist.md.

    Returns a (timesteps + 1, batch, pixels) tensor; index t gives x_t.
    """
    sigmas = torch.linspace(sigma_min, sigma_max, timesteps, device=x0.device)
    chain = [x0]
    x = x0
    for sigma in sigmas:
        x = (x + sigma * torch.randn_like(x)).clamp(0.0, 1.0)
        chain.append(x)
    return torch.stack(chain, dim=0)
