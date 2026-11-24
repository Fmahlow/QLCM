from __future__ import annotations

import torch


def apply_single_qubit(state: torch.Tensor, gate: torch.Tensor, wire: int, n_qubits: int) -> torch.Tensor:
    """Apply a batched 2x2 gate to `wire` of an n_qubits statevector batch.

    `state` is (batch, 2**n_qubits); wire 0 is the most significant bit,
    matching the convention used throughout src/qlcm (e.g. QuantumScoreHead).
    """
    batch = state.shape[0]
    shape = [batch] + [2] * n_qubits
    tensor = state.reshape(shape)
    perm = [0] + [axis for axis in range(1, n_qubits + 1) if axis != wire + 1] + [wire + 1]
    inv_perm = [0] * len(perm)
    for idx, axis in enumerate(perm):
        inv_perm[axis] = idx
    tensor = tensor.permute(perm).reshape(batch, -1, 2)
    tensor = torch.einsum("bij,bkj->bki", gate, tensor)
    tensor = tensor.reshape([batch] + [2] * (n_qubits - 1) + [2]).permute(inv_perm)
    return tensor.reshape(batch, 2**n_qubits)


def rz_gate(theta: torch.Tensor) -> torch.Tensor:
    phase = theta / 2.0
    gate = torch.zeros(theta.shape[0], 2, 2, device=theta.device, dtype=torch.complex64)
    gate[:, 0, 0] = torch.exp(-1j * phase).to(torch.complex64)
    gate[:, 1, 1] = torch.exp(1j * phase).to(torch.complex64)
    return gate


def ry_gate(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2.0
    cos, sin = torch.cos(half), torch.sin(half)
    gate = torch.zeros(theta.shape[0], 2, 2, device=theta.device, dtype=torch.complex64)
    gate[:, 0, 0] = cos.to(torch.complex64)
    gate[:, 0, 1] = -sin.to(torch.complex64)
    gate[:, 1, 0] = sin.to(torch.complex64)
    gate[:, 1, 1] = cos.to(torch.complex64)
    return gate


def rx_gate(theta: torch.Tensor) -> torch.Tensor:
    half = theta / 2.0
    cos, sin = torch.cos(half), torch.sin(half)
    gate = torch.zeros(theta.shape[0], 2, 2, device=theta.device, dtype=torch.complex64)
    gate[:, 0, 0] = cos.to(torch.complex64)
    gate[:, 0, 1] = (-1j * sin).to(torch.complex64)
    gate[:, 1, 0] = (-1j * sin).to(torch.complex64)
    gate[:, 1, 1] = cos.to(torch.complex64)
    return gate


def rot_gate(phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """PennyLane's Rot(phi, theta, omega) = RZ(omega) @ RY(theta) @ RZ(phi),
    the general single-qubit rotation used by StronglyEntanglingLayers, the
    ansatz Koelle et al. 2024 (arXiv:2401.07049) build Q-Dense on."""
    g1 = rz_gate(phi)
    g2 = ry_gate(theta)
    g3 = rz_gate(omega)
    return torch.bmm(g3, torch.bmm(g2, g1))


def cnot_permutation(n_qubits: int, control: int, target: int, device=None) -> torch.Tensor:
    """Index permutation implementing CNOT(control, target) on a flat
    2**n_qubits statevector: CNOT only permutes basis amplitudes, so it can
    be applied as a single gather instead of a permute+einsum."""
    basis = torch.arange(2**n_qubits, device=device)
    pos_c = n_qubits - control - 1
    pos_t = n_qubits - target - 1
    control_bit = (basis >> pos_c) & 1
    flipped = basis ^ (1 << pos_t)
    return torch.where(control_bit == 1, flipped, basis)
