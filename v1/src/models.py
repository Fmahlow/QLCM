from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class ClassicalScoreNet(nn.Module):
    def __init__(self, image_size: int = 8, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()
        pixels = image_size * image_size
        self.image_size = image_size
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(pixels + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pixels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.view(batch, -1)
        features = torch.cat([flat, self.time(t)], dim=1)
        return self.net(features).view_as(x)


class QuantumScoreHead(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ry_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))
        self.rz_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits))
        basis = torch.arange(2**n_qubits)
        z_signs = []
        cz_masks = []
        for wire in range(n_qubits):
            bit = (basis >> (n_qubits - wire - 1)) & 1
            z_signs.append(torch.where(bit == 0, 1.0, -1.0))
            next_wire = (wire + 1) % n_qubits
            next_bit = (basis >> (n_qubits - next_wire - 1)) & 1
            cz_masks.append(torch.where((bit == 1) & (next_bit == 1), -1.0, 1.0))
        self.register_buffer("z_signs", torch.stack(z_signs))
        self.register_buffer("cz_masks", torch.stack(cz_masks))

    def _apply_single_qubit(self, state: torch.Tensor, gate: torch.Tensor, wire: int):
        batch = state.shape[0]
        shape = [batch] + [2] * self.n_qubits
        tensor = state.reshape(shape)
        perm = [0] + [axis for axis in range(1, self.n_qubits + 1) if axis != wire + 1] + [wire + 1]
        inv_perm = [0] * len(perm)
        for idx, axis in enumerate(perm):
            inv_perm[axis] = idx
        tensor = tensor.permute(perm).reshape(batch, -1, 2)
        tensor = torch.einsum("bij,bkj->bki", gate, tensor)
        tensor = tensor.reshape([batch] + [2] * (self.n_qubits - 1) + [2]).permute(inv_perm)
        return tensor.reshape(batch, 2**self.n_qubits)

    def _ry(self, theta: torch.Tensor) -> torch.Tensor:
        half = theta / 2.0
        cos = torch.cos(half)
        sin = torch.sin(half)
        gate = torch.zeros(theta.shape[0], 2, 2, device=theta.device, dtype=torch.complex64)
        gate[:, 0, 0] = cos.to(torch.complex64)
        gate[:, 0, 1] = -sin.to(torch.complex64)
        gate[:, 1, 0] = sin.to(torch.complex64)
        gate[:, 1, 1] = cos.to(torch.complex64)
        return gate

    def _rz(self, theta: torch.Tensor) -> torch.Tensor:
        phase = theta / 2.0
        gate = torch.zeros(theta.shape[0], 2, 2, device=theta.device, dtype=torch.complex64)
        gate[:, 0, 0] = torch.exp(-1j * phase).to(torch.complex64)
        gate[:, 1, 1] = torch.exp(1j * phase).to(torch.complex64)
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        state = torch.zeros(batch, 2**self.n_qubits, device=x.device, dtype=torch.complex64)
        state[:, 0] = 1.0 + 0.0j

        for wire in range(self.n_qubits):
            state = self._apply_single_qubit(state, self._ry(x[:, wire]), wire)
            state = self._apply_single_qubit(state, self._rz(x[:, wire + self.n_qubits]), wire)

        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                theta_y = self.ry_weights[layer, wire].expand(batch)
                theta_z = self.rz_weights[layer, wire].expand(batch)
                state = self._apply_single_qubit(state, self._ry(theta_y), wire)
                state = self._apply_single_qubit(state, self._rz(theta_z), wire)
            for wire in range(self.n_qubits):
                state = state * self.cz_masks[wire].to(device=x.device, dtype=torch.complex64)

        probabilities = state.real.square() + state.imag.square()
        return probabilities @ self.z_signs.to(device=x.device).T


class HybridQuantumScoreNet(nn.Module):
    def __init__(
        self,
        image_size: int = 8,
        hidden_dim: int = 128,
        time_dim: int = 32,
        n_qubits: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        pixels = image_size * image_size
        self.image_size = image_size
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.encoder = nn.Sequential(
            nn.Linear(pixels + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * n_qubits),
            nn.Tanh(),
        )
        self.quantum = QuantumScoreHead(n_qubits=n_qubits, n_layers=n_layers)
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pixels),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        time_emb = self.time(t)
        flat = x.view(batch, -1)
        q_inputs = math.pi * self.encoder(torch.cat([flat, time_emb], dim=1))
        q_features = self.quantum(q_inputs)
        out = self.decoder(torch.cat([q_features, time_emb], dim=1))
        return out.view_as(x)
