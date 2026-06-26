from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image

from src.data import BarsAndStripes
from src.diffusion import DiffusionSchedule, sample_ddpm
from src.models import ClassicalScoreNet, HybridQuantumScoreNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classical", "quantum"], default="classical")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--timesteps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results/smoke")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_grid(images: torch.Tensor, path: Path):
    images = ((images[:, 0] + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
    nrow = 8
    padding = 2
    tile_h, tile_w = images.shape[1:]
    grid_h = nrow * tile_h + (nrow - 1) * padding
    grid_w = nrow * tile_w + (nrow - 1) * padding
    grid = np.full((grid_h, grid_w), 255, dtype=np.uint8)
    for idx, image in enumerate(images[: nrow * nrow]):
        row = idx // nrow
        col = idx % nrow
        y = row * (tile_h + padding)
        x = col * (tile_w + padding)
        grid[y : y + tile_h, x : x + tile_w] = image
    Image.fromarray(grid, mode="L").save(path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = BarsAndStripes(n_samples=args.samples, size=8, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    schedule = DiffusionSchedule(timesteps=args.timesteps, device=str(device))

    if args.model == "classical":
        model = ClassicalScoreNet(image_size=8).to(device)
    else:
        model = HybridQuantumScoreNet(image_size=8, n_qubits=4, n_layers=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    losses = []

    model.train()
    for epoch in range(args.epochs):
        for x0, _ in loader:
            x0 = x0.to(device)
            t = schedule.sample_timesteps(x0.shape[0])
            xt, noise = schedule.add_noise(x0, t)
            pred = model(xt, t)
            loss = loss_fn(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch + 1} loss={sum(losses[-len(loader):]) / len(loader):.6f}")

    generated = sample_ddpm(model, schedule, (64, 1, 8, 8), device=str(device))
    save_grid(generated, output_dir / f"{args.model}_samples.png")

    torch.save(model.state_dict(), output_dir / f"{args.model}_model.pt")
    with (output_dir / f"{args.model}_metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"model={args.model}\n")
        handle.write(f"parameters={count_parameters(model)}\n")
        handle.write(f"final_loss={losses[-1]:.8f}\n")
        handle.write(f"epochs={args.epochs}\n")
        handle.write(f"timesteps={args.timesteps}\n")


if __name__ == "__main__":
    main()
