from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.datasets import build_dataset
from src.qlcm.autoencoder import ConvAutoencoder, MlpAutoencoder
from src.qlcm.teacher import LatentNoiseSchedule, LatentTeacherScoreNet
from src.training import ConvergenceTracker


def build_autoencoder(cfg, device):
    if cfg["image_size"] == 8:
        return MlpAutoencoder(image_size=8, latent_dim=cfg["latent_dim"]).to(device)
    return ConvAutoencoder(latent_dim=cfg["latent_dim"]).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--autoencoder", default=None)
    parser.add_argument("--max-epochs", type=int, default=300, help="Shared hard cap (docs/research_design.md).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cpu"))
    dataset = build_dataset(cfg["dataset"], max_samples=cfg.get("max_samples"))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    autoencoder = build_autoencoder(cfg, device)
    if args.autoencoder:
        autoencoder.load_state_dict(torch.load(args.autoencoder, map_location=device))
    autoencoder.eval()
    teacher = LatentTeacherScoreNet(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)
    schedule = LatentNoiseSchedule(timesteps=cfg["timesteps"], device=str(device))
    if args.dry_run:
        x, _ = next(iter(loader))
        with torch.no_grad():
            z0 = autoencoder.encode(x.to(device))
        t = schedule.sample_timesteps(z0.shape[0])
        z_t, noise = schedule.add_noise(z0, t)
        pred = teacher(z_t, t)
        print({"z_shape": tuple(z0.shape), "pred_shape": tuple(pred.shape), "noise_shape": tuple(noise.shape)})
        return
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-3)
    tracker = ConvergenceTracker(max_epochs=args.max_epochs)
    while True:
        losses = []
        for x, _ in loader:
            with torch.no_grad():
                z0 = autoencoder.encode(x.to(device))
            t = schedule.sample_timesteps(z0.shape[0])
            z_t, noise = schedule.add_noise(z0, t)
            pred = teacher(z_t, t)
            loss = F.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_loss = sum(losses) / len(losses)
        print(f"epoch={tracker.epochs_run + 1} teacher_noise_mse={epoch_loss:.6f}")
        if tracker.update(epoch_loss):
            break
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(teacher.state_dict(), output_dir / "teacher.pt")
    (output_dir / "teacher_training_log.json").write_text(
        json.dumps({"epochs_run": tracker.epochs_run, "hit_cap": tracker.hit_cap, "final_loss": tracker.history[-1]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
