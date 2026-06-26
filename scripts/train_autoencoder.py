from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import load_config
from src.datasets import build_dataset
from src.qlcm.autoencoder import ConvAutoencoder, MlpAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cpu"))
    dataset = build_dataset(cfg["dataset"], max_samples=cfg.get("max_samples"))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    if cfg["image_size"] == 8:
        model = MlpAutoencoder(image_size=8, latent_dim=cfg["latent_dim"]).to(device)
    else:
        model = ConvAutoencoder(latent_dim=cfg["latent_dim"]).to(device)
    if args.dry_run:
        x, _ = next(iter(loader))
        recon, z = model(x.to(device))
        print({"recon_shape": tuple(recon.shape), "latent_shape": tuple(z.shape)})
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(cfg["epochs"]["autoencoder"]):
        losses = []
        for x, _ in loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = F.mse_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch + 1} reconstruction_mse={sum(losses) / len(losses):.6f}")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "autoencoder.pt")


if __name__ == "__main__":
    main()
