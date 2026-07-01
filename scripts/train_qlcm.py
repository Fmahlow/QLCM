from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.datasets import build_dataset
from src.qlcm.autoencoder import ConvAutoencoder, MlpAutoencoder
from src.qlcm.consistency import ClassicalConsistencyModel, QuantumLatentConsistencyModel
from src.qlcm.losses import combined_qlcm_loss
from src.qlcm.teacher import LatentNoiseSchedule, LatentTeacherScoreNet


def build_autoencoder(cfg, device):
    if cfg["image_size"] == 8:
        return MlpAutoencoder(image_size=8, latent_dim=cfg["latent_dim"]).to(device)
    return ConvAutoencoder(latent_dim=cfg["latent_dim"]).to(device)


def build_student(cfg, placement, device):
    if placement == "classical":
        return ClassicalConsistencyModel(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            time_dim=cfg["model"]["time_dim"],
        ).to(device)
    return QuantumLatentConsistencyModel(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        time_dim=cfg["model"]["time_dim"],
        n_qubits=cfg["model"]["qubits"],
        n_layers=cfg["model"]["depth"],
        placement=placement,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--placement", choices=["classical", "head", "residual", "time"], default=None)
    parser.add_argument("--qubits", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--autoencoder", default=None)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.qubits is not None:
        cfg["model"]["qubits"] = args.qubits
    if args.depth is not None:
        cfg["model"]["depth"] = args.depth
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    torch.manual_seed(seed)
    placement = args.placement or cfg["model"].get("placement", "residual")
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
    if args.teacher:
        teacher.load_state_dict(torch.load(args.teacher, map_location=device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = build_student(cfg, placement, device)
    schedule = LatentNoiseSchedule(timesteps=cfg["timesteps"], device=str(device))
    if args.dry_run:
        x, _ = next(iter(loader))
        with torch.no_grad():
            z0 = autoencoder.encode(x.to(device))
        loss, parts = combined_qlcm_loss(student, teacher, z0, schedule)
        print({"placement": placement, "loss": float(loss.detach().cpu()), **parts})
        return
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    for epoch in range(cfg["epochs"]["consistency"]):
        losses = []
        for x, _ in loader:
            with torch.no_grad():
                z0 = autoencoder.encode(x.to(device))
            loss, _ = combined_qlcm_loss(student, teacher, z0, schedule)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch + 1} qlcm_loss={sum(losses) / len(losses):.6f}")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.run_id or placement
    torch.save(student.state_dict(), output_dir / f"student_{suffix}.pt")
    metadata = {
        "run_id": args.run_id,
        "placement": placement,
        "qubits": cfg["model"]["qubits"],
        "depth": cfg["model"]["depth"],
        "sampling_steps": args.sampling_steps,
        "latent_dim": cfg["latent_dim"],
        "seed": seed,
    }
    (output_dir / f"student_{suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
