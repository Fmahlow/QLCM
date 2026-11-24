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
from src.metrics import count_parameters
from src.qlcm.autoencoder import ConvAutoencoder, MlpAutoencoder
from src.qlcm.consistency import ClassicalConsistencyModel, QuantumLatentConsistencyModel
from src.qlcm.ema import EMATarget
from src.qlcm.losses import combined_qlcm_loss
from src.qlcm.teacher import LatentNoiseSchedule, LatentTeacherScoreNet
from src.training import ConvergenceTracker


def build_autoencoder(cfg, device):
    if cfg["image_size"] == 8:
        return MlpAutoencoder(image_size=8, latent_dim=cfg["latent_dim"]).to(device)
    return ConvAutoencoder(latent_dim=cfg["latent_dim"]).to(device)


def build_student(cfg, placement, hidden_dim, device):
    if placement == "classical":
        return ClassicalConsistencyModel(
            latent_dim=cfg["latent_dim"],
            hidden_dim=hidden_dim,
            time_dim=cfg["model"]["time_dim"],
        ).to(device)
    return QuantumLatentConsistencyModel(
        latent_dim=cfg["latent_dim"],
        hidden_dim=hidden_dim,
        time_dim=cfg["model"]["time_dim"],
        n_qubits=cfg["model"]["qubits"],
        n_layers=cfg["model"]["depth"],
        placement=placement,
    ).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--placement", choices=["classical", "head", "residual", "time"], default=None)
    parser.add_argument("--qubits", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Overrides model.hidden_dim; used to hit a target total-parameter "
        "point for the sub-/equal-/over-parametrized curve (docs/research_design.md, Study 1).",
    )
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--autoencoder", default=None)
    parser.add_argument("--teacher", required=True)
    parser.add_argument(
        "--target",
        choices=["ema", "self"],
        default="ema",
        help="'ema' (default, v2 methodology fix) uses an EMA target network for the "
        "consistency loss; 'self' reproduces v1's approximation (online student as its "
        "own target) for direct before/after comparison only.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-epochs", type=int, default=300, help="Shared hard cap (docs/research_design.md).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.qubits is not None:
        cfg["model"]["qubits"] = args.qubits
    if args.depth is not None:
        cfg["model"]["depth"] = args.depth
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else cfg["model"]["hidden_dim"]
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
    teacher.load_state_dict(torch.load(args.teacher, map_location=device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = build_student(cfg, placement, hidden_dim, device)
    target_model = EMATarget(student, decay=args.ema_decay).to(device) if args.target == "ema" else student
    schedule = LatentNoiseSchedule(timesteps=cfg["timesteps"], device=str(device))
    if args.dry_run:
        x, _ = next(iter(loader))
        with torch.no_grad():
            z0 = autoencoder.encode(x.to(device))
        loss, parts = combined_qlcm_loss(student, target_model, teacher, z0, schedule)
        print({"placement": placement, "hidden_dim": hidden_dim, "student_parameters": count_parameters(student), "loss": float(loss.detach().cpu()), **parts})
        return
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    tracker = ConvergenceTracker(max_epochs=args.max_epochs)
    while True:
        losses = []
        for x, _ in loader:
            with torch.no_grad():
                z0 = autoencoder.encode(x.to(device))
            loss, _ = combined_qlcm_loss(student, target_model, teacher, z0, schedule)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.target == "ema":
                target_model.update(student)
            losses.append(float(loss.item()))
        epoch_loss = sum(losses) / len(losses)
        print(f"epoch={tracker.epochs_run + 1} qlcm_loss={epoch_loss:.6f}")
        if tracker.update(epoch_loss):
            break
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.run_id or placement
    torch.save(student.state_dict(), output_dir / f"student_{suffix}.pt")
    metadata = {
        "run_id": args.run_id,
        "placement": placement,
        "qubits": cfg["model"]["qubits"] if placement != "classical" else 0,
        "depth": cfg["model"]["depth"] if placement != "classical" else 0,
        "hidden_dim": hidden_dim,
        "student_parameters": count_parameters(student),
        "target": args.target,
        "ema_decay": args.ema_decay if args.target == "ema" else None,
        "sampling_steps": args.sampling_steps,
        "latent_dim": cfg["latent_dim"],
        "seed": seed,
        "epochs_run": tracker.epochs_run,
        "hit_cap": tracker.hit_cap,
        "final_loss": tracker.history[-1],
    }
    (output_dir / f"student_{suffix}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
