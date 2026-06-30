from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from src.config import load_config
from src.eval_metrics import evaluate_generated_samples, real_reference_features, train_or_load_classifier
from src.metrics import count_parameters, time_call
from src.qlcm.autoencoder import ConvAutoencoder, MlpAutoencoder
from src.qlcm.consistency import ClassicalConsistencyModel, QuantumLatentConsistencyModel
from src.qlcm.sampling import sample_consistency


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


def save_grid(images: torch.Tensor, path: Path):
    images = ((images[:, 0] + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
    nrow = int(np.ceil(np.sqrt(images.shape[0])))
    padding = 2
    tile_h, tile_w = images.shape[1:]
    grid = np.full(
        (nrow * tile_h + (nrow - 1) * padding, nrow * tile_w + (nrow - 1) * padding),
        255,
        dtype=np.uint8,
    )
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        y = row * (tile_h + padding)
        x = col * (tile_w + padding)
        grid[y : y + tile_h, x : x + tile_w] = image
    Image.fromarray(grid, mode="L").save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--placement", choices=["classical", "head", "residual", "time"], default="residual")
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--qubits", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.qubits is not None:
        cfg["model"]["qubits"] = args.qubits
    if args.depth is not None:
        cfg["model"]["depth"] = args.depth
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    torch.manual_seed(seed)
    device = torch.device(cfg.get("device", "cpu"))
    autoencoder = build_autoencoder(cfg, device)
    student = build_student(cfg, args.placement, device)
    autoencoder.load_state_dict(torch.load(args.autoencoder, map_location=device))
    student.load_state_dict(torch.load(args.student, map_location=device))
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    images, elapsed = time_call(
        sample_consistency,
        student,
        autoencoder,
        cfg["latent_dim"],
        args.samples,
        args.steps,
        cfg["timesteps"],
        str(device),
    )
    tag = args.run_id or f"{args.placement}_{args.steps}step"
    sample_path = output_dir / f"samples_{tag}.png"
    save_grid(images, sample_path)
    metrics = {
        "run_id": args.run_id,
        "seed": seed,
        "placement": args.placement,
        "qubits": cfg["model"].get("qubits") if args.placement != "classical" else 0,
        "depth": cfg["model"].get("depth") if args.placement != "classical" else 0,
        "sampling_steps": args.steps,
        "samples": args.samples,
        "student_parameters": count_parameters(student),
        "sampling_seconds": elapsed,
        "sample_grid": str(sample_path),
    }

    if cfg["dataset"] in {"mnist", "fashion_mnist"}:
        classifier_cache = Path(cfg["output_dir"]) / f"{cfg['dataset']}_classifier.pt"
        classifier = train_or_load_classifier(cfg["dataset"], device, str(classifier_cache))
        real_features_cache = Path(cfg["output_dir"]) / f"{cfg['dataset']}_real_features.npy"
        if real_features_cache.exists():
            real_features = np.load(real_features_cache)
        else:
            real_features = real_reference_features(classifier, cfg["dataset"], device)
            np.save(real_features_cache, real_features)
        metrics.update(evaluate_generated_samples(classifier, images, real_features, device))

    (output_dir / f"metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
