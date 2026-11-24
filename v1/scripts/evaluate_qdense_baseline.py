from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.eval_metrics import evaluate_generated_samples, real_reference_features, train_or_load_classifier
from src.metrics import count_parameters, time_call
from src.qlcm.qdense import QDenseDiffusionCircuit


def save_grid(images_8x8: torch.Tensor, path: Path):
    images = (images_8x8.clamp(0, 1) * 255.0).byte().cpu().numpy()
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


@torch.no_grad()
def sample_qdense(model, n_samples: int, steps: int, device: str, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand(n_samples, 64, generator=generator).to(device)
    classes = torch.randint(0, 2, (n_samples,), generator=generator).to(device)
    for _ in range(steps):
        x = model(x, classes)
    return x.clamp(0.0, 1.0), classes


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Q-Dense literature-reproduction baseline "
        "(scripts/train_qdense_baseline.py) at a given NFE budget (1..tau), for "
        "comparison against QLCM's 1/2/4-step consistency sampling."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-image-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=47)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--grid-samples", type=int, default=64)
    parser.add_argument("--output-dir", default="results/qdense")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = QDenseDiffusionCircuit(
        n_image_qubits=args.n_image_qubits, n_layers=args.n_layers, n_classes=2
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (flat_images, classes), elapsed = time_call(sample_qdense, model, args.samples, args.steps, str(device), args.seed)
    images_8x8 = flat_images.view(-1, 8, 8)
    tag = args.run_id or f"qdense_{args.steps}step"
    save_grid(images_8x8[: args.grid_samples], output_dir / f"samples_{tag}.png")

    # Upsample to 28x28 and rescale to [-1, 1] to reuse the shared MNIST
    # classifier-feature metric, which is trained on native 28x28 MNIST.
    images_28 = F.interpolate(images_8x8.unsqueeze(1), size=(28, 28), mode="bilinear", align_corners=False)
    images_28 = images_28 * 2.0 - 1.0

    metrics = {
        "run_id": args.run_id,
        "seed": args.seed,
        "placement": "qdense_literature",
        "qubits": args.n_image_qubits + 1,
        "depth": args.n_layers,
        "sampling_steps": args.steps,
        "samples": args.samples,
        "student_parameters": count_parameters(model),
        "sampling_seconds": elapsed,
        "sample_grid": str(output_dir / f"samples_{tag}.png"),
    }

    classifier_cache = Path("results/mnist") / "mnist_classifier.pt"
    classifier = train_or_load_classifier("mnist", device, str(classifier_cache))
    real_features_cache = Path("results/mnist") / "mnist_real_features.npy"
    if real_features_cache.exists():
        real_features = np.load(real_features_cache)
    else:
        real_features = real_reference_features(classifier, "mnist", device)
        np.save(real_features_cache, real_features)
    metrics.update(evaluate_generated_samples(classifier, images_28, real_features, device))

    (output_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
