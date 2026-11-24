from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets import build_dataset
from src.qlcm.qdense import QDenseDiffusionCircuit, forward_chain


def main():
    parser = argparse.ArgumentParser(
        description="Train a faithful reproduction of the Q-Dense quantum diffusion "
        "architecture from Koelle et al. 2024 (arXiv:2401.07049), Sec 3.1/3.3/4.4: "
        "amplitude-embedded 8x8 MNIST digits 0/1, ancilla-guided, 47 dense layers, "
        "trained to predict x_{t-1} from x_t over tau=10 diffusion steps. This is the "
        "literature-standard many-step quantum diffusion baseline for comparing "
        "against QLCM's few-step consistency sampling."
    )
    parser.add_argument("--n-image-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=47)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/qdense")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(
        "mnist", train=True, max_samples=args.max_samples, image_size=8, digit_filter=[0, 1], signed=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = QDenseDiffusionCircuit(
        n_image_qubits=args.n_image_qubits, n_layers=args.n_layers, n_classes=2
    ).to(device)

    if args.dry_run:
        x, y = next(iter(loader))
        x0 = x.view(x.shape[0], -1).to(device)
        chain = forward_chain(x0, args.timesteps)
        pred = model(chain[1], y.to(device))
        print({"x0_shape": tuple(x0.shape), "chain_shape": tuple(chain.shape), "pred_shape": tuple(pred.shape)})
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        losses = []
        for x, y in loader:
            batch = x.shape[0]
            x0 = x.view(batch, -1).to(device)
            y = y.to(device)
            with torch.no_grad():
                chain = forward_chain(x0, args.timesteps)
                t = torch.randint(1, args.timesteps + 1, (batch,), device=device)
                idx = torch.arange(batch, device=device)
                x_t = chain[t, idx]
                x_prev = chain[t - 1, idx]
            pred = model(x_t, y)
            loss = F.mse_loss(pred, x_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"epoch={epoch + 1} qdense_x_prev_mse={sum(losses) / len(losses):.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.run_id or f"qdense_seed{args.seed}"
    torch.save(model.state_dict(), output_dir / f"{tag}.pt")


if __name__ == "__main__":
    main()
