from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.qlcm.consistency import ClassicalConsistencyModel, QuantumLatentConsistencyModel


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    classical = ClassicalConsistencyModel(
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        time_dim=cfg["model"]["time_dim"],
    )
    print(f"classical_consistency,{count_parameters(classical)}")
    for placement in ["head", "residual", "time"]:
        model = QuantumLatentConsistencyModel(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["model"]["hidden_dim"],
            time_dim=cfg["model"]["time_dim"],
            n_qubits=cfg["model"]["qubits"],
            n_layers=cfg["model"]["depth"],
            placement=placement,
        )
        print(f"qlcm_{placement},{count_parameters(model)}")


if __name__ == "__main__":
    main()
