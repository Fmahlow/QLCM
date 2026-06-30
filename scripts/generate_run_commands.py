from __future__ import annotations

import argparse
import csv
from pathlib import Path


def config_for_dataset(dataset: str) -> str:
    if dataset == "mnist":
        return "configs/mnist.yaml"
    if dataset == "fashion_mnist":
        return "configs/fashion_mnist.yaml"
    return "configs/smoke.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="experiments/run_matrix.csv")
    parser.add_argument("--only-status", default="planned")
    args = parser.parse_args()
    rows = list(csv.DictReader(Path(args.matrix).read_text(encoding="utf-8").splitlines()))
    for row in rows:
        if args.only_status and row["status"] != args.only_status:
            continue
        cfg = config_for_dataset(row["dataset"])
        model = row["model"]
        if model == "autoencoder":
            print(f"# {row['run_id']}")
            print(f"python3 scripts/train_autoencoder.py --config {cfg}")
        elif model == "latent_diffusion_teacher":
            print(f"# {row['run_id']}")
            print(f"python3 scripts/train_teacher.py --config {cfg} --autoencoder results/{row['dataset']}/autoencoder.pt")
        elif model in {"classical_consistency", "qlcm"}:
            placement = row["placement"]
            if model == "classical_consistency":
                placement = "classical"
            extras = [
                f"--run-id {row['run_id']}",
                f"--sampling-steps {row['sampling_steps']}",
                f"--seed {row['seed']}",
            ]
            if row["qubits"] != "0":
                extras.append(f"--qubits {row['qubits']}")
            if row["depth"] != "0":
                extras.append(f"--depth {row['depth']}")
            print(f"# {row['run_id']}")
            print(
                "python3 scripts/train_qlcm.py "
                f"--config {cfg} --placement {placement} "
                f"--autoencoder results/{row['dataset']}/autoencoder.pt "
                + " ".join(extras)
            )


if __name__ == "__main__":
    main()
