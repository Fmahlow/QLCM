from __future__ import annotations

import csv
import json
from pathlib import Path

FIELDS = [
    "source_file",
    "run_id",
    "dataset",
    "placement",
    "qubits",
    "depth",
    "seed",
    "sampling_steps",
    "samples",
    "student_parameters",
    "sampling_seconds",
    "fid_like_classifier_features",
    "classifier_mean_confidence",
    "label_entropy",
    "label_entropy_max_bits",
]


def main():
    rows = []
    for metrics_path in sorted(Path("results").glob("*/metrics_*.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        dataset = metrics_path.parent.name
        row = {"source_file": str(metrics_path), "dataset": dataset}
        row.update({k: data.get(k) for k in FIELDS if k not in ("source_file", "dataset")})
        rows.append(row)

    out_path = Path("results/summary.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
