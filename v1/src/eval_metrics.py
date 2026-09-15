from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class SmallClassifier(nn.Module):
    """Small CNN classifier used as a feature extractor for FID-like and
    recognizability metrics on MNIST/Fashion-MNIST. Not a substitute for an
    Inception-based FID; results must be reported as 'FID-like (classifier
    features)', consistent with experiment_plan.md Phase 4."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        return F.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))


def train_or_load_classifier(
    dataset_name: str, device: torch.device, cache_path: str, epochs: int = 3
) -> SmallClassifier:
    from src.datasets import build_dataset

    cache_path_p = Path(cache_path)
    model = SmallClassifier().to(device)
    if cache_path_p.exists():
        model.load_state_dict(torch.load(cache_path_p, map_location=device))
        model.eval()
        return model

    dataset = build_dataset(dataset_name, train=True, max_samples=10000)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    cache_path_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cache_path_p)
    return model


def frechet_distance(feat_real: np.ndarray, feat_fake: np.ndarray, eps: float = 1e-6) -> float:
    from scipy.linalg import sqrtm

    mu1, mu2 = feat_real.mean(axis=0), feat_fake.mean(axis=0)
    sigma1 = np.cov(feat_real, rowvar=False)
    sigma2 = np.cov(feat_fake, rowvar=False)
    # Regularize: sample covariance is near-singular when n_samples <= n_features,
    # which makes sqrtm numerically unstable and can return NaN/complex values.
    identity = np.eye(sigma1.shape[0])
    sigma1 = sigma1 + eps * identity
    sigma2 = sigma2 + eps * identity
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


@torch.no_grad()
def real_reference_features(classifier: SmallClassifier, dataset_name: str, device: torch.device, n: int = 1000) -> np.ndarray:
    from src.datasets import build_dataset

    dataset = build_dataset(dataset_name, train=False, max_samples=n)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    feats = []
    for x, _ in loader:
        feats.append(classifier.features(x.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def evaluate_generated_samples(
    classifier: SmallClassifier, images: torch.Tensor, real_features: np.ndarray, device: torch.device
) -> dict:
    images = images.to(device)
    feats = classifier.features(images).cpu().numpy()
    logits = classifier(images)
    probs = F.softmax(logits, dim=1)
    confidence = float(probs.max(dim=1).values.mean().item())
    preds = probs.argmax(dim=1).cpu().numpy()
    counts = np.bincount(preds, minlength=10).astype(np.float64)
    p = counts / counts.sum()
    p_nonzero = p[p > 0]
    label_entropy = float(-(p_nonzero * np.log(p_nonzero)).sum())
    return {
        "fid_like_classifier_features": frechet_distance(real_features, feats),
        "classifier_mean_confidence": confidence,
        "label_entropy": label_entropy,
        "label_entropy_max_bits": float(np.log(10)),
        "predicted_label_histogram": counts.astype(int).tolist(),
    }
