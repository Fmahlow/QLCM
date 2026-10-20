from __future__ import annotations

from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.data import BarsAndStripes


def _medmnist(name: str, root: str, train: bool):
    """MedMNIST v2 28x28 grayscale subsets (CC BY 4.0), read from the official Zenodo npz."""
    from pathlib import Path
    from urllib.request import urlretrieve

    import numpy as np
    import torch
    from torch.utils.data import TensorDataset

    path = Path(root) / "medmnist" / f"{name}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(f"https://zenodo.org/records/10519652/files/{name}.npz?download=1", path)
    data = np.load(path)
    split = "train" if train else "test"
    images = torch.from_numpy(data[f"{split}_images"]).float().div(255.0).mul(2.0).sub(1.0).unsqueeze(1)
    labels = torch.from_numpy(data[f"{split}_labels"]).long().flatten()
    return TensorDataset(images, labels)


def build_dataset(name: str, root: str = "data", train: bool = True, max_samples: int | None = None):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    if name == "bars_stripes":
        dataset = BarsAndStripes(n_samples=max_samples or 1024, size=8, seed=0)
    elif name == "mnist":
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    elif name == "fashion_mnist":
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=True)
    elif name == "pneumoniamnist":
        dataset = _medmnist(name, root, train)
    elif name == "kmnist":
        dataset = datasets.KMNIST(root=root, train=train, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_samples is not None and name != "bars_stripes":
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset
