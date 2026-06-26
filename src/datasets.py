from __future__ import annotations

from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.data import BarsAndStripes


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
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_samples is not None and name != "bars_stripes":
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset
