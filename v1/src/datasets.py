from __future__ import annotations

from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.data import BarsAndStripes


def build_dataset(
    name: str,
    root: str = "data",
    train: bool = True,
    max_samples: int | None = None,
    image_size: int | None = None,
    digit_filter: list[int] | None = None,
    signed: bool = True,
):
    """`image_size` resizes MNIST/Fashion-MNIST (default: native 28x28).
    `digit_filter` restricts MNIST to the given labels (e.g. [0, 1]), applied
    before `max_samples`. `signed` rescales pixels to [-1, 1] (the default
    used everywhere else in this repo); set False to keep raw [0, 1]
    intensities, needed for amplitude-embedding baselines whose readout
    (a probability distribution) cannot represent negative values.
    """
    transform_steps = [transforms.ToTensor()]
    if image_size is not None:
        transform_steps.insert(0, transforms.Resize((image_size, image_size)))
    if signed:
        transform_steps.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
    transform = transforms.Compose(transform_steps)

    if name == "bars_stripes":
        dataset = BarsAndStripes(n_samples=max_samples or 1024, size=8, seed=0)
    elif name == "mnist":
        dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    elif name == "fashion_mnist":
        dataset = datasets.FashionMNIST(root=root, train=train, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if digit_filter is not None and name in {"mnist", "fashion_mnist"}:
        allowed = set(digit_filter)
        indices = [i for i, label in enumerate(dataset.targets.tolist()) if label in allowed]
        dataset = Subset(dataset, indices)

    if max_samples is not None and name != "bars_stripes":
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset
