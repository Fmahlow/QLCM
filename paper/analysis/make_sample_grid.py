"""Qualitative sample grid (paper/TODO.md item 6): teacher (32 DDIM steps), hybrid student (1 step) and classical
student (1 step) of the hybrid teacher, seed 0, four datasets. Reads the samples_*.pt tensors that v2.run eval saves
in the retrained campaign (not versioned) and writes paper/figures/fig_samples.pdf (versioned).

    python paper/analysis/make_sample_grid.py [--campaign results/v2-campaign-datasets-teachers] [--seed 0]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
ROOT = Path(__file__).resolve().parents[2]
DATASETS = [('mnist', 'MNIST'), ('fashion_mnist', 'Fashion-MNIST'), ('kmnist', 'KMNIST'), ('pneumoniamnist', 'PneumoniaMNIST')]
MODELS = [('teacher', 32, 'teacher, 32 DDIM steps'), ('hybrid', 1, 'hybrid student, 1 step'), ('classical', 1, 'classical student, 1 step')]


def tile(x, n=6):
    x = x[:n * n, 0].clamp(-1, 1).add(1).div(2)                       # (n*n, 28, 28) in [0, 1]
    return x.reshape(n, n, 28, 28).permute(0, 2, 1, 3).reshape(n * 28, n * 28)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--campaign', default=str(ROOT / 'results/v2-campaign-datasets-teachers')); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    fig, axes = plt.subplots(len(DATASETS), len(MODELS), figsize=(5.4, 7.4))
    for i, (d, dn) in enumerate(DATASETS):
        for j, (m, st, lab) in enumerate(MODELS):
            x = torch.load(Path(a.campaign) / d / f'seed-{a.seed}' / 'teacher-hybrid' / f'samples_{m}_{st}.pt', weights_only=True)
            ax = axes[i, j]; ax.imshow(tile(x).numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if i == 0: ax.set_title(lab, fontsize=8)
            if j == 0: ax.set_ylabel(dn, fontsize=8)
    fig.tight_layout(pad=0.3)
    out = ROOT / 'paper/figures/fig_samples.pdf'; fig.savefig(out, bbox_inches='tight', dpi=300); print('wrote', out)


if __name__ == '__main__': main()
