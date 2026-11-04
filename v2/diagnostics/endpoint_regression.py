"""E8 (2026-11-04 preregistration): direct endpoint regression baseline (Luhman & Luhman, 2021).

Run from repository root:
  python v2/diagnostics/endpoint_regression.py --campaign results/v2-campaign-datasets-teachers
For every dataset/seed, a classical network of the student's size (hidden 128) is trained for 200 epochs (EMA 0.95)
to map 10,000 fixed Gaussian latents to the endpoint of the hybrid teacher's 32-step DDIM trajectory.
Saved as student_endpoint.pt next to the campaign's hybrid teacher and evaluated by review_eval.py at 1 step.
"""
import argparse
import json
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'v1'))
from v2.core import Schedule, Student, frozen_copy, update_ema, sample_latents, parameter_counts
from v2.run import get_teacher, net, sha, config_signature, write


def train(directory, device, pairs, epochs, name='student_endpoint', val_pairs=0):
    c = dict(json.loads((directory / 'config.json').read_text()), device=device)
    if (directory / f'{name}.pt').exists(): return
    torch.set_num_threads(2); dev = torch.device(device); schedule = Schedule(c['timesteps'], dev)
    teacher = get_teacher(c, directory, dev)
    torch.manual_seed(c['seed'] + 200); noise = torch.randn(pairs, c['latent'], device=dev)
    endpoint = torch.cat([sample_latents(teacher, schedule, x, c['timesteps'], teacher=True) for x in noise.split(20000)])
    if val_pairs:  # held-out pairs (R5 of the round-1 preregistration), drawn after the training pairs
        val_noise = torch.randn(val_pairs, c['latent'], device=dev); val_end = sample_latents(teacher, schedule, val_noise, c['timesteps'], teacher=True)
    torch.manual_seed(c['seed'])
    student = Student(net(c, 'classical')).to(dev); target = frozen_copy(student)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3); losses = []
    t = torch.full((c['batch'],), c['timesteps'], device=dev, dtype=torch.long)
    torch.manual_seed(c['seed'] + 100); start = time.monotonic()
    for _ in range(epochs):
        for idx in torch.randperm(pairs, device=dev).split(c['batch']):
            loss = (student(noise[idx], t[:len(idx)], schedule) - endpoint[idx]).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            opt.zero_grad(); loss.backward(); opt.step(); update_ema(target, student, c['ema_decay']); losses.append(float(loss.detach()))
    torch.save({'online': student.state_dict(), 'ema': target.state_dict(), 'signature': config_signature(c), 'kind': 'endpoint_classical',
                'teacher_source': sha(directory / 'teacher.pt'), 'ae_source': sha(directory / 'autoencoder.pt')}, directory / f'{name}.pt')
    val_loss = None
    if val_pairs:
        with torch.no_grad():
            tv = torch.full((val_pairs,), c['timesteps'], device=dev, dtype=torch.long); val_loss = float((target(val_noise, tv, schedule) - val_end).square().mean())
    write(directory / f'{name}_training.json', {'losses': losses[::10], 'final_loss': losses[-1], 'ema_val_loss': val_loss, 'pairs': pairs, 'epochs': epochs,
          'ema_decay': c['ema_decay'], 'parameters': parameter_counts(student), 'seconds': time.monotonic() - start, 'device': device})
    print('DONE', directory, round(time.monotonic() - start, 1), 'final loss', round(losses[-1], 4), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--campaign', default='results/v2-campaign-datasets-teachers')
    p.add_argument('--datasets', default='mnist,fashion_mnist,kmnist,pneumoniamnist'); p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--device', default='cpu'); p.add_argument('--pairs', type=int, default=10000); p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--workers', type=int, default=6); p.add_argument('--name', default='student_endpoint'); p.add_argument('--val-pairs', type=int, default=0); a = p.parse_args()
    dirs = [Path(a.campaign) / d / f'seed-{s}' / 'teacher-hybrid' for d in a.datasets.split(',') for s in a.seeds.split(',')]
    # Separate processes: small models are GIL-bound in threads.
    with ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(train, dirs, [a.device] * len(dirs), [a.pairs] * len(dirs), [a.epochs] * len(dirs), [a.name] * len(dirs), [a.val_pairs] * len(dirs)))


if __name__ == '__main__': main()
