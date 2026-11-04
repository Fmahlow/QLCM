"""E6, E7 and the solver part of E8 (2026-11-04 preregistration). Inference only, on the retrained campaign.

Run from repository root (GPU):
  python v2/diagnostics/review_eval.py --campaign results/v2-campaign-datasets-teachers --output results/v2-review-eval
Per dataset/seed: 5,000 generated samples per model/sampler from one shared initial noise, scored against a real
reference set (5,000 test images; PneumoniaMNIST: the 624 test images) with
  FID-like and precision/recall (k=3, Kynkaenniemi et al. 2019) in classifier features, label entropy, confidence;
  Frechet distance and MMD (RBF, median bandwidth) in the normalized AE latent, against encoded reference images.
Floors: other real images (5,000 test; PneumoniaMNIST: 524 validation) and their AE reconstructions.
E6 positive control: branch perturbations (scale of the input-dependent part, permutation across samples) and
Gaussian noise added to generated latents, for the hybrid teacher (32 steps) and hybrid student (1 step).
E8 solvers: DPM-Solver++(2M) and Heun on the hybrid teacher, on the integer levels of the DDIM schedule.
One JSON per dataset/seed (resumable), then report.json.
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'v1'))
from v2.core import Schedule, Student, sample_latents
from v2.run import ae, get_teacher, net, load, data, latents
from src.datasets import build_dataset
from src.eval_metrics import SmallClassifier, frechet_distance

SCALES = (0.0, 0.5, 1.5, 2.0, 3.0)
LATENT_NOISE = (0.05, 0.1, 0.2, 0.4)
CONTAMINATION = (0.02, 0.05, 0.1, 0.2)   # post hoc


# ------------------------------------------------------------------ real data
def real_sets(dataset, n):
    """Reference and floor image sets (disjoint). The third set is a post hoc floor (added after the first run,
    see PROTOCOL.md): training images never seen by the AE/teacher (indices 10,000-15,000), because the two halves
    of the MNIST-family test sets are not exchangeable. PneumoniaMNIST has no unused training images (None)."""
    if dataset == 'pneumoniamnist':
        raw = np.load(ROOT / 'data/medmnist/pneumoniamnist.npz')
        conv = lambda a: torch.from_numpy(a).float().div(255).mul(2).sub(1).unsqueeze(1)
        return conv(raw['test_images']), conv(raw['val_images']), None
    test = build_dataset(dataset, root=str(ROOT / 'data'), train=False)
    images = torch.stack([test[i][0] for i in range(len(test))])
    train = build_dataset(dataset, root=str(ROOT / 'data'), train=True)
    unseen = torch.stack([train[i][0] for i in range(10000, 10000 + n)])
    return images[:n], images[n:2 * n], unseen


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def features(classifier, images, device):
    out_f, out_p = [], []
    for x in images.split(1000):
        x = x.to(device); out_f.append(classifier.features(x)); out_p.append(classifier(x).softmax(1))
    return torch.cat(out_f), torch.cat(out_p)


@torch.no_grad()
def precision_recall(real, fake, k=3):
    def radii(x):
        d = torch.cdist(x, x); return d.kthvalue(k + 1, dim=1).values   # k-th neighbour, excluding self
    r_real, r_fake = radii(real), radii(fake)
    d = torch.cdist(fake, real)
    precision = (d <= r_real[None, :]).any(1).float().mean()
    recall = (d.T <= r_fake[None, :]).any(1).float().mean()
    return float(precision), float(recall)


@torch.no_grad()
def mmd_rbf(x, y, bandwidth):
    """Unbiased squared MMD with a Gaussian kernel."""
    k = lambda a, b: torch.exp(-torch.cdist(a, b).square() / (2 * bandwidth ** 2))
    n, m = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    return float((kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) - 2 * kxy.mean())


class Scorer:
    def __init__(self, classifier, real_images, real_latents, device):
        self.c, self.device = classifier, device
        self.rf, _ = features(classifier, real_images, device)
        self.rf_np = self.rf.cpu().double().numpy()
        self.rl = real_latents
        g = torch.Generator(device='cpu').manual_seed(0)
        sub = real_latents[torch.randperm(len(real_latents), generator=g)[:2000]]
        self.bandwidth = float(torch.cdist(sub, sub).median())     # fixed per reference set

    def __call__(self, images, latents=None):
        f, p = features(self.c, images, self.device)
        counts = torch.bincount(p.argmax(1), minlength=p.shape[1]).double(); q = counts / counts.sum(); q = q[q > 0]
        if len(f) > len(self.rf):  # R4: equal set sizes for P/R (PneumoniaMNIST: 624 real), mean of 10 subsamples
            g = torch.Generator(device='cpu').manual_seed(0); pr = [precision_recall(self.rf, f[torch.randperm(len(f), generator=g)[:len(self.rf)].to(f.device)]) for _ in range(10)]
            precision, recall = float(np.mean([x[0] for x in pr])), float(np.mean([x[1] for x in pr]))
        else: precision, recall = precision_recall(self.rf, f)
        row = {'n': len(images), 'fid_like': frechet_distance(self.rf_np, f.cpu().double().numpy()), 'precision': precision, 'recall': recall,
               'label_entropy': float(-(q * q.log()).sum()), 'mean_confidence': float(p.max(1).values.mean()),
               'label_histogram': counts.long().tolist(), 'pixel_variance': float(images.var(0).mean())}
        if latents is not None:
            lat = latents.float()
            row['latent_frechet'] = frechet_distance(self.rl.cpu().double().numpy(), lat.cpu().double().numpy())
            row['latent_mmd'] = mmd_rbf(lat.to(self.device), self.rl.to(self.device), self.bandwidth)
            row['latent_energy_gt4_fraction'] = float((lat.square().mean(1) > 4).float().mean())
        return row


# ------------------------------------------------------------------ samplers
def levels_for(schedule, steps, device):
    return torch.linspace(schedule.steps, 1, steps, device=device).round().long().tolist() + [0]


@torch.no_grad()
def dpm_solver_pp_2m(model, schedule, noise, nfe):
    """DPM-Solver++(2M), data prediction, first order at the first and last step (lower_order_final)."""
    lv = levels_for(schedule, nfe, noise.device); z = noise.clone(); prev = None
    lam = lambda t: math.log(math.sqrt(float(schedule.abar[t])) / math.sqrt(1 - float(schedule.abar[t]))) if t > 0 else math.inf
    for i in range(nfe):
        t, s = lv[i], lv[i + 1]
        tt = torch.full((len(z),), t, device=z.device, dtype=torch.long)
        x0, _ = schedule.decode_v(z, tt, model(z, tt, schedule))
        if s == 0: z = x0; break
        a_s, b_s = math.sqrt(float(schedule.abar[s])), math.sqrt(1 - float(schedule.abar[s])); b_t = math.sqrt(1 - float(schedule.abar[t]))
        h = lam(s) - lam(t)
        if prev is None: d = x0
        else:
            r = (lam(t) - lam(prev[0])) / h; d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * prev[1]
        z = (b_s / b_t) * z - a_s * math.expm1(-h) * d
        prev = (t, x0)
    return z


@torch.no_grad()
def heun(model, schedule, noise, steps):
    """Heun (2nd order) on the DDIM ODE in x/alpha, sigma/alpha coordinates. Euler (= DDIM) on the first step,
    because sigma/alpha ~ 1e3 at the terminal level (abar_T = 1e-6) makes the trapezoidal correction there unstable,
    and on the last step to t=0. Model evaluations: 2*steps-2."""
    lv = levels_for(schedule, steps, noise.device); z = noise.clone()
    ab = lambda t: (math.sqrt(float(schedule.abar[t])), math.sqrt(1 - float(schedule.abar[t])))
    def eps(z, t):
        tt = torch.full((len(z),), t, device=z.device, dtype=torch.long)
        return schedule.decode_v(z, tt, model(z, tt, schedule))[1]
    for i in range(steps):
        t, s = lv[i], lv[i + 1]; (a_t, b_t), (a_s, b_s) = ab(t), ab(s)
        xbar = z / a_t; e_t = eps(z, t); xs = xbar + (b_s / a_s - b_t / a_t) * e_t
        if s > 0 and i > 0:
            e_s = eps(a_s * xs, s); xs = xbar + (b_s / a_s - b_t / a_t) * 0.5 * (e_t + e_s)
        z = a_s * xs
    return z


# ------------------------------------------------------------------ perturbations (E6)
def network_of(model): return model.network if isinstance(model, Student) else model


@torch.no_grad()
def branch_mean(network, z, schedule):
    outs = []
    for level in range(1, schedule.steps + 1):
        t = torch.full((len(z),), level, device=z.device, dtype=torch.long); noisy, _ = schedule.noise(z, t)
        u = t.float() / schedule.steps
        f = torch.cat([noisy, u[:, None], torch.sin(math.pi * u)[:, None], torch.cos(math.pi * u)[:, None]], -1)
        outs.append(network.branch(f))
    return torch.cat(outs).mean(0)


def perturbed(model, kind, value, mean=None, seed=0):
    model = copy.deepcopy(model); network = network_of(model); original = network.branch
    if kind == 'scale':
        network.branch = lambda f: mean + value * (original(f) - mean)
    elif kind == 'readout_zeroed':  # post hoc: removes the whole branch, including its learned constant (E0)
        network.branch = lambda f: torch.zeros_like(original(f))
    else:  # permutation of branch outputs across the samples of the batch
        g = torch.Generator(device=mean.device).manual_seed(seed)
        network.branch = lambda f: original(f)[torch.randperm(len(f), generator=g, device=f.device)]
    return model


# ------------------------------------------------------------------ main loop
def evaluate_seed(args, dataset, seed, device, classifier, ref_images, floor_images, unseen_images):
    base = Path(args.campaign) / dataset / f'seed-{seed}'
    out = Path(args.output) / dataset / f'seed-{seed}.json'
    if out.exists(): return json.loads(out.read_text())
    start = time.monotonic()
    hyb = base / 'teacher-hybrid'
    c = dict(json.loads((hyb / 'config.json').read_text()), device=str(device)); schedule = Schedule(c['timesteps'], device)
    ck = load(hyb / 'autoencoder.pt', device); auto = ae(c).to(device); auto.load_state_dict(ck['state']); auto.eval()
    norm = lambda z: (z - ck['mean']) / ck['std']
    with torch.no_grad():
        enc = lambda imgs: torch.cat([norm(auto.encode(x.to(device))) for x in imgs.split(1000)])
        decode = lambda lat: torch.cat([auto.decode(l * ck['std'] + ck['mean']) for l in lat.split(1000)])
        ref_lat = enc(ref_images); floor_lat = enc(floor_images)
        recon = torch.cat([auto.decode(auto.encode(x.to(device))) for x in floor_images.split(1000)])
    score = Scorer(classifier, ref_images, ref_lat, device)
    rows = [dict(model='real (floor)', sampler='-', steps=0, **score(floor_images, floor_lat)),
            dict(model='AE reconstruction (floor)', sampler='-', steps=0, **score(recon.cpu(), None))]
    if unseen_images is not None:
        with torch.no_grad(): unseen_lat = enc(unseen_images)
        rows.append(dict(model='real unseen train (floor, post hoc)', sampler='-', steps=0, **score(unseen_images, unseen_lat)))
        with torch.no_grad(): rec_u = torch.cat([auto.decode(auto.encode(x.to(device))) for x in unseen_images.split(1000)])
        rows.append(dict(model='AE reconstruction of unseen train (R4)', sampler='-', steps=0, **score(rec_u.cpu(), None)))
    with torch.no_grad(): rec_r = torch.cat([auto.decode(auto.encode(x.to(device))) for x in ref_images.split(1000)])
    rows.append(dict(model='AE reconstruction of reference (R4)', sampler='-', steps=0, **score(rec_r.cpu(), None)))
    torch.manual_seed(seed); noise = torch.randn(args.n, c['latent'], device=device)

    def run(label, model, sampler, steps, fn, extra=None):
        torch.manual_seed(seed + 1)
        with torch.no_grad(): lat = fn()
        if not torch.isfinite(lat).all(): lat = torch.nan_to_num(lat, nan=0.0, posinf=1e3, neginf=-1e3)
        rows.append(dict(model=label, sampler=sampler, steps=steps, **(extra or {}), **score(decode(lat).cpu(), lat)))
        return lat

    def load_student(directory, kind, file=None):
        saved = load(directory / (file or f'student_{kind}.pt'), device)
        m = Student(net(dict(json.loads((directory / 'config.json').read_text()), device=str(device)), kind)).to(device)
        m.load_state_dict(saved['ema']); m.eval(); return m

    for tk in ('hybrid', 'classical', 'no_entanglement'):
        d = base / f'teacher-{tk}'
        cfg = dict(json.loads((d / 'config.json').read_text()), device=str(device))
        teacher = get_teacher(cfg, d, device)
        for st in (1, 2, 4, 8, 32):
            run(f'teacher-{tk}', teacher, 'ddim', st, lambda: sample_latents(teacher, schedule, noise, st, True), {'teacher_kind': tk})
        kinds = ('hybrid', 'classical', 'surrogate') if tk == 'hybrid' else ('classical',)
        for kind in kinds:
            if not (d / f'student_{kind}.pt').exists(): continue
            s = load_student(d, kind)
            for st in (1, 4):
                run(f'student-{kind}', s, 'consistency', st, lambda: sample_latents(s, schedule, noise, st), {'teacher_kind': tk})
        if tk != 'hybrid': continue
        # E8: solvers on the hybrid teacher
        for nfe in (3, 4, 6, 8):
            run('teacher-hybrid', teacher, 'dpm_solver_pp_2m', nfe, lambda: dpm_solver_pp_2m(teacher, schedule, noise, nfe), {'teacher_kind': tk, 'nfe': nfe})
        for st in (3, 4, 5):
            run('teacher-hybrid', teacher, 'heun', st, lambda: heun(teacher, schedule, noise, st), {'teacher_kind': tk, 'nfe': 2 * st - 2})
        for fname, label in (('student_endpoint.pt', 'student-endpoint_regression'), ('student_endpoint_100k.pt', 'student-endpoint_regression_100k')):
            if (d / fname).exists():
                e = load_student(d, 'classical', fname)
                run(label, e, 'direct', 1, lambda: sample_latents(e, schedule, noise, 1), {'teacher_kind': tk})
        # E6: positive control
        z_train, _, _ = latents(cfg, d, DataLoader(data(cfg), batch_size=cfg['batch']), device)
        torch.manual_seed(seed + 7); sub = z_train[torch.randperm(len(z_train), device=device)[:2048]]
        student = load_student(d, 'hybrid')
        for label, model, is_teacher, st in (('teacher-hybrid', teacher, True, 32), ('student-hybrid', student, False, 1)):
            mean = branch_mean(network_of(model), sub, schedule)
            for a in SCALES:
                m = perturbed(model, 'scale', a, mean)
                run(label, m, 'perturb_scale', st, lambda: sample_latents(m, schedule, noise, st, is_teacher), {'teacher_kind': tk, 'perturbation': 'scale', 'value': a})
            m = perturbed(model, 'shuffle', None, mean, seed)
            run(label, m, 'perturb_shuffle', st, lambda: sample_latents(m, schedule, noise, st, is_teacher), {'teacher_kind': tk, 'perturbation': 'shuffle', 'value': None})
            m = perturbed(model, 'readout_zeroed', None, mean)
            run(label, m, 'perturb_readout_zeroed', st, lambda: sample_latents(m, schedule, noise, st, is_teacher), {'teacher_kind': tk, 'perturbation': 'readout_zeroed', 'value': None, 'post_hoc': True})
            torch.manual_seed(seed + 1)
            with torch.no_grad(): clean = sample_latents(model, schedule, noise, st, is_teacher)
            # Post hoc contamination control: a known fraction p of the samples replaced by 2-step DDIM samples of the
            # same teacher (a degenerate sampler), i.e. a degradation of known size and direction.
            torch.manual_seed(seed + 1)
            with torch.no_grad(): bad = sample_latents(teacher, schedule, noise, 2, True)
            for frac in CONTAMINATION:
                k = int(round(frac * len(clean))); lat = torch.cat([bad[:k], clean[k:]])
                rows.append(dict(model=label, sampler='perturb_contamination', steps=st, teacher_kind=tk, perturbation='contamination', value=frac,
                                 post_hoc=True, **score(decode(lat).cpu(), lat)))
            for sd in LATENT_NOISE:
                g = torch.Generator(device=device).manual_seed(seed + 11)
                lat = clean + sd * torch.randn(clean.shape, generator=g, device=device)
                rows.append(dict(model=label, sampler='perturb_latent_noise', steps=st, teacher_kind=tk, perturbation='latent_noise', value=sd,
                                 **score(decode(lat).cpu(), lat)))
    result = {'dataset': dataset, 'seed': seed, 'n_generated': args.n, 'n_reference': len(ref_images), 'n_floor': len(floor_images),
              'mmd_bandwidth': score.bandwidth, 'seconds': time.monotonic() - start, 'rows': rows}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=1) + '\n')
    print('DONE', dataset, seed, round(result['seconds'], 1), 's', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--campaign', default='results/v2-campaign-datasets-teachers'); p.add_argument('--output', default='results/v2-review-eval')
    p.add_argument('--classifier-dir', default='results/v2-metrics'); p.add_argument('--datasets', default='mnist,fashion_mnist,kmnist,pneumoniamnist')
    p.add_argument('--seeds', default='0,1,2'); p.add_argument('--n', type=int, default=5000); p.add_argument('--device', default='cuda'); a = p.parse_args()
    device = torch.device(a.device); results = []; torch.set_grad_enabled(False)
    for dataset in a.datasets.split(','):
        classifier = SmallClassifier().to(device); classifier.load_state_dict(load(Path(a.classifier_dir) / f'{dataset}_classifier.pt', device)); classifier.eval()
        ref, floor, unseen = real_sets(dataset, a.n)
        for seed in [int(s) for s in a.seeds.split(',')]:
            results.append(evaluate_seed(a, dataset, seed, device, classifier, ref, floor, unseen))
    report = {'scope': 'E6/E7/E8 of the 2026-11-04 preregistration (inference only, retrained campaign)', 'n_generated': a.n,
              'metrics': 'FID-like and P/R(k=3) in classifier features; latent Frechet and unbiased RBF-MMD in normalized AE latent',
              'results': results}
    (Path(a.output) / 'report.json').write_text(json.dumps(report) + '\n')
    print('REVIEW EVAL COMPLETE', flush=True)


if __name__ == '__main__': main()
