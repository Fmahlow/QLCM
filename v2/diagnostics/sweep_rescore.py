"""R1/R2 of the 2026-11-04 round-1 preregistration: re-score the E9 and E10 sweeps with the E7 protocol, and run the
E6 branch perturbations where the branch carries the model (E10, h in {0, 4, 16}). Inference only.

Run from repository root (GPU):
  python v2/diagnostics/sweep_rescore.py --sweep results/v2-e9-correlators --seeds 0,1,2 --output results/v2-r1-e9
  python v2/diagnostics/sweep_rescore.py --sweep results/v2-e10-capacity-seeds --seeds 3,4,5 --output results/v2-r1-e10 --perturb
Per dataset/seed: 5,000 samples from one shared initial noise, one-step student and 32-step teacher of every pipeline,
scored against the first 5,000 test images (FID-like, P/R, latent Frechet/MMD against the reference encoded by the
pipeline's own AE; all pipelines of a dataset/seed share that AE). One JSON per dataset/seed (resumable).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'v1')); sys.path.insert(0, str(ROOT / 'v2/diagnostics'))
from v2.core import Schedule, Student, sample_latents
from v2.run import ae, get_teacher, net, load, data, latents
from src.eval_metrics import SmallClassifier
from review_eval import Scorer, real_sets, perturbed, branch_mean, network_of, SCALES

KINDS = ('hybrid', 'surrogate', 'classical_branch', 'classical', 'no_entanglement')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--sweep', required=True); p.add_argument('--output', required=True)
    p.add_argument('--datasets', default='mnist,fashion_mnist'); p.add_argument('--seeds', required=True)
    p.add_argument('--classifier-dir', default='results/v2-metrics'); p.add_argument('--n', type=int, default=5000)
    p.add_argument('--perturb', action='store_true', help='R2: branch perturbations on hybrid pipelines with h in {0,4,16}')
    a = p.parse_args(); device = torch.device('cuda'); torch.set_grad_enabled(False); out_root = Path(a.output); results = []
    for dataset in a.datasets.split(','):
        classifier = SmallClassifier().to(device); classifier.load_state_dict(load(Path(a.classifier_dir) / f'{dataset}_classifier.pt', device)); classifier.eval()
        ref, _, _ = real_sets(dataset, a.n)
        for seed in [int(s) for s in a.seeds.split(',')]:
            out = out_root / dataset / f'seed-{seed}.json'
            if out.exists(): results.append(json.loads(out.read_text())); continue
            start = time.monotonic(); base = Path(a.sweep) / dataset / f'seed-{seed}'; rows = []; scorers = {}
            torch.manual_seed(seed); noise = None
            for d in sorted(x for x in base.iterdir() if x.is_dir() and (x / 'config_eval.json').exists()):
                kind = d.name.split('-h')[0]
                if kind not in KINDS or not (d / f'student_{kind}.pt').exists(): continue
                c = dict(json.loads((d / 'config_eval.json').read_text()), device='cuda'); schedule = Schedule(c['timesteps'], device)
                ck = load(d / 'autoencoder.pt', device); key = hashlib.sha256((d / 'autoencoder.pt').read_bytes()).hexdigest()
                auto = ae(c).to(device); auto.load_state_dict(ck['state']); auto.eval()
                if key not in scorers:
                    ref_lat = torch.cat([(auto.encode(x.to(device)) - ck['mean']) / ck['std'] for x in ref.split(1000)])
                    scorers[key] = Scorer(classifier, ref, ref_lat, device)
                score = scorers[key]
                if noise is None:
                    torch.manual_seed(seed); noise = torch.randn(a.n, c['latent'], device=device)
                decode = lambda lat: torch.cat([auto.decode(l * ck['std'] + ck['mean']) for l in lat.split(1000)]).cpu()
                teacher = get_teacher(c, d, device)
                student = Student(net(c, kind)).to(device); student.load_state_dict(load(d / f'student_{kind}.pt', device)['ema']); student.eval()
                meta = dict(pipeline=d.name, kind=kind, hidden=c['hidden'], qubits=c['qubits'], depth=c['depth'], readout=c.get('readout', 'z'))
                def run(role, model, is_teacher, st, extra=None):
                    torch.manual_seed(seed + 1); lat = sample_latents(model, schedule, noise, st, is_teacher)
                    lat = torch.nan_to_num(lat, nan=0.0, posinf=1e3, neginf=-1e3)
                    rows.append(dict(meta, role=role, steps=st, **(extra or {}), **score(decode(lat), lat)))
                run('student', student, False, 1); run('teacher', teacher, True, 32)
                if a.perturb and kind == 'hybrid' and c['hidden'] in (0, 4, 16):
                    z_train, _, _ = latents(c, d, DataLoader(data(c), batch_size=c['batch']), device)
                    torch.manual_seed(seed + 7); sub = z_train[torch.randperm(len(z_train), device=device)[:2048]]
                    for role, model, is_teacher, st in (('student', student, False, 1), ('teacher', teacher, True, 32)):
                        mean = branch_mean(network_of(model), sub, schedule)
                        for s_ in SCALES:
                            run(role, perturbed(model, 'scale', s_, mean), is_teacher, st, {'perturbation': 'scale', 'value': s_})
                        run(role, perturbed(model, 'shuffle', None, mean, seed), is_teacher, st, {'perturbation': 'shuffle', 'value': None})
                        run(role, perturbed(model, 'readout_zeroed', None, mean), is_teacher, st, {'perturbation': 'readout_zeroed', 'value': None})
            result = {'dataset': dataset, 'seed': seed, 'sweep': a.sweep, 'n_generated': a.n, 'seconds': time.monotonic() - start, 'rows': rows}
            out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result) + '\n'); results.append(result)
            print('DONE', dataset, seed, len(rows), 'rows', round(result['seconds'], 1), 's', flush=True)
    (out_root / 'report.json').write_text(json.dumps({'scope': 'R1/R2 of the 2026-11-04 round-1 preregistration', 'sweep': a.sweep, 'results': results}) + '\n')
    print('RESCORE COMPLETE', flush=True)


if __name__ == '__main__': main()
