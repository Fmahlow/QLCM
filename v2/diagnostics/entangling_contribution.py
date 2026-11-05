"""Post hoc (round-2 review): how much does entanglement change the trained circuit's output?

With even depth and no re-uploading the CZ ring cancels at zero rotations (CZ_ring^2 = I), so the entangled circuit
is initialised at the separable function. For each trained hybrid pipeline of E9 and E10 (teacher and EMA student),
this measures the RMS difference between the circuit's features with and without the CZ gates (same weights),
on the circuit's actual input angles (compressor applied to noised training latents at every level), relative
to the SD of the entangled features. Also at initialisation (same RNG draw as training).

Run from repository root:
  python v2/diagnostics/entangling_contribution.py --output v2/reports/review-2026-11/r2-entangling-contribution.json
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'v1'))
from v2.core import Schedule, Student, QuantumFeatures
from v2.run import get_teacher, net, load, data, latents


@torch.no_grad()
def contribution(network, z, schedule):
    q = network.quantum; sep = copy.deepcopy(q); sep.entangle = False
    angles = []
    for level in range(1, schedule.steps + 1):
        t = torch.full((len(z),), level, device=z.device, dtype=torch.long); noisy, _ = schedule.noise(z, t); u = t.float() / schedule.steps
        f = torch.cat([noisy, u[:, None], torch.sin(math.pi * u)[:, None], torch.cos(math.pi * u)[:, None]], -1)
        angles.append(math.pi * network.compressor(f))
    a = torch.cat(angles); e, s = q(a), sep(a)
    return float((e - s).square().mean().sqrt() / e.std(0).mean()), float(q.weights.square().mean().sqrt())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); p.add_argument('--device', default='cuda'); a = p.parse_args()
    device = torch.device(a.device); torch.set_grad_enabled(False); rows = []
    for sweep, seeds in (('results/v2-e9-correlators', (0, 1, 2)), ('results/v2-e10-capacity-seeds', (3, 4, 5))):
        for dataset in ('mnist', 'fashion_mnist'):
            for seed in seeds:
                base = ROOT / sweep / dataset / f'seed-{seed}'; z_cache = {}
                for d in sorted(x for x in base.iterdir() if x.name.startswith('hybrid-') and (x / 'student_hybrid.pt').exists()):
                    c = dict(json.loads((d / 'config_eval.json').read_text()), device=a.device); schedule = Schedule(c['timesteps'], device)
                    if 'z' not in z_cache:
                        z, _, _ = latents(c, d, DataLoader(data(c), batch_size=c['batch']), device)
                        torch.manual_seed(seed + 7); z_cache['z'] = z[torch.randperm(len(z), device=device)[:1024]]
                    z = z_cache['z']
                    teacher = get_teacher(c, d, device)
                    student = Student(net(c, 'hybrid')).to(device); student.load_state_dict(load(d / 'student_hybrid.pt', device)['ema']); student.eval()
                    torch.manual_seed(c['seed']); init = net(c, 'hybrid').to(device).eval()   # same draw as the start of teacher training
                    meta = dict(sweep=Path(sweep).name, dataset=dataset, seed=seed, pipeline=d.name, hidden=c['hidden'], qubits=c['qubits'], depth=c['depth'],
                                readout=c.get('readout', 'z'), reupload=c.get('reupload', False))
                    for role, network in (('teacher', teacher), ('student', student.network), ('init', init)):
                        rel, w = contribution(network, z, schedule)
                        rows.append(dict(meta, role=role, relative_rms_entangling=rel, weight_rms=w))
                print('DONE', sweep, dataset, seed, flush=True)
    Path(a.output).write_text(json.dumps({'scope': 'post hoc, round-2 review: entangling contribution of trained hybrid circuits', 'rows': rows}, indent=1) + '\n')


if __name__ == '__main__': main()
