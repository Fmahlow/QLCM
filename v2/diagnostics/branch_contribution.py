"""E0 of the 2026-10-20 preregistration: how much does the circuit branch contribute in existing models?

Run from repository root:
  python v2/diagnostics/branch_contribution.py --campaign results/v2-campaign-datasets-teachers --output v2/reports/extension-circuit/e0-branch-contribution.json
For every model with a branch: RMS(branch)/RMS(backbone) on noised training latents per t (also for the
input-dependent part of the branch, i.e. without its mean), and FID-like of three variants (teacher 32 DDIM
steps, students 1 and 4 steps): unmodified; readout zeroed (removes the branch, including its learned
constant); features_mean (circuit features replaced by their mean over noised training latents, so only the
input-dependent computation of the circuit is removed).
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'v1'))
from v2.core import Schedule,Student,sample_latents
from v2.run import data,latents,get_teacher,net,load
from src.eval_metrics import SmallClassifier,real_reference_features,evaluate_generated_samples

def features(z,t,schedule):
    u=t.float()/schedule.steps
    return torch.cat([z,u[:,None],torch.sin(math.pi*u)[:,None],torch.cos(math.pi*u)[:,None]],-1)

@torch.no_grad()
def ratios(network,z,schedule):
    rows=[]
    for level in range(1,schedule.steps+1):
        t=torch.full((len(z),),level,device=z.device,dtype=torch.long); noisy,_=schedule.noise(z,t); f=features(noisy,t,schedule)
        out=network.branch(f); branch=out.square().mean().sqrt(); varying=(out-out.mean(0)).square().mean().sqrt()
        backbone=network.backbone(f).square().mean().sqrt() if network.backbone is not None else torch.tensor(0.)
        rows.append({'t':level,'branch_rms':float(branch),'branch_varying_rms':float(varying),'backbone_rms':float(backbone),'ratio':float(branch/backbone) if float(backbone) else None})
    return rows

@torch.no_grad()
def mean_features(network,z,schedule):
    feats=[]
    for level in range(1,schedule.steps+1):
        t=torch.full((len(z),),level,device=z.device,dtype=torch.long); noisy,_=schedule.noise(z,t)
        feats.append(network.quantum(math.pi*network.compressor(features(noisy,t,schedule))))
    return torch.cat(feats).mean(0)

def ablated(model,variant,mean=None):
    model=copy.deepcopy(model); network=model.network if isinstance(model,Student) else model
    with torch.no_grad():
        if variant=='features_mean': network.readout.bias.copy_(network.readout(mean))
        else: network.readout.bias.zero_()
        network.readout.weight.zero_()
    return model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--campaign',default='results/v2-campaign-datasets-teachers'); p.add_argument('--output',required=True)
    p.add_argument('--classifier-dir',default='results/v2-metrics'); p.add_argument('--datasets',default='mnist,fashion_mnist,kmnist'); p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--device',default='cuda'); args=p.parse_args(); device=torch.device(args.device); results=[]
    for dataset in args.datasets.split(','):
        classifier=SmallClassifier().to(device); classifier.load_state_dict(load(Path(args.classifier_dir)/f'{dataset}_classifier.pt',device)); classifier.eval()
        real=real_reference_features(classifier,dataset,device)
        for seed in [int(s) for s in args.seeds.split(',')]:
            for teacher_kind,students in (('hybrid',('hybrid','surrogate')),('no_entanglement',())):
                out=Path(args.campaign)/dataset/f'seed-{seed}'/f'teacher-{teacher_kind}'
                c=dict(json.loads((out/'config.json').read_text()),device=args.device); schedule=Schedule(c['timesteps'],device)
                z,decoder,checkpoint=latents(c,out,DataLoader(data(c),batch_size=c['batch']),device)
                models=[('teacher',teacher_kind,get_teacher(c,out,device),True)]
                for kind in students:
                    m=Student(net(c,kind)).to(device); m.load_state_dict(load(out/f'student_{kind}.pt',device)['ema']); m.eval(); models.append(('student',kind,m,False))
                torch.manual_seed(seed); noise=torch.randn(c['samples'],c['latent'],device=device)
                for role,kind,model,is_teacher in models:
                    network=model if is_teacher else model.network
                    torch.manual_seed(seed+7); sample=z[torch.randperm(len(z),device=device)[:2048]]
                    entry={'dataset':dataset,'seed':seed,'teacher_kind':teacher_kind,'role':role,'kind':kind,'ratios':ratios(network,sample,schedule),'fid':[]}
                    for steps in ((32,) if is_teacher else (1,4)):
                        mean=mean_features(network,sample,schedule)
                        for variant,m in (('full',model),('readout_zeroed',ablated(model,'readout_zeroed')),('features_mean',ablated(model,'features_mean',mean))):
                            torch.manual_seed(seed+1)
                            with torch.no_grad(): images=decoder.decode(sample_latents(m,schedule,noise,steps,is_teacher)*checkpoint['std']+checkpoint['mean'])
                            metrics=evaluate_generated_samples(classifier,images,real,device)
                            entry['fid'].append({'steps':steps,'variant':variant,'fid_like':metrics['fid_like_classifier_features'],'label_entropy':metrics['label_entropy']})
                    results.append(entry); print(dataset,seed,teacher_kind,role,kind,[(f['steps'],f['variant'],round(f['fid_like'],1)) for f in entry['fid']],
                                                 'median ratio',round(sorted(r['ratio'] for r in entry['ratios'])[16],4),flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps({'scope':'E0 of the 2026-10-20 preregistration; descriptive','campaign':args.campaign,'results':results},indent=2)+'\n')
if __name__=='__main__': main()
