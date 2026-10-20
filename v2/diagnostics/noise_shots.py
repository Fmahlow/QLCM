"""E3 of the 2026-10-20 preregistration: finite shots and local depolarizing noise, at inference only.

Run from repository root:
  python v2/diagnostics/noise_shots.py --pipeline results/v2-campaign-datasets-teachers/mnist/seed-{seed}/teacher-hybrid --output ...
Models are trained noiselessly. Each circuit evaluation is simulated as a density matrix with depolarizing
noise p on every qubit after the encoding layer and after each trainable layer; all qubits are measured in
the computational basis after the final mixing, with `shots` samples per evaluation (None = exact).
"""
import argparse
import json
import sys
import types
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'v1'))
from v2.core import QuantumFeatures,Schedule,Student,sample_latents
from v2.run import data,latents,get_teacher,net,load,teacher_kind
from src.eval_metrics import SmallClassifier,real_reference_features,evaluate_generated_samples

def depolarize(rho,q,p):
    """Local depolarizing channel on every qubit: (1-3p/4) rho + p/4 (X rho X + Y rho Y + Z rho Z)."""
    if p==0: return rho
    batch=rho.shape[0]; n=2**q
    for w in range(q):
        r=rho.reshape([batch]+[2]*(2*q)); sign=torch.tensor([1.,-1.],device=rho.device)
        shape=[1]*(2*q+1); shape[1+w]=2; z=sign.reshape(shape); shape=[1]*(2*q+1); shape[1+q+w]=2; z=z*sign.reshape(shape)
        zrz=r*z; flip=lambda x:x.flip(1+w,1+q+w)
        rho=((1-.75*p)*r+.25*p*(flip(r)+zrz+flip(zrz))).reshape(batch,n,n)
    return rho

def evolve(rho,matrix):
    # Row convention: a pure state maps as state @ matrix, i.e. psi -> matrix^T psi.
    m=matrix.transpose(-1,-2); return m@rho@m.conj().transpose(-1,-2)

def noisy_forward(self,x,p,shots,generator):
    q=self.qubits; psi=self.product_state(x); rho=depolarize(psi[:,:,None]*psi[:,None,:].conj(),q,p)
    for layer in range(self.depth):
        if layer and self.reupload:
            mats=self.single_qubit(x[:,:q],x[:,q:]); encoding=mats[:,0]
            for w in range(1,q): encoding=(encoding[:,:,None,:,None]*mats[:,w,None,:,None,:]).reshape(x.shape[0],2**(w+1),-1)
            rho=evolve(rho,encoding)
        rho=depolarize(evolve(rho,self.layer_matrix(layer)),q,p)
    probs=evolve(rho,self.mixing_matrix()).diagonal(dim1=-2,dim2=-1).real.clamp_min(0); probs=probs/probs.sum(-1,keepdim=True)
    if shots is None: return probs@self.signs
    counts=torch.zeros_like(probs).scatter_add_(1,torch.multinomial(probs,shots,replacement=True,generator=generator),torch.ones(x.shape[0],shots,device=x.device))
    return (counts/shots)@self.signs

def set_noise(model,p,shots,generator):
    for module in model.modules():
        if isinstance(module,QuantumFeatures):
            module.forward=types.MethodType(lambda self,x:noisy_forward(self,x,p,shots,generator),module)

def main():
    a=argparse.ArgumentParser(); a.add_argument('--pipeline',action='append',required=True,help='pipeline directory with teacher.pt and student_hybrid.pt; repeatable')
    a.add_argument('--output',required=True); a.add_argument('--classifier-dir',default='results/v2-metrics'); a.add_argument('--device',default='cuda')
    a.add_argument('--noise',default='0,0.001,0.01,0.03'); a.add_argument('--shots',default='exact,1024,256,64'); args=a.parse_args(); device=torch.device(args.device); rows=[]
    noises=[float(v) for v in args.noise.split(',')]; shot_list=[None if v=='exact' else int(v) for v in args.shots.split(',')]
    for pipeline in map(Path,args.pipeline):
        c=dict(json.loads((pipeline/'config.json').read_text()),device=args.device); schedule=Schedule(c['timesteps'],device)
        if teacher_kind(c)!='hybrid': raise ValueError('E3 needs a hybrid teacher')
        classifier=SmallClassifier().to(device); classifier.load_state_dict(load(Path(args.classifier_dir)/f"{c['dataset']}_classifier.pt",device)); classifier.eval()
        real=real_reference_features(classifier,c['dataset'],device)
        _,decoder,checkpoint=latents(c,pipeline,DataLoader(data(c),batch_size=c['batch']),device)
        student=Student(net(c,'hybrid')).to(device); student.load_state_dict(load(pipeline/'student_hybrid.pt',device)['ema']); student.eval()
        models=[('teacher',get_teacher(c,pipeline,device),True,(4,8,32)),('student',student,False,(1,2,4))]
        torch.manual_seed(c['seed']); noise=torch.randn(c['samples'],c['latent'],device=device)
        for p in noises:
            for shots in shot_list:
                for role,model,is_teacher,steps_list in models:
                    for steps in steps_list:
                        generator=torch.Generator(device=device); generator.manual_seed(1000*c['seed']+steps); set_noise(model,p,shots,generator)
                        torch.manual_seed(c['seed']+1)
                        with torch.no_grad(): images=decoder.decode(sample_latents(model,schedule,noise,steps,is_teacher)*checkpoint['std']+checkpoint['mean'])
                        metrics=evaluate_generated_samples(classifier,images,real,device)
                        rows.append({'pipeline':str(pipeline),'dataset':c['dataset'],'seed':c['seed'],'hidden':c['hidden'],'qubits':c['qubits'],'depth':c['depth'],
                                     'reupload':c.get('reupload',False),'role':role,'steps':steps,'depolarizing_p':p,'shots_per_evaluation':shots,
                                     'shots_per_sample':None if shots is None else shots*steps,'fid_like':metrics['fid_like_classifier_features'],'label_entropy':metrics['label_entropy']})
                        print(pipeline.name,c['seed'],role,steps,p,shots,round(rows[-1]['fid_like'],1),flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps({'scope':'E3 of the 2026-10-20 preregistration; inference-only noise, models trained noiselessly; descriptive','rows':rows},indent=2)+'\n')
if __name__=='__main__': main()
