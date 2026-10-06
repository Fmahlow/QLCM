"""Run from repository root: python -m v2.run --stage smoke --output ..."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from v2.core import *
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v1'))
from src.qlcm.autoencoder import MlpAutoencoder,ConvAutoencoder
from src.data import BarsAndStripes

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def finite(loss):
    if not torch.isfinite(loss): raise RuntimeError('nonfinite loss; stop before checkpoint')
def load(path,device): return torch.load(path,map_location=device,weights_only=True)
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')
def net(c,kind): return Network(c['latent'],c['hidden'],c['qubits'],c['depth'],kind)
def data(c):
    if c['dataset']=='bars_stripes': return BarsAndStripes(c['max_samples'],seed=c['seed'])
    from src.datasets import build_dataset
    return build_dataset(c['dataset'],root=str(ROOT/'data'),max_samples=c['max_samples'])
def ae(c):
    return MlpAutoencoder(8,c['latent']) if c['dataset']=='bars_stripes' else ConvAutoencoder(c['latent'])
def config_signature(c):
    return {k:c[k] for k in ('dataset','max_samples','latent','hidden','qubits','depth','timesteps','seed')}
def train_ae(c,out,loader,device):
    model=ae(c).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    losses=[]
    for _ in range(c['ae_epochs']):
        for x,_ in loader:
            x=x.to(device); loss=F.mse_loss(model(x)[0],x); finite(loss); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad(): z=torch.cat([model.encode(x.to(device)) for x,_ in loader])
    mean=z.mean(0); std=z.std(0).clamp_min(.01)
    torch.save({'state':model.state_dict(),'mean':mean,'std':std,'signature':config_signature(c)},out/'autoencoder.pt')
    write(out/'ae_training.json',{'losses':losses,'latent_std_min':float(std.min())})
def latents(c,out,loader,device):
    checkpoint=load(out/'autoencoder.pt',device)
    if checkpoint['signature']!=config_signature(c): raise ValueError('AE config signature mismatch')
    model=ae(c).to(device); model.load_state_dict(checkpoint['state']); model.eval(); model.requires_grad_(False)
    with torch.no_grad(): z=torch.cat([(model.encode(x.to(device))-checkpoint['mean'])/checkpoint['std'] for x,_ in loader])
    return z,model,checkpoint

def train_teacher(c,out,z,device,schedule):
    model=net(c,'hybrid').to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); losses=[]
    for _ in range(c['teacher_epochs']):
        for batch in z[torch.randperm(len(z),device=device)].split(c['batch']):
            t,_=schedule.pairs(len(batch)); noisy,eps=schedule.noise(batch,t); a,b=schedule.coeff(t)
            loss=F.mse_loss(model(noisy,t,schedule),a*eps-b*batch)
            finite(loss); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    torch.save({'state':model.state_dict(),'signature':config_signature(c),'prediction':'v','ae_source':sha(out/'autoencoder.pt')},out/'teacher.pt')
    write(out/'teacher_training.json',{'losses':losses,'parameters':parameter_counts(model)})

def get_teacher(c,out,device):
    checkpoint=load(out/'teacher.pt',device)
    if checkpoint['signature']!=config_signature(c) or checkpoint['prediction']!='v' or checkpoint.get('ae_source')!=sha(out/'autoencoder.pt'): raise ValueError('teacher config mismatch')
    model=net(c,'hybrid').to(device); model.load_state_dict(checkpoint['state']); model.eval(); model.requires_grad_(False)
    return model

def train_student(c,out,z,device,schedule,kind):
    teacher=get_teacher(c,out,device)
    # Reset initialization/data/noise RNG for paired student runs.
    torch.manual_seed(c['seed'])
    student=Student(net(c,kind)).to(device); target=frozen_copy(student)
    opt=torch.optim.AdamW(student.parameters(),lr=1e-3); losses=[]
    torch.manual_seed(c['seed']+100)  # identical minibatch/noise draws across architectures
    for _ in range(c['student_epochs']):
        for batch in z[torch.randperm(len(z),device=device)].split(c['batch']):
            loss=distillation_loss(student,target,teacher,batch,schedule)
            finite(loss); opt.zero_grad(); loss.backward(); opt.step(); update_ema(target,student,c['ema_decay']); losses.append(float(loss.detach()))
    torch.save({'online':student.state_dict(),'ema':target.state_dict(),'signature':config_signature(c),'kind':kind,'teacher_source':sha(out/'teacher.pt'),'ae_source':sha(out/'autoencoder.pt')},out/f'student_{kind}.pt')
    write(out/f'student_{kind}_training.json',{'losses':losses,'parameters':parameter_counts(student),'ema_decay':c['ema_decay']})

def evaluate(c,out,device,schedule,classifier_path=None):
    _,decoder,checkpoint=latents(c,out,DataLoader(data(c),batch_size=c['batch']),device)
    teacher=get_teacher(c,out,device); models=[('teacher',teacher,True)]
    for kind in ('hybrid','classical','no_entanglement','surrogate'):
        path=out/f'student_{kind}.pt'
        if not path.exists(): continue
        saved=load(path,device)
        if saved['signature']!=config_signature(c) or saved.get('teacher_source')!=sha(out/'teacher.pt') or saved.get('ae_source')!=sha(out/'autoencoder.pt'): raise ValueError('student config mismatch')
        m=Student(net(c,kind)).to(device); m.load_state_dict(saved['ema']); m.eval(); models.append((kind,m,False))
    torch.manual_seed(c['seed']); noise=torch.randn(c['samples'],c['latent'],device=device)
    classifier=None
    if c['dataset']!='bars_stripes':
        if not classifier_path: raise ValueError('real-data evaluation requires an independently validated classifier checkpoint')
        from src.eval_metrics import SmallClassifier,real_reference_features,evaluate_generated_samples
        classifier=SmallClassifier().to(device); classifier.load_state_dict(load(classifier_path,device)); classifier.eval()
        real=real_reference_features(classifier,c['dataset'],device)
    rows=[]
    for name,model,is_teacher in models:
        for steps in c['evaluation_steps']:
            # Same random draws in every timing repeat; random cost is included.
            def generate():
                torch.manual_seed(c['seed']+1)
                with torch.no_grad():
                    latent=sample_latents(model,schedule,noise,steps,is_teacher)
                    return decoder.decode(latent*checkpoint['std']+checkpoint['mean'])
            images,timing=synchronized_time(generate,device)
            rows.append({'model':name,'steps':steps,'samples':len(images),'parameters':sum(p.numel() for p in model.parameters()),
                         'quantum_circuit_evaluations_per_sample':steps if name in ('teacher','hybrid','no_entanglement') else 0,
                         'quantum_circuit_evaluations_total':steps*len(images) if name in ('teacher','hybrid','no_entanglement') else 0,
                         'hardware_shots':None,'hardware_noise':False,'simulation':'exact statevector or algebraic/classical',
                         'pixel_variance':float(images.var(0).mean()),'timing':timing})
            if classifier is not None: rows[-1].update(evaluate_generated_samples(classifier,images,real,device))
            torch.save(images.cpu(),out/f'samples_{name}_{steps}.pt')
    write(out/'evaluation.json',{'scope':'synthetic smoke only' if c['dataset']=='bars_stripes' else 'pilot evaluation','config':c,'rows':rows})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default=str(ROOT/'v2/smoke.json'))
    p.add_argument('--stage',choices=['smoke','ae','teacher','distill','eval','dry-run'],required=True)
    p.add_argument('--output',required=True); p.add_argument('--kind',choices=['hybrid','classical','no_entanglement','surrogate'],default='hybrid'); p.add_argument('--classifier')
    args=p.parse_args(); c=json.loads(Path(args.config).read_text()); torch.manual_seed(c['seed']); torch.set_num_threads(2)
    device=torch.device(c['device']); schedule=Schedule(c['timesteps'],device); out=Path(args.output)
    if args.stage=='dry-run':
        z=torch.randn(2,c['latent'],device=device); teacher=net(c,'hybrid').to(device).eval(); teacher.requires_grad_(False)
        student=Student(net(c,args.kind)).to(device); loss=distillation_loss(student,frozen_copy(student),teacher,z,schedule); loss.backward()
        print({'dry_run_untrained':True,'loss':float(loss.detach()),'parameters':parameter_counts(student)}); return
    if args.stage=='smoke' and c['dataset']!='bars_stripes': raise ValueError('smoke is restricted to synthetic data')
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='smoke' and any(out.iterdir()): raise ValueError('smoke requires empty output directory')
    if args.stage in ('smoke','ae'):
        if (out/'autoencoder.pt').exists(): raise FileExistsError('use new output directory')
        train_ae(c,out,DataLoader(data(c),batch_size=c['batch'],shuffle=True),device)
    if args.stage in ('smoke','teacher','distill'):
        z,_,_=latents(c,out,DataLoader(data(c),batch_size=c['batch']),device)
        if args.stage in ('smoke','teacher'):
            if (out/'teacher.pt').exists(): raise FileExistsError('teacher checkpoint exists')
            train_teacher(c,out,z,device,schedule)
        if args.stage=='distill':
            if (out/f'student_{args.kind}.pt').exists(): raise FileExistsError('student checkpoint exists')
            train_student(c,out,z,device,schedule,args.kind)
        if args.stage=='smoke':
            for kind in ('hybrid','classical','no_entanglement','surrogate'): train_student(c,out,z,device,schedule,kind)
    if args.stage in ('smoke','eval'): evaluate(c,out,device,schedule,args.classifier)
    print('Completed',args.stage,'at',out)
if __name__=='__main__': main()
