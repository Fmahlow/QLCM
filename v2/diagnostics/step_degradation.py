"""E4a of the 2026-10-20 preregistration: why do students get worse with more sampling steps?

Run from repository root:
  python v2/diagnostics/step_degradation.py --pipeline <campaign dir>/mnist/seed-0/teacher-hybrid ... --output ...
Per level t, the student endpoint f(z_t,t) is compared with the teacher's full DDIM trajectory from the same
z_t down to 0 (latent MSE, normalized latents), for two input distributions:
  data: z_t = noise(x0, t) with x0 a training latent;
  self: z_t = noise(f(z_T, T), t), the input a multi-step student actually sees after its first step.
Also the per-sample latent energy |z|^2/dim of k-step samples (teacher DDIM and students), to separate
typical-sample error from divergence of a few samples.
"""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'v1'))
from v2.core import Schedule,Student,sample_latents
from v2.run import data,latents,get_teacher,net,load

@torch.no_grad()
def teacher_endpoint(teacher,schedule,z,level):
    for t in range(level,0,-1):
        tt=torch.full((len(z),),t,device=z.device,dtype=torch.long); z=schedule.ddim(z,tt,tt-1,teacher(z,tt,schedule))
    return z

def energy_row(c,model,steps,energy):
    q=lambda v:float(energy.quantile(v))
    return {'dataset':c['dataset'],'seed':c['seed'],'model':model,'steps':steps,'mean':float(energy.mean()),'median':q(.5),'p90':q(.9),'p99':q(.99),
            'max':float(energy.max()),'fraction_above_4':float((energy>4).float().mean())}

def main():
    tails=[]
    a=argparse.ArgumentParser(); a.add_argument('--pipeline',action='append',required=True); a.add_argument('--output',required=True)
    a.add_argument('--device',default='cuda'); a.add_argument('--samples',type=int,default=2048); args=a.parse_args(); device=torch.device(args.device); rows=[]
    for pipeline in map(Path,args.pipeline):
        c=dict(json.loads((pipeline/'config.json').read_text()),device=args.device); schedule=Schedule(c['timesteps'],device)
        z,_,_=latents(c,pipeline,DataLoader(data(c),batch_size=c['batch']),device); teacher=get_teacher(c,pipeline,device)
        torch.manual_seed(c['seed']+11); x0=z[torch.randperm(len(z),device=device)[:args.samples]]; prior=torch.randn_like(x0)
        top=torch.full((len(x0),),schedule.steps,device=device,dtype=torch.long)
        for steps in (1,2,4,8,16,32):
            torch.manual_seed(c['seed']+1); energy=sample_latents(teacher,schedule,prior,steps,True).square().mean(1); tails.append(energy_row(c,'teacher',steps,energy))
        for kind in ('hybrid','classical','surrogate'):
            if not (pipeline/f'student_{kind}.pt').exists(): continue
            student=Student(net(c,kind)).to(device); student.load_state_dict(load(pipeline/f'student_{kind}.pt',device)['ema']); student.eval()
            with torch.no_grad(): first=student(prior,top,schedule)
            for level in range(1,schedule.steps+1):
                t=torch.full((len(x0),),level,device=device,dtype=torch.long); eps=torch.randn_like(x0)
                for source,clean in (('data',x0),('self',first)):
                    zt,_=schedule.noise(clean,t,eps)
                    with torch.no_grad(): pred=student(zt,t,schedule); target=teacher_endpoint(teacher,schedule,zt,level)
                    rows.append({'dataset':c['dataset'],'seed':c['seed'],'student':kind,'t':level,'input':source,
                                 'student_vs_teacher_mse':float((pred-target).square().mean()),'teacher_vs_input_clean_mse':float((target-clean).square().mean()),
                                 'student_vs_input_clean_mse':float((pred-clean).square().mean())})
            for steps in (1,2,4,8,16,32):
                torch.manual_seed(c['seed']+1); energy=sample_latents(student,schedule,prior,steps).square().mean(1); tails.append(energy_row(c,kind,steps,energy))
            print(pipeline,kind,'data t=1,8,32:',[round(r['student_vs_teacher_mse'],4) for r in rows if r['student']==kind and r['input']=='data' and r['seed']==c['seed'] and r['dataset']==c['dataset'] and r['t'] in (1,8,32)],flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps({'scope':'E4a of the 2026-10-20 preregistration; descriptive','rows':rows,'latent_energy':tails},indent=2)+'\n')
if __name__=='__main__': main()
