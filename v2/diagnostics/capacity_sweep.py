"""E1 (capacity sweep) and E2 (circuit scaling) from the 2026-10-20 preregistration in PROTOCOL.md.

Run from repository root:
  python v2/diagnostics/capacity_sweep.py --grid e1 --output results/v2-e1-capacity
  python v2/diagnostics/capacity_sweep.py --grid e2 --output results/v2-e2-scaling
E9/E10 of the 2026-11-04 preregistration (correlator readout; more E1 seeds):
  python v2/diagnostics/capacity_sweep.py --grid e9 --output results/v2-e9-correlators
  python v2/diagnostics/capacity_sweep.py --grid e10 --seeds 3,4,5 --output results/v2-e10-capacity-seeds \
      --ae-sources results/v2-mnist-replication
Matched pipelines: teacher and student of the same kind and size, on the campaign AE of each dataset/seed.
Training runs on CPU in parallel; final evaluations run one at a time on the GPU.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BRANCH_KINDS=('hybrid','surrogate','classical_branch')

def grid(name,datasets,seeds):
    jobs=[]
    if name in ('e1','e10'):
        for d in datasets:
            for s in seeds:
                for kind in BRANCH_KINDS:
                    for h in (0,4,8,16,32,128): jobs.append(dict(dataset=d,seed=s,kind=kind,hidden=h,qubits=4,depth=2,reupload=False))
                for h in (2,4,8,16,32,128): jobs.append(dict(dataset=d,seed=s,kind='classical',hidden=h,qubits=4,depth=2,reupload=False))
    elif name=='e9':
        for d in datasets:
            for s in seeds:
                for readout in ('zz','z'):
                    for kind in BRANCH_KINDS:
                        for h,q,depth in ((0,4,2),(4,4,2),(16,4,2),(0,6,4)): jobs.append(dict(dataset=d,seed=s,kind=kind,hidden=h,qubits=q,depth=depth,reupload=False,readout=readout))
    else:
        for d in datasets:
            for s in seeds:
                for q in (4,6,8):
                    for depth in (2,4):
                        for re in (False,True):
                            for kind in ('hybrid','surrogate'): jobs.append(dict(dataset=d,seed=s,kind=kind,hidden=0,qubits=q,depth=depth,reupload=re))
                    jobs.append(dict(dataset=d,seed=s,kind='classical_branch',hidden=0,qubits=q,depth=2,reupload=False))
    return jobs

def name(j): return f"{j['kind']}-h{j['hidden']}-q{j['qubits']}-d{j['depth']}"+('-re' if j['reupload'] else '')+('-zz' if j.get('readout','z')=='zz' else '')
def cost(j):  # rough relative cost, heaviest first to shorten the tail
    return (2**j['qubits'] if j['kind']=='hybrid' else 1)*j['depth']*(2 if j['reupload'] else 1)*(2 if j.get('readout')=='zz' else 1)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--grid',choices=['e1','e2','e9','e10'],required=True); p.add_argument('--output',required=True)
    p.add_argument('--config',default='v2/gpu_pilot.json'); p.add_argument('--campaign',default='results/v2-campaign-datasets-teachers')
    p.add_argument('--classifier-dir',default='results/v2-metrics'); p.add_argument('--datasets'); p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--workers',type=int,default=15); p.add_argument('--student-epochs',type=int,default=200)
    p.add_argument('--ae-sources',default='',help='extra campaign-like dirs searched for <dataset>/seed-<s>/ae/autoencoder.pt; else the AE is trained here')
    p.add_argument('--kinds',help='comma-separated subset of pipeline kinds (e.g. to run controls in a second process)'); args=p.parse_args()
    datasets=(args.datasets or ('mnist' if args.grid=='e2' else 'mnist,fashion_mnist')).split(',')
    sources=[Path(args.campaign).resolve()]+[Path(x).resolve() for x in args.ae_sources.split(',') if x]
    base=json.loads(Path(args.config).read_text()); out=Path(args.output).resolve(); campaign=Path(args.campaign).resolve(); metrics=Path(args.classifier_dir).resolve()
    jobs=sorted([j for j in grid(args.grid,datasets,[int(s) for s in args.seeds.split(',')]) if not args.kinds or j['kind'] in args.kinds.split(',')],key=cost,reverse=True); failures=[]
    def directory(j): return out/j['dataset']/f"seed-{j['seed']}"/name(j)
    def stage(d,config,stage_name,extra,tag):
        command=[sys.executable,'-m','v2.run','--config',str(d/config),'--output',str(d),'--stage',stage_name]+extra
        start=time.monotonic()
        with (d/f'log-{tag}.txt').open('w') as log: subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',d.relative_to(out),tag,round(time.monotonic()-start,1),flush=True)
    def train(j):
        d=directory(j); d.mkdir(parents=True,exist_ok=True)
        config=dict(base,dataset=j['dataset'],seed=j['seed'],hidden=j['hidden'],qubits=j['qubits'],depth=j['depth'],reupload=j['reupload'],
                    teacher_kind=j['kind'],student_epochs=args.student_epochs,ema_decay=.95,device='cpu')
        if j.get('readout','z')!='z': config['readout']=j['readout']
        (d/'config.json').write_text(json.dumps(config,indent=2)+'\n'); (d/'config_eval.json').write_text(json.dumps(dict(config,device='cuda'),indent=2)+'\n')
        found=[x/j['dataset']/f"seed-{j['seed']}" for x in sources if (x/j['dataset']/f"seed-{j['seed']}"/'ae'/'autoencoder.pt').exists()]
        seed_dir=found[0] if found else None
        if not (d/'autoencoder.pt').exists():
            if seed_dir: shutil.copy2(seed_dir/'ae'/'autoencoder.pt',d/'autoencoder.pt')
            else: shared_ae(j,config)
            if not seed_dir: shutil.copy2(out/'_ae'/j['dataset']/f"seed-{j['seed']}"/'autoencoder.pt',d/'autoencoder.pt')
        # The default-size hybrid and classical pipelines already exist in the campaign (same AE, same budget).
        reuse=seed_dir/f"teacher-{j['kind']}" if seed_dir and args.grid in ('e1','e10') and j['hidden']==base['hidden'] and j['kind'] in ('hybrid','classical') and (seed_dir/f"teacher-{j['kind']}"/f"student_{j['kind']}.pt").exists() else None
        try:
            if not (d/'teacher.pt').exists():
                if reuse: shutil.copy2(reuse/'teacher.pt',d/'teacher.pt'); shutil.copy2(reuse/'teacher_training.json',d/'teacher_training.json')
                else: stage(d,'config.json','teacher',[],'teacher')
            if not (d/f"student_{j['kind']}.pt").exists():
                if reuse:
                    for f in (f"student_{j['kind']}.pt",f"student_{j['kind']}_training.json"): shutil.copy2(reuse/f,d/f)
                else: stage(d,'config.json','distill',['--kind',j['kind']],f"distill-{j['kind']}")
        except Exception as error:
            failures.append((name(j),j['dataset'],j['seed'],str(error))); print('FAILED',d,str(error)[:200],flush=True)
    locks={}; guard=__import__('threading').Lock()
    def shared_ae(j,config):
        # One AE per dataset/seed when no campaign AE exists (trained once, then copied into every pipeline).
        key=(j['dataset'],j['seed'])
        with guard: lock=locks.setdefault(key,__import__('threading').Lock())
        with lock:
            a=out/'_ae'/j['dataset']/f"seed-{j['seed']}"; a.mkdir(parents=True,exist_ok=True)
            if not (a/'autoencoder.pt').exists():
                (a/'config.json').write_text(json.dumps(dict(config,hidden=base['hidden'],teacher_kind='hybrid'),indent=2)+'\n'); stage(a,'config.json','ae',[],'ae')
    with ThreadPoolExecutor(args.workers) as pool: list(pool.map(train,jobs))
    rows=[]
    for j in sorted(jobs,key=lambda j:(j['dataset'],j['seed'],name(j))):
        d=directory(j)
        if not (d/f"student_{j['kind']}.pt").exists(): continue
        try:
            if not (d/'final_evaluation.done').exists():
                stage(d,'config_eval.json','eval',['--classifier',str(metrics/f"{j['dataset']}_classifier.pt")],'eval'); (d/'final_evaluation.done').write_text('')
            for row in json.loads((d/'evaluation.json').read_text())['rows']: rows.append(dict(row,**j,pipeline=name(j)))
        except Exception as error:
            failures.append((name(j),j['dataset'],j['seed'],str(error))); print('FAILED',d,str(error)[:200],flush=True)
    report=out/('report.json' if not args.kinds else f"report.{args.kinds.replace(',','-')}.json")
    report.write_text(json.dumps({'scope':f'{args.grid} of the 2026-10-20 preregistration; not pilot protocol; not evidence of quantum advantage',
        'config':base,'student_epochs':args.student_epochs,'ema_decay':.95,'training_device':'cpu','evaluation_device':'cuda','failures':failures,'rows':rows},indent=2)+'\n')
    print('SWEEP COMPLETE' if not failures else 'SWEEP FINISHED WITH FAILURES',report,flush=True)
    if failures: sys.exit(1)
if __name__=='__main__': main()
