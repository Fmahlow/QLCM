"""Dataset extension + teacher control: does distillation generalize, and does the teacher circuit matter?

Run from repository root:
  python v2/diagnostics/dataset_teacher_campaign.py --pilot results/v2-pilot-mnist --diag results/v2-diag-student-budget/e200_ema095
Per dataset and seed: one AE, then hybrid/classical/no_entanglement teachers on that same AE.
Hybrid teacher gets hybrid/classical/surrogate students; control teachers get a classical student.
Not part of the pilot protocol: student budget is 200 epochs / EMA 0.95 (see RESULTS.md).
With --pilot/--diag, MNIST AE, hybrid teacher and its students are copied instead of retrained.
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TEACHERS=('hybrid','classical','no_entanglement')
STUDENTS={'hybrid':('hybrid','classical','surrogate'),'classical':('classical',),'no_entanglement':('classical',)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='v2/gpu_pilot.json'); p.add_argument('--output',default='results/v2-campaign-datasets-teachers')
    p.add_argument('--datasets',default='mnist,fashion_mnist,kmnist'); p.add_argument('--seeds',default='0,1,2'); p.add_argument('--classifier-dir',default='results/v2-metrics')
    p.add_argument('--pilot'); p.add_argument('--diag'); p.add_argument('--student-epochs',type=int,default=200); p.add_argument('--ema-decay',type=float,default=.95)
    p.add_argument('--workers',type=int,default=12); p.add_argument('--no-gate',action='store_true',help='smoke tests only'); args=p.parse_args()
    base=json.loads(Path(args.config).read_text()); out=Path(args.output).resolve(); seeds=[int(s) for s in args.seeds.split(',')]; datasets=args.datasets.split(',')
    metrics=Path(args.classifier_dir).resolve(); slots=threading.Semaphore(args.workers); failures=[]; gates={}
    validation={}
    for dataset in datasets:
        validation[dataset]=json.loads((metrics/f'{dataset}_classifier.validation.json').read_text())
        if validation[dataset].get('heldout') is not True or validation[dataset].get('test_accuracy',0)<.8: p.error(f'{dataset}: classifier accuracy < 0.8 or not held-out')
    def stage(directory,name,extra,tag):
        command=[sys.executable,'-m','v2.run','--config',str(directory/'config.json'),'--output',str(directory),'--stage',name]+extra
        with slots:
            start=time.monotonic()
            with (directory/f'log-{tag}.txt').open('w') as log: subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',directory.relative_to(out),tag,round(time.monotonic()-start,1),flush=True)
    def classifier(dataset): return ['--classifier',str(metrics/f'{dataset}_classifier.pt')]
    def guarded(fn,*a):
        def run():
            try: fn(*a)
            except Exception as error:
                failures.append((str(a),str(error))); print('FAILED',a,str(error)[:200],flush=True)
        thread=threading.Thread(target=run); thread.start(); return thread
    def configure(directory,dataset,seed,kind):
        directory.mkdir(parents=True,exist_ok=True)
        config=dict(base,dataset=dataset,seed=seed,teacher_kind=kind,student_epochs=args.student_epochs,ema_decay=args.ema_decay)
        (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def student(directory,kind,reuse):
        if (directory/f'student_{kind}.pt').exists(): return
        if reuse:
            for file in (f'student_{kind}.pt',f'student_{kind}_training.json'): shutil.copy2(reuse/file,directory/file)
        else: stage(directory,'distill',['--kind',kind],f'distill-{kind}')
    def teacher_chain(dataset,seed,kind,ae):
        directory=out/dataset/f'seed-{seed}'/f'teacher-{kind}'; configure(directory,dataset,seed,kind)
        if not (directory/'autoencoder.pt').exists(): shutil.copy2(ae/'autoencoder.pt',directory/'autoencoder.pt')
        pilot=Path(args.pilot).resolve()/f'seed-{seed}' if args.pilot and dataset=='mnist' and kind=='hybrid' else None
        if not (directory/'teacher.pt').exists():
            if pilot:
                for file in ('teacher.pt','teacher_training.json'): shutil.copy2(pilot/file,directory/file)
            else: stage(directory,'teacher',[],'teacher')
        if not (directory/'teacher_gate.json').exists():
            stage(directory,'eval',classifier(dataset),'eval-teacher')
            evaluation=json.loads((directory/'evaluation.json').read_text()); (directory/'teacher_evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')
            row=[r for r in evaluation['rows'] if r['model']=='teacher' and r['steps']==base['timesteps']][0]
            # Same heuristic gate as v2.orchestrate; for non-10-class data the entropy reference is the real training-label entropy.
            passed=row['label_entropy']>=.8*validation[dataset].get('train_label_entropy',math.log(10)) and math.isfinite(row['fid_like_classifier_features']) and row['classifier_mean_confidence']>=.8*validation[dataset]['real_mean_confidence']
            (directory/'teacher_gate.json').write_text(json.dumps({'passed':passed,'heuristic_default':True,'row':row},indent=2)+'\n')
        gates[str(directory.relative_to(out))]=json.loads((directory/'teacher_gate.json').read_text())['passed']
        print('GATE',directory.relative_to(out),gates[str(directory.relative_to(out))],flush=True)
        if not gates[str(directory.relative_to(out))] and not args.no_gate: return
        reuse=Path(args.diag).resolve()/f'seed-{seed}' if pilot and args.diag else None
        for thread in [guarded(student,directory,k,reuse) for k in STUDENTS[kind]]: thread.join()
    def seed_chain(dataset,seed):
        ae=out/dataset/f'seed-{seed}'/'ae'; configure(ae,dataset,seed,'hybrid')
        if not (ae/'autoencoder.pt').exists():
            if args.pilot and dataset=='mnist':
                for file in ('autoencoder.pt','ae_training.json'): shutil.copy2(Path(args.pilot).resolve()/f'seed-{seed}'/file,ae/file)
            else: stage(ae,'ae',[],'ae')
        for thread in [guarded(teacher_chain,dataset,seed,k,ae) for k in TEACHERS]: thread.join()
    for thread in [guarded(seed_chain,d,s) for d in datasets for s in seeds]: thread.join()
    # Final evaluation is sequential so inference timings are not contended by training.
    rows=[]
    for dataset in datasets:
        for seed in seeds:
            for kind in TEACHERS:
                directory=out/dataset/f'seed-{seed}'/f'teacher-{kind}'
                if not (directory/'teacher.pt').exists(): continue
                try:
                    if not (directory/'final_evaluation.done').exists():
                        stage(directory,'eval',classifier(dataset),'eval-final'); (directory/'final_evaluation.done').write_text('')
                    for row in json.loads((directory/'evaluation.json').read_text())['rows']: rows.append(dict(row,dataset=dataset,seed=seed,teacher_kind=kind))
                except Exception as error:
                    failures.append((str(directory),str(error))); print('FAILED',directory,str(error)[:200],flush=True)
    (out/'report.json').write_text(json.dumps({'scope':'dataset extension and teacher control; not pilot protocol; not evidence of quantum advantage','config':base,'student_epochs':args.student_epochs,'ema_decay':args.ema_decay,'gate_disabled':args.no_gate,'teacher_gates':gates,'classifier_validation':validation,'failures':failures,'rows':rows},indent=2)+'\n')
    print('CAMPAIGN COMPLETE' if not failures else 'CAMPAIGN FINISHED WITH FAILURES',out/'report.json',flush=True)
    if failures: sys.exit(1)
if __name__=='__main__': main()
