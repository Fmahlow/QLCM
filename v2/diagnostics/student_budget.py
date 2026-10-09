"""Student-budget diagnostic: reuse pilot AE/teacher, vary only student epochs/EMA.

Run from repository root after the pilot completed:
  python v2/diagnostics/student_budget.py --pilot results/v2-pilot-mnist --output results/v2-diag-student-budget --classifier results/v2-metrics/mnist_classifier.pt
Not part of the pilot protocol; changes the student budget on purpose. Pilot outputs are read-only here.
no_entanglement is omitted: surrogate is its exact algebraic equivalent (identical pilot metrics).
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
VARIANTS={'e50_ema095':{'student_epochs':50,'ema_decay':.95},'e200_ema0999':{'student_epochs':200,'ema_decay':.999},'e200_ema095':{'student_epochs':200,'ema_decay':.95}}
KINDS=('hybrid','classical','surrogate')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--pilot',default='results/v2-pilot-mnist'); p.add_argument('--output',default='results/v2-diag-student-budget')
    p.add_argument('--classifier',required=True); p.add_argument('--seeds',default='0,1,2'); p.add_argument('--workers',type=int,default=9); args=p.parse_args()
    pilot=Path(args.pilot).resolve(); out=Path(args.output).resolve(); seeds=[int(s) for s in args.seeds.split(',')]
    def call(directory,stage,extra,tag):
        command=[sys.executable,'-m','v2.run','--config',str(directory/'config.json'),'--output',str(directory),'--stage',stage]+extra
        start=time.monotonic()
        with (directory/f'log-{tag}.txt').open('w') as log: subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',directory.parent.name,directory.name,tag,round(time.monotonic()-start,1),flush=True)
    jobs=[]
    for name,change in VARIANTS.items():
        for seed in seeds:
            directory=out/name/f'seed-{seed}'; directory.mkdir(parents=True,exist_ok=True)
            source=pilot/f'seed-{seed}'; config=dict(json.loads((source/'config.json').read_text()),**change)
            (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            for file in ('autoencoder.pt','teacher.pt'):
                if not (directory/file).exists(): shutil.copy2(source/file,directory/file)
            for kind in KINDS:
                if not (directory/f'student_{kind}.pt').exists(): jobs.append((directory,'distill',['--kind',kind],f'distill-{kind}'))
    # Longest jobs first; training runs concurrently, so training wall time is not a benchmark.
    jobs.sort(key=lambda j:(-json.loads((j[0]/'config.json').read_text())['student_epochs'],j[2][1]!='hybrid'))
    with ThreadPoolExecutor(args.workers) as pool:
        for future in [pool.submit(call,*job) for job in jobs]: future.result()
    # Evaluation is sequential so inference timings are not contended.
    rows=[]
    for name,change in VARIANTS.items():
        for seed in seeds:
            directory=out/name/f'seed-{seed}'
            if not (directory/'evaluation.json').exists(): call(directory,'eval',['--classifier',str(Path(args.classifier).resolve())],'eval')
            for row in json.loads((directory/'evaluation.json').read_text())['rows']: rows.append(dict(row,seed=seed,variant=name,**change))
    (out/'report.json').write_text(json.dumps({'scope':'student-budget diagnostic on pilot AE/teacher checkpoints; not pilot protocol','variants':VARIANTS,'kinds':KINDS,'pilot':str(pilot),'rows':rows},indent=2)+'\n')
    print('DIAGNOSTIC COMPLETE',out/'report.json',flush=True)
if __name__=='__main__': main()
