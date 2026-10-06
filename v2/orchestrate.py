"""Conservative restart at stage boundaries; never resume incompatible runs."""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='v2/gpu_pilot.json'); p.add_argument('--output',default='results/v2-pilot-mnist')
    p.add_argument('--classifier'); p.add_argument('--classifier-validation'); p.add_argument('--dry-run',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--seeds',default='0,1,2'); p.add_argument('--stage-hours',type=float,default=6); p.add_argument('--total-hours',type=float,default=24)
    p.add_argument('--min-entropy-fraction',type=float,default=.8); args=p.parse_args()
    config=json.loads(Path(args.config).read_text()); seeds=[int(s) for s in args.seeds.split(',')]
    if len(set(seeds))!=len(seeds): p.error('duplicate seeds')
    sources=list((ROOT/'v2').glob('*.py'))+[ROOT/'v1/src/qlcm/autoencoder.py',ROOT/'v1/src/datasets.py',ROOT/'v1/src/eval_metrics.py',ROOT/'v1/src/data.py']
    identity={'config':config,'seeds':seeds,'sources':{str(f.relative_to(ROOT)):sha(f) for f in sorted(sources)},'classifier':sha(Path(args.classifier)) if args.classifier else None,'classifier_validation':sha(Path(args.classifier_validation)) if args.classifier_validation else None,'gate_entropy_fraction':args.min_entropy_fraction,'gate_confidence_real_ratio':.8,'stage_hours':args.stage_hours,'total_hours':args.total_hours}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=Path(args.output).resolve()
    plan=['ae','teacher','teacher evaluation + gate','distill hybrid','distill classical','distill no_entanglement','distill surrogate','eval']
    if args.dry_run:
        print(json.dumps({'dataset':config['dataset'],'output':str(out),'fingerprint':fingerprint,'seeds':seeds,'stages':plan,'limits':{'stage_hours':args.stage_hours,'invocation_hours':args.total_hours},'required':['validated classifier state_dict','validation JSON with checkpoint_sha256, dataset, test_accuracy, real_mean_confidence, heldout=true'],'gate':'default heuristic: label entropy >= 0.8 ln(10); confidence >= 0.8 real mean confidence; not a fidelity guarantee'},indent=2)); return
    if config['dataset']!='bars_stripes':
        if not args.classifier or not args.classifier_validation: p.error('classifier and held-out validation JSON required')
        val=json.loads(Path(args.classifier_validation).read_text())
        if val.get('checkpoint_sha256')!=sha(Path(args.classifier)) or val.get('dataset')!=config['dataset'] or val.get('heldout') is not True or val.get('test_accuracy',0)<.8 or not 0<val.get('real_mean_confidence',0)<=1:
            p.error('classifier validation mismatch or accuracy < 0.8 (heuristic default)')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: p.error('another orchestrator owns this output directory')
    manifest=out/'manifest.json'
    if manifest.exists():
        state=json.loads(manifest.read_text())
        if not args.resume: p.error('existing run requires --resume')
        if state['fingerprint']!=fingerprint: p.error('fingerprint mismatch; use new output directory')
    else:
        if any(f.name!='.lock' for f in out.iterdir()): p.error('nonempty output without manifest')
        out.mkdir(parents=True,exist_ok=True); state={'fingerprint':fingerprint,'identity':identity,'completed':{},'events':[]}
    def save(): manifest.write_text(json.dumps(state,indent=2)+'\n')
    save(); started=time.monotonic()
    def run(seed,stage,extra=None):
        key=f'{seed}:{stage}:'+str(extra or '')
        if key in state['completed']:
            for file,checksum in state['completed'][key].items():
                if not Path(file).exists() or sha(Path(file))!=checksum: raise RuntimeError('completed artifact changed/missing: '+file)
            print('RESUME skip',key,flush=True); return
        remaining=args.total_hours*3600-(time.monotonic()-started)
        if remaining<=0: raise RuntimeError('invocation budget exhausted')
        directory=out/f'seed-{seed}'; directory.mkdir(exist_ok=True)
        effective=dict(config,seed=seed); cfg=directory/'config.json'; cfg.write_text(json.dumps(effective,indent=2)+'\n')
        command=[sys.executable,'-m','v2.run','--config',str(cfg),'--output',str(directory),'--stage',stage]+(extra or [])
        if stage=='eval' and args.classifier: command+=['--classifier',str(Path(args.classifier).resolve())]
        event={'key':key,'command':command,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}; state['events'].append(event); save()
        print('RUN',key,flush=True); stage_start=time.monotonic()
        with (directory/f'log-{stage}-{len(state["events"])}.txt').open('w') as log:
            try: subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=min(args.stage_hours*3600,remaining))
            except Exception as error:
                event['failure']=str(error); save(); raise
        if stage=='ae': artifacts=[directory/'autoencoder.pt']
        elif stage=='teacher': artifacts=[directory/'teacher.pt']
        elif stage=='distill': artifacts=[directory/f'student_{extra[-1]}.pt']
        else:
            if extra:
                snapshot=directory/'teacher_evaluation.json'; snapshot.write_bytes((directory/'evaluation.json').read_bytes()); artifacts=[snapshot]
            else: artifacts=[directory/'evaluation.json']
        state['completed'][key]={str(f):sha(f) for f in artifacts}; event['completed']=True; event['elapsed_seconds']=time.monotonic()-stage_start; save()
    for seed in seeds:
        run(seed,'ae'); run(seed,'teacher'); run(seed,'eval',['--kind','classical'])
        directory=out/f'seed-{seed}'; evaluation=json.loads((directory/'teacher_evaluation.json').read_text())
        full=[r for r in evaluation['rows'] if r['model']=='teacher' and r['steps']==config['timesteps']]
        if not full: raise RuntimeError('evaluation_steps must include full teacher schedule')
        row=full[0]
        if config['dataset']=='bars_stripes': passed=row['pixel_variance']>0
        else: passed=row.get('label_entropy',0)>=args.min_entropy_fraction*__import__('math').log(10) and __import__('math').isfinite(row.get('fid_like_classifier_features',float('nan'))) and row.get('classifier_mean_confidence',0)>=.8*val['real_mean_confidence']
        (directory/'teacher_gate.json').write_text(json.dumps({'passed':passed,'heuristic_default':True,'row':row,'limitations':'Entropy and finite FID do not certify fidelity. Inspect saved samples before campaign; this gate permits only the pilot.'},indent=2)+'\n')
        if not passed: raise RuntimeError('teacher gate failed; no students trained')
        for kind in ('hybrid','classical','no_entanglement','surrogate'): run(seed,'distill',['--kind',kind])
        run(seed,'eval')
    rows=[]
    for seed in seeds:
        for row in json.loads((out/f'seed-{seed}/evaluation.json').read_text())['rows']: rows.append(dict(row,seed=seed))
    (out/'report.json').write_text(json.dumps({'scope':'pilot, not evidence of quantum advantage','rows':rows,'quality_tolerance_proposal':'No more than 10% worse FID-like than full teacher, label entropy loss <= 0.1 ln(10), and >=2x median end-to-end speedup; confirm across all three seeds. Default research gate, not user-approved scientific threshold.'},indent=2)+'\n')
    print('PILOT COMPLETE',out/'report.json')
if __name__=='__main__': main()
