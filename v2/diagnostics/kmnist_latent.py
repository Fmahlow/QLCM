"""E4b of the 2026-10-20 preregistration: is KMNIST quality limited by the AE latent size?

Run from repository root:
  python v2/diagnostics/kmnist_latent.py --output results/v2-e4b-kmnist-latent
Reconstruction FID-like of the AE (512 test images, encode -> decode) for latent 16 (campaign AE) and for new
AEs with latent 32 and 64; for the new AEs, a hybrid teacher (hidden 128) and a classical student, evaluated as
in the campaign. The real-image row (512 other test images) is the floor of the metric.
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'v1'))
from v2.run import ae,load
from src.datasets import build_dataset
from src.eval_metrics import SmallClassifier,real_reference_features,evaluate_generated_samples

def stage(d,config,name,extra):
    with (d/f'log-{name}.txt').open('w') as log:
        subprocess.run([sys.executable,'-m','v2.run','--config',str(d/config),'--output',str(d),'--stage',name]+extra,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',d,name,flush=True)

def main():
    a=argparse.ArgumentParser(); a.add_argument('--output',required=True); a.add_argument('--campaign',default='results/v2-campaign-datasets-teachers')
    a.add_argument('--config',default='v2/gpu_pilot.json'); a.add_argument('--classifier',default='results/v2-metrics/kmnist_classifier.pt'); a.add_argument('--seeds',default='0,1,2')
    a.add_argument('--latents',default='32,64'); a.add_argument('--train-device',default='cpu'); args=a.parse_args(); out=Path(args.output).resolve(); base=json.loads(Path(args.config).read_text())
    seeds=[int(s) for s in args.seeds.split(',')]; sizes=[int(v) for v in args.latents.split(',')]
    def pipeline(seed,latent):
        d=out/f'latent-{latent}'/f'seed-{seed}'; d.mkdir(parents=True,exist_ok=True)
        config=dict(base,dataset='kmnist',seed=seed,latent=latent,teacher_kind='hybrid',student_epochs=200,ema_decay=.95,device=args.train_device)
        (d/'config.json').write_text(json.dumps(config,indent=2)+'\n'); (d/'config_eval.json').write_text(json.dumps(dict(config,device='cuda'),indent=2)+'\n')
        if not (d/'autoencoder.pt').exists(): stage(d,'config.json','ae',[])
        if not (d/'teacher.pt').exists(): stage(d,'config.json','teacher',[])
        if not (d/'student_classical.pt').exists(): stage(d,'config.json','distill',['--kind','classical'])
    with ThreadPoolExecutor(len(seeds)*len(sizes)) as pool: list(pool.map(lambda job:pipeline(*job),[(s,l) for s in seeds for l in sizes]))
    for seed in seeds:
        for latent in sizes:
            d=out/f'latent-{latent}'/f'seed-{seed}'
            if not (d/'evaluation.json').exists(): stage(d,'config_eval.json','eval',['--classifier',str(Path(args.classifier).resolve())])
    device=torch.device('cuda'); classifier=SmallClassifier().to(device); classifier.load_state_dict(load(Path(args.classifier),device)); classifier.eval()
    real=real_reference_features(classifier,'kmnist',device); test=build_dataset('kmnist',root=str(ROOT/'data'),train=False)
    images=torch.stack([test[i][0] for i in range(1000,1512)]).to(device); rows=[]
    rows.append({'model':'real test images','latent':None,'seed':None,**evaluate_generated_samples(classifier,images,real,device)})
    for seed in seeds:
        for latent in [base['latent']]+sizes:
            path=Path(args.campaign)/'kmnist'/f'seed-{seed}'/'ae'/'autoencoder.pt' if latent==base['latent'] else out/f'latent-{latent}'/f'seed-{seed}'/'autoencoder.pt'
            model=ae(dict(base,dataset='kmnist',latent=latent)).to(device); model.load_state_dict(load(path,device)['state']); model.eval()
            with torch.no_grad(): recon=model.decode(model.encode(images))
            rows.append({'model':'ae reconstruction','latent':latent,'seed':seed,**evaluate_generated_samples(classifier,recon,real,device)})
            print('recon',latent,seed,round(rows[-1]['fid_like_classifier_features'],1),flush=True)
    generation=[]
    for seed in seeds:
        for latent in sizes:
            for row in json.loads((out/f'latent-{latent}'/f'seed-{seed}'/'evaluation.json').read_text())['rows']: generation.append(dict(row,latent=latent,seed=seed))
    (out/'report.json').write_text(json.dumps({'scope':'E4b of the 2026-10-20 preregistration; descriptive','reconstruction':rows,'generation':generation},indent=2)+'\n')
    print('E4B COMPLETE',out/'report.json',flush=True)
if __name__=='__main__': main()
