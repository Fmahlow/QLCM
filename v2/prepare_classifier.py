"""Prepare an independent classifier from the training split; validate separately."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v1'))
from src.eval_metrics import SmallClassifier
from src.datasets import build_dataset

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['mnist','fashion_mnist'],default='mnist')
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--max-samples',type=int,default=10000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cpu')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.epochs<1 or a.max_samples<2: p.error('epochs>=1 and max-samples>=2 required')
    out=Path(a.output)
    if out.exists() or out.with_suffix('.training.json').exists(): p.error('output exists; reuse and validate it, or choose a new path')
    if a.dry_run:
        print(json.dumps({'dataset':a.dataset,'split':'train only','epochs':a.epochs,'max_samples':a.max_samples,'seed':a.seed,'device':a.device,'output':str(out),'downloads_data':False,'trains':False,'next':'python -m v2.validate_classifier --checkpoint '+str(out)+' --dataset '+a.dataset+' --output '+str(out.with_suffix('.validation.json'))},indent=2))
        return
    torch.manual_seed(a.seed); torch.set_num_threads(2)
    model=SmallClassifier().to(a.device)
    dataset=build_dataset(a.dataset,root=str(ROOT/'data'),train=True,max_samples=a.max_samples)
    loader=DataLoader(dataset,batch_size=128,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    model.train(); losses=[]
    for _ in range(a.epochs):
        for x,y in loader:
            loss=F.cross_entropy(model(x.to(a.device)),y.to(a.device))
            if not torch.isfinite(loss): raise RuntimeError('nonfinite classifier loss; no checkpoint saved')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),out)
    out.with_suffix('.training.json').write_text(json.dumps({'dataset':a.dataset,'train_split_only':True,'samples':len(dataset),'epochs':a.epochs,'seed':a.seed,'device':a.device,'losses':losses},indent=2)+'\n')
    print('Prepared classifier:',out,'; held-out validation still required')
if __name__=='__main__': main()
