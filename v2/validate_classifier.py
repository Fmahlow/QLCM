"""Evaluate an existing v1 SmallClassifier on the entire held-out test split."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'v1'))
from src.eval_metrics import SmallClassifier
from src.datasets import build_dataset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--dataset',choices=['mnist','fashion_mnist'],required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cpu'); a=p.parse_args()
    model=SmallClassifier().to(a.device); model.load_state_dict(torch.load(a.checkpoint,map_location=a.device,weights_only=True)); model.eval()
    dataset=build_dataset(a.dataset,root=str(ROOT/'data'),train=False); correct=total=0; confidence_sum=0.0
    with torch.no_grad():
        for x,y in DataLoader(dataset,batch_size=256):
            probs=model(x.to(a.device)).softmax(1); correct+=int((probs.argmax(1).cpu()==y).sum()); total+=len(y); confidence_sum+=float(probs.max(1).values.sum())
    result={'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'dataset':a.dataset,'heldout':True,'test_accuracy':correct/total,'samples':total,'real_mean_confidence':confidence_sum/total}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
if __name__=='__main__': main()
