"""Apply the 2026-10-20 preregistered criteria to the E1/E2 sweep reports and print markdown tables.

Run from repository root:
  python v2/diagnostics/summarize_extension.py --e1 results/v2-e1-capacity/report.json --e2 results/v2-e2-scaling/report.json
"""
import argparse
import json
import statistics
a_min_seeds=3
from collections import defaultdict

def table(report):
    # (dataset, pipeline-config, model role, steps) -> {seed: fid}
    out=defaultdict(dict); params={}
    for r in json.load(open(report))['rows']:
        role='teacher' if r['model']=='teacher' else 'student'
        key=(r['dataset'],r['kind'],r['hidden'],r['qubits'],r['depth'],r['reupload'],role,r['steps'])
        out[key][r['seed']]=r['fid_like_classifier_features']; params[key]=r['parameters']
    return out,params

def fmt(values): return f"{statistics.mean(values):.1f} ± {statistics.pstdev(values):.1f}" if values else '—'

def verdict(target,control):
    """Preregistered: target better in every paired seed and mean at least 10% lower."""
    seeds=sorted(set(target)&set(control))
    if len(seeds)<a_min_seeds: return 'incompleto'
    wins=all(target[s]<control[s] for s in seeds); gain=1-statistics.mean(target[s] for s in seeds)/statistics.mean(control[s] for s in seeds)
    return f"{'SIM' if wins and gain>=.10 else 'não'} ({gain*100:+.0f}%)"

def e1(path):
    data,params=table(path)
    for dataset in sorted({k[0] for k in data}):
        print(f'\n### E1 {dataset}\n')
        print('| hidden | role/passos | hybrid | surrogate | classical_branch | classical | circuito contribui? | entrelaçamento contribui? |')
        print('|---|---|---|---|---|---|---|---|')
        for role,steps in (('student',1),('student',4),('teacher',32)):
            for h in (0,2,4,8,16,32,128):
                cell=lambda kind:data.get((dataset,kind,h,4,2,False,role,steps),{})
                if not any(cell(k) for k in ('hybrid','surrogate','classical_branch','classical')): continue
                hy=cell('hybrid')
                print(f"| {h} | {role} {steps} | {fmt(list(hy.values()))} | {fmt(list(cell('surrogate').values()))} | {fmt(list(cell('classical_branch').values()))} | {fmt(list(cell('classical').values()))} | "
                      f"{verdict(hy,cell('classical_branch')) if hy else '—'} | {verdict(hy,cell('surrogate')) if hy else '—'} |")
        print(f'\nFronteira (estudante 1 passo): hybrid contra o melhor classical com <= parâmetros\n')
        print('| hidden hybrid | parâmetros | hybrid | melhor classical (hidden, parâmetros) | classical |'); print('|---|---|---|---|---|')
        for h in (0,4,8,16,32,128):
            key=(dataset,'hybrid',h,4,2,False,'student',1)
            if key not in data: continue
            options=[(statistics.mean(v.values()),k[2],params[k]) for k,v in data.items() if k[0]==dataset and k[1]=='classical' and k[6:]==('student',1) and params[k]<=params[key] and v]
            best=min(options) if options else None
            print(f"| {h} | {params[key]} | {fmt(list(data[key].values()))} | {f'{best[1]}, {best[2]}' if best else 'nenhum'} | {f'{best[0]:.1f}' if best else '—'} |")

def e2(path):
    data,params=table(path)
    print('\n### E2 mnist, hidden=0\n')
    print('| qubits | prof. | reupload | role/passos | hybrid | surrogate | classical_branch | circuito contribui? | entrelaçamento contribui? | parâmetros hybrid |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for role,steps in (('student',1),('teacher',32)):
        for q in (4,6,8):
            for d in (2,4):
                for re in (False,True):
                    hy=data.get(('mnist','hybrid',0,q,d,re,role,steps),{}); su=data.get(('mnist','surrogate',0,q,d,re,role,steps),{}); cb=data.get(('mnist','classical_branch',0,q,2,False,role,steps),{})
                    if not hy: continue
                    print(f"| {q} | {d} | {'sim' if re else 'não'} | {role} {steps} | {fmt(list(hy.values()))} | {fmt(list(su.values()))} | {fmt(list(cb.values()))} | {verdict(hy,cb)} | {verdict(hy,su)} | {params[('mnist','hybrid',0,q,d,re,role,steps)]} |")

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--e1'); p.add_argument('--e2'); a=p.parse_args()
    if a.e1: e1(a.e1)
    if a.e2: e2(a.e2)
