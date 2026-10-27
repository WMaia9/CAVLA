"""Cross-scene feasibility-screen baseline using local 3D annotation candidates.

Learns a geometric proximity predicate for synthetic query curves against logged
agent motion, not safety, action optimality, or reactive counterfactual futures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch
from torch import nn

from cavla.nuscenes_geometry import signed_point_box_distance


def query_curve(speed,curvature):
    t=np.arange(1,65)*.1
    if curvature==0:return np.stack([speed*t,np.zeros_like(t)],-1)
    angle=speed*curvature*t
    return np.stack([np.sin(angle)/curvature,(1-np.cos(angle))/curvature],-1)


def auc(y,p):
    y=np.asarray(y,dtype=bool);n=y.sum();m=len(y)-n
    return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def metrics(y,p):
    y=np.asarray(y,dtype=bool);pred=p>=.5
    fn=float((~pred[y]).mean()) if y.any() else None
    fp=float(pred[~y].mean()) if (~y).any() else None
    return {'auc':auc(y,p),'balanced_error':(fn+fp)/2 if fn is not None and fp is not None else None,
            'false_negative_rate':fn,'false_positive_rate':fp,'positive_rate':float(y.mean()),'n':len(y)}


def pca_fit_transform(x,train,rank=16):
    mean=x[train].mean(0);scale=np.sqrt(np.square(x[train]-mean).mean(0).mean()+1e-12)
    z=(x-mean)/scale
    _,_,vt=np.linalg.svd(z[train],full_matrices=False)
    basis=vt[:rank].T
    return z@basis,mean,scale,basis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    manifest={'scope':'Baseline gate, not a novel ICML method',
       'geometry_sha256':hashlib.sha256((args.geometry/'records.json').read_bytes()).hexdigest(),
       'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
       'target':'Minimum centerline-to-target-box distance <= 2m in the query horizon',
       'folds':'Three folds, deterministic SHA256(scene) modulo 3; all objects/queries from a scene stay together',
       'train_grid':{'speed':[2,5,10,15],'curvature':[-.03,-.015,0,.015,.03],'horizon':[2,4,6.4]},
       'ood_grid':{'speed':[3.5,7.5,12.5],'curvature':[-.0225,-.0075,.0075,.0225],'horizon':[3,5]},
       'learner':{'hidden':[128,128],'optimizer':'AdamW','lr':.001,'weight_decay':.01,'batch':256,
                  'loss':'BCEWithLogitsLoss positive weight n_negative/n_positive on train only',
                  'pca_rank':16,'threshold':.5},
       'limitations':['Automatic joins not verified edit identities','Query curves need not match current velocity or road rules',
         'Agent motions are logged and nonreactive','2m threshold is a screening convention, not collision ground truth',
         'Features are from archived padded-front Alpamayo-1; no instruction conditioning',
         'This trains readout heads, not the VLA backbone']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    rows=json.loads((args.geometry/'records.json').read_text())
    records=[r for r in rows if r['status']=='automatic_3d_candidate' and r['valid_future_steps']==64]
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/comp_results_fin_comb')
    labels=['car','person','truck','bus','bicycle','motorcycle']
    metadata=[];originals=[];deltas=[];input_hashes={}
    for r in records:
        folder=root/r['scene']/'whitebox';orig=folder/'original_traj_token.npz';edit=folder/f"{r['stem']}_traj_token.npz"
        with np.load(orig) as z:a=z['traj_token_hidden'].reshape(-1)
        with np.load(edit) as z:b=z['traj_token_hidden'].reshape(-1)
        originals.append(a);deltas.append(b-a)
        x0,y0,x1,y1=r['detection_box']
        metadata.append([x0/1600,y0/900,x1/1600,y1/900,*[float(r['label']==c) for c in labels]])
        for path in [orig,edit]:input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    metadata=np.array(metadata,np.float32);originals=np.array(originals,np.float32);deltas=np.array(deltas,np.float32)
    folds=np.array([int(hashlib.sha256(r['scene'].encode()).hexdigest()[:8],16)%3 for r in records])
    rng=np.random.default_rng(20261021);random_features=rng.normal(size=(len(records),32)).astype(np.float32)
    object_ids=[];targets=[];queries=[];query_modes=[]
    for i,r in enumerate(records):
        centers=np.array(r['agent_centers_xy']);yaws=np.array(r['agent_yaws']);sizes=np.array(r['agent_sizes'])
        for mode,speeds,curvatures,horizons in [('id',[2,5,10,15],[-.03,-.015,0,.015,.03],[2,4,6.4]),
                                             ('ood',[3.5,7.5,12.5],[-.0225,-.0075,.0075,.0225],[3,5])]:
            for speed in speeds:
                for curvature in curvatures:
                    clearance=signed_point_box_distance(query_curve(speed,curvature),centers,yaws,sizes)
                    for horizon in horizons:
                        object_ids.append(i);targets.append(float(clearance[:round(horizon*10)].min()<=2))
                        queries.append([speed/15,curvature/.03,horizon/6.4]);query_modes.append(mode)
    object_ids=np.array(object_ids);targets=np.array(targets,np.float32);queries=np.array(queries,np.float32);query_modes=np.array(query_modes)
    np.savez_compressed(args.out/'dataset.npz',metadata=metadata,originals=originals,deltas=deltas,
        object_ids=object_ids,targets=targets,queries=queries,query_modes=query_modes,object_folds=folds)
    (args.out/'objects.json').write_text(json.dumps([{'scene':r['scene'],'stem':r['stem'],'fold':int(f)} for r,f in zip(records,folds)],indent=2))
    (args.out/'feature_hashes.json').write_text(json.dumps(input_hashes,indent=2))
    print(json.dumps({'objects':len(records),'scenes':len({r['scene'] for r in records}),'queries':len(targets),
                      'positive_rate':float(targets.mean())}),flush=True)
    results=[];prediction_rows=[]
    for fold in range(3):
        train_objects=folds!=fold
        projected_original,mean0,scale0,basis0=pca_fit_transform(originals,train_objects)
        projected_delta,mean1,scale1,basis1=pca_fit_transform(deltas,train_objects)
        np.savez_compressed(args.out/f'fold{fold}_pca.npz',mean0=mean0,scale0=scale0,basis0=basis0,mean1=mean1,scale1=scale1,basis1=basis1)
        designs={'metadata':metadata,'metadata_random':np.c_[metadata,random_features],
                 'original':np.c_[metadata,projected_original],
                 'original_delta':np.c_[metadata,projected_original,projected_delta]}
        train=(folds[object_ids]!=fold)&(query_modes=='id')
        test=(folds[object_ids]==fold)
        train_indices=np.flatnonzero(train);test_indices=np.flatnonzero(test)
        y=torch.tensor(targets,device='cuda');pos_weight=torch.tensor((1-targets[train].mean())/targets[train].mean(),device='cuda')
        for name,features in designs.items():
            design=np.c_[features[object_ids],queries].astype(np.float32)
            mean=design[train].mean(0);sd=design[train].std(0);sd[sd<1e-5]=1
            tensor=torch.tensor((design-mean)/sd,device='cuda')
            for seed in args.seeds:
                torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
                model=nn.Sequential(nn.Linear(design.shape[1],128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1)).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                generator=torch.Generator(device='cuda').manual_seed(seed)
                available=torch.tensor(train_indices,device='cuda')
                model.train()
                for step in range(args.steps):
                    batch=available[torch.randint(len(available),(256,),generator=generator,device='cuda')]
                    optimizer.zero_grad(set_to_none=True)
                    loss=criterion(model(tensor[batch]).squeeze(-1),y[batch]);loss.backward();optimizer.step()
                model.eval()
                with torch.no_grad():pred=model(tensor[test]).squeeze(-1).sigmoid().cpu().numpy()
                for mode in ['id','ood']:
                    chosen=query_modes[test]==mode
                    result={'method':name,'seed':seed,'fold':fold,'mode':mode,**metrics(targets[test][chosen],pred[chosen])}
                    results.append(result);print(json.dumps(result),flush=True)
                np.savez_compressed(args.out/f'fold{fold}_{name}_seed{seed}_predictions.npz',indices=test_indices,probability=pred)
                torch.save({'state_dict':model.cpu().state_dict(),'input_mean':torch.from_numpy(mean),'input_std':torch.from_numpy(sd)},
                           args.out/f'fold{fold}_{name}_seed{seed}.pt')
                del model,optimizer
        (args.out/'results.json').write_text(json.dumps(results,indent=2))
    summary={}
    for name in designs:
        summary[name]={mode:{k:float(np.mean([r[k] for r in results if r['method']==name and r['mode']==mode])) for k in
            ['auc','balanced_error','false_negative_rate','false_positive_rate']} for mode in ['id','ood']}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
