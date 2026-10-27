"""Weak-label geometric bottleneck versus matched-input unconstrained readout.

Supervision consists only of query proximity predicates. Annotated 3D centers
are withheld from optimization, and used only for an auxiliary geometric audit.
The latent boxes are not assumed to be identifiable physical scene geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cavla.geometric_learning import metrics, pca_fit_transform
from cavla.nuscenes_geometry import rotation


class GeometryHead(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,16))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.bias=nn.Parameter(torch.zeros(()))

    def geometry(self,context,physical):
        z=self.net(context)
        # physical: ray_xy, camera_origin_xy, pinhole_depth_prior, prior_width_length.
        depth=(physical[:,4]*torch.exp(2*torch.tanh(z[:,0]))).clamp(1,180)
        center=physical[:,2:4]+depth[:,None]*physical[:,:2]
        center=center+torch.stack([torch.zeros_like(z[:,1]),8*torch.tanh(z[:,1])],-1)
        offsets=torch.cat([torch.zeros(len(z),1,2,device=z.device),40*torch.tanh(z[:,2:10]).reshape(-1,4,2)],1)
        knots=center[:,None]+offsets
        centers=F.interpolate(knots.transpose(1,2),size=65,mode='linear',align_corners=True).transpose(1,2)[:,1:]
        yaw_knots=np.pi*torch.tanh(z[:,10:15])
        yaws=F.interpolate(yaw_knots[:,None],size=65,mode='linear',align_corners=True)[:,0,1:]
        # Width and length share one multiplicative scale to avoid a yaw/size gauge.
        sizes=physical[:,5:7]*torch.exp(torch.tanh(z[:,15,None]))
        return centers,yaws,sizes

    def forward(self,context,physical,query):
        centers,yaws,sizes=self.geometry(context,physical)
        t=torch.arange(1,65,device=context.device)[None]*.1
        speed=query[:,0,None]*15;k=query[:,1,None]*.03
        angle=speed*k*t
        # Continuous formula at k=0 avoids NaN gradients through torch.where.
        x=speed*t*torch.sinc(angle/np.pi)
        y=speed*t*(angle/2)*torch.sinc(angle/(2*np.pi)).square()
        delta=torch.stack([x,y],-1)-centers
        c=torch.cos(yaws);s=torch.sin(yaws)
        local=torch.stack([c*delta[:,:,0]+s*delta[:,:,1],-s*delta[:,:,0]+c*delta[:,:,1]],-1)
        q=local.abs()-sizes[:,None,[1,0]]/2
        distance=torch.linalg.vector_norm(q.clamp(min=0),dim=-1)+q.max(-1).values.clamp(max=0)
        horizon=query[:,2,None]*6.4
        margin=(distance-2).masked_fill(t>horizon+1e-5,1e4).min(-1).values
        return -margin/.5+self.bias


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--spatial-features',type=Path)
    p.add_argument('--methods',nargs='+',default=['direct_metadata','direct_original','geometry_metadata','geometry_original'])
    p.add_argument('--direct-width',type=int,default=128)
    p.add_argument('--validation',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest={'scope':'Exploratory architecture gate, no established novelty or policy improvement',
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'supervision':'Same binary query labels for all methods; no annotated 3D positions/sizes in training inputs/loss',
        'input':'2D box/class plus camera rays and fixed class-size priors for every method; original-state PCA in designated methods',
        'class_prior_order':'width,length,height',
        'class_priors':{'car':[1.8,4.5,1.6],'person':[.6,.6,1.7],'truck':[2.5,8,3],
                       'bus':[2.5,11,3.2],'bicycle':[.6,1.8,1.4],'motorcycle':[.8,2.2,1.4]},
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01,'batch':256,'steps':args.steps},
        'geometry':'Five time knots, centered oriented rectangle, shared size scale, analytic query geometry; no reactive traffic',
        'selection':('Minimum inner-validation balanced error, checked every 250 updates; fixed threshold .5' if args.validation
                     else 'Final update only; no checkpoint/threshold selection using test'),
        'data_manifest_sha256':hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest(),
        'geometry_records_sha256':hashlib.sha256((args.geometry/'records.json').read_bytes()).hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with np.load(args.data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((args.data/'objects.json').read_text())
    geometry={(r['scene'],r['stem']):r for r in json.loads((args.geometry/'records.json').read_text())}
    priors=manifest['class_priors'];physical=[];true_centers=[]
    for obj in objects:
        r=geometry[obj['scene'],obj['stem']];camera=r['input_camera'];box=r['detection_box']
        uv=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,1])
        ray=rotation(camera['rotation']).apply(np.linalg.solve(np.array(camera['camera_intrinsic']),uv))
        width,length,height=priors[r['label']]
        depth=np.clip(camera['camera_intrinsic'][1][1]*height/max(box[3]-box[1],1),1,180)
        physical.append([*ray[:2],*camera['translation'][:2],depth,width,length])
        true_centers.append(r['agent_centers_xy'])
    physical=np.array(physical,np.float32);true_centers=np.array(true_centers,np.float32)
    spatial=None
    if args.spatial_features:
        spatial_objects=json.loads((args.spatial_features.parent/'objects.json').read_text())
        if [(r['scene'],r['stem']) for r in spatial_objects]!=[(r['scene'],r['stem']) for r in objects]:
            raise ValueError('Spatial feature object ordering mismatch')
        with np.load(args.spatial_features) as z:spatial=z['features'].reshape(len(objects),-1)
    oid=d['object_ids'];query=d['queries'];target=d['targets'];folds=d['object_folds'];modes=d['query_modes']
    validation_candidates=np.array([int(hashlib.sha256((r.get('validation_group',r['scene'])+'/validation').encode()).hexdigest()[:8],16)%5==0 for r in objects])
    outputs=[]
    for fold in range(3):
        train_objects=folds!=fold
        validation_objects=train_objects&validation_candidates if args.validation else np.zeros(len(objects),bool)
        fit_objects=train_objects&~validation_objects
        if args.validation:
            original,om,os,ob=pca_fit_transform(d['originals'],fit_objects)
            np.savez_compressed(args.out/f'fold{fold}_original_pca.npz',mean=om,scale=os,basis=ob)
        else:
            with np.load(args.data/f'fold{fold}_pca.npz') as pca:
                original=((d['originals']-pca['mean0'])/pca['scale0'])@pca['basis0']
        base=np.c_[d['metadata'],physical].astype(np.float32)
        if spatial is not None:
            spatial_pca,spatial_mean,spatial_scale,spatial_basis=pca_fit_transform(spatial,fit_objects)
            np.savez_compressed(args.out/f'fold{fold}_spatial_pca.npz',mean=spatial_mean,scale=spatial_scale,basis=spatial_basis)
        train=fit_objects[oid]&(modes=='id');test=folds[oid]==fold
        validation=validation_objects[oid]&(modes=='id')
        validation_indices=np.flatnonzero(validation)
        (args.out/f'fold{fold}_split.json').write_text(json.dumps({'train_objects':np.flatnonzero(fit_objects).tolist(),
            'validation_objects':np.flatnonzero(validation_objects).tolist(),'test_objects':np.flatnonzero(folds==fold).tolist()},indent=2))
        train_indices=np.flatnonzero(train);test_indices=np.flatnonzero(test)
        y=torch.tensor(target,device='cuda');query_tensor=torch.tensor(query,device='cuda')
        physical_tensor=torch.tensor(physical[oid],device='cuda')
        pw=torch.tensor((1-target[train].mean())/target[train].mean(),device='cuda')
        criterion=nn.BCEWithLogitsLoss(pos_weight=pw)
        available=torch.tensor(train_indices,device='cuda')
        for method in args.methods:
            structured=method.startswith('geometry_')
            features=np.c_[base,original] if method.endswith('_original') else base
            if method.endswith('_spatial'):
                if spatial is None:raise ValueError('Spatial features required')
                features=np.c_[base,spatial_pca]
            mean=features[fit_objects].mean(0);sd=features[fit_objects].std(0);sd[sd<1e-5]=1
            tensor=torch.tensor((features[oid]-mean)/sd,device='cuda')
            for seed in args.seeds:
                torch.manual_seed(seed)
                if structured:model=GeometryHead(features.shape[1]).cuda()
                else:
                    width=args.direct_width
                    model=nn.Sequential(nn.Linear(features.shape[1]+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1)).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                generator=torch.Generator(device='cuda').manual_seed(seed)
                best_state=None;best_score=float('inf');best_step=None;validation_history=[]
                def predict(indices):
                    if structured:return model(tensor[indices],physical_tensor[indices],query_tensor[indices])
                    return model(torch.cat([tensor[indices],query_tensor[indices]],-1)).squeeze(-1)
                model.train()
                for step in range(args.steps):
                    batch=available[torch.randint(len(available),(256,),generator=generator,device='cuda')]
                    optimizer.zero_grad(set_to_none=True);loss=criterion(predict(batch),y[batch]);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
                    if args.validation and (step+1)%250==0:
                        model.eval();val_prob=[]
                        with torch.no_grad():
                            for start in range(0,len(validation_indices),2048):
                                val_prob.append(predict(validation_indices[start:start+2048]).sigmoid().cpu().numpy())
                        score=metrics(target[validation_indices],np.concatenate(val_prob))['balanced_error']
                        validation_history.append({'step':step+1,'balanced_error':score})
                        if score<best_score:
                            best_score=score;best_step=step+1
                            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        model.train()
                if args.validation:
                    if best_state is None:raise ValueError('Validation requires at least 250 steps')
                    model.load_state_dict(best_state)
                    (args.out/f'fold{fold}_{method}_seed{seed}_validation.json').write_text(json.dumps(validation_history,indent=2))
                model.eval();probability=[]
                with torch.no_grad():
                    for start in range(0,len(test_indices),2048):
                        probability.append(predict(test_indices[start:start+2048]).sigmoid().cpu().numpy())
                probability=np.concatenate(probability)
                extra={}
                if structured:
                    unique=np.flatnonzero(folds==fold)
                    with torch.no_grad():
                        context=torch.tensor((features[unique]-mean)/sd,device='cuda')
                        centers,_,_=model.geometry(context,torch.tensor(physical[unique],device='cuda'))
                        extra['auxiliary_center_error_m']=float(np.linalg.norm(centers.cpu().numpy()-true_centers[unique],axis=-1).mean())
                for mode in ['id','ood']:
                    chosen=modes[test_indices]==mode
                    result={'method':method,'fold':fold,'seed':seed,'mode':mode,
                        **metrics(target[test_indices][chosen],probability[chosen]),**extra}
                    if args.validation:result.update(selected_step=best_step,validation_balanced_error=best_score)
                    result['n_parameters']=sum(p.numel() for p in model.parameters())
                    outputs.append(result);print(json.dumps(result),flush=True)
                np.savez_compressed(args.out/f'fold{fold}_{method}_seed{seed}_predictions.npz',indices=test_indices,probability=probability)
                torch.save({'state_dict':model.cpu().state_dict(),'context_mean':torch.from_numpy(mean),'context_std':torch.from_numpy(sd)},
                    args.out/f'fold{fold}_{method}_seed{seed}.pt')
                del model,optimizer
        (args.out/'results.json').write_text(json.dumps(outputs,indent=2))
    summary={name:{mode:{k:float(np.mean([r[k] for r in outputs if r['method']==name and r['mode']==mode])) for k in
        ['auc','balanced_error','false_negative_rate','false_positive_rate']} for mode in ['id','ood']} for name in
        args.methods}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
