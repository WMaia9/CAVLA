"""Learn algebraic deletion responses from original plus one singleton bit."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from cavla.geometric_calibration import optimal_threshold, threshold_metrics
from cavla.geometric_learning import metrics, pca_fit_transform
from cavla.nuscenes_geometry import rotation
from cavla.partial_composition_models import FactorModel, SetModel, or_logits, joint_pair_nll


METHODS=['deepsets','mlp_or','geometry_or','mlp_joint','geometry_joint']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--reference-study',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    p.add_argument('--methods',nargs='+',default=METHODS,choices=METHODS)
    p.add_argument('--metadata-only',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    reference=json.loads((args.reference_study/'manifest.json').read_text());rc=reference['config']
    with np.load(args.data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((args.data/'objects.json').read_text());scenes=json.loads((args.data/'scenes.json').read_text())
    feature_objects=json.loads((args.features.parent/'objects.json').read_text())
    assert [(r['scene'],r['stem']) for r in objects]==[(r['scene'],r['stem']) for r in feature_objects]
    with np.load(args.features) as z:features=z['features'].reshape(len(objects),-1)
    geometries={(r['scene'],r['stem']):r for r in json.loads((Path(rc['geometry'])/'records.json').read_text())}
    physical=[];metadata=[];classes=['car','person','truck','bus','bicycle','motorcycle']
    for obj in objects:
        r=geometries[obj['scene'],obj['stem']];camera=r['input_camera'];box=r['detection_box']
        uv=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,1])
        ray=rotation(camera['rotation']).apply(np.linalg.solve(np.array(camera['camera_intrinsic']),uv))
        width,length,height=reference['class_priors'][r['label']]
        depth=np.clip(camera['camera_intrinsic'][1][1]*height/max(box[3]-box[1],1),1,180)
        physical.append([*ray[:2],*camera['translation'][:2],depth,width,length])
        metadata.append([box[0]/1600,box[1]/900,box[2]/1600,box[3]/900,*[float(r['label']==c) for c in classes]])
    physical=np.array(physical,np.float32);base=np.c_[metadata,physical].astype(np.float32)
    manifest={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'scope':'Partial composed-label architecture pilot; no physical compound edit or ICML novelty established',
        'supervision':'Two composed bits per training scene/ID query; individual targets never used in optimization',
        'objective':'Unweighted mean pair BCE or exact joint NLL / 2; independent latent Bernoulli modeling assumption',
        'selection':'Minimum validation weak-pair ID BER at .5 each 250 steps, then threshold calibration on same validation pairs',
        'optimizer':{'name':'AdamW','lr':.001,'weight_decay':.01,'batch_groups':64,'clip':10},
        'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_COMPOSICAO_PARCIAL_V1.md').read_bytes()).hexdigest(),
        'dataset_sha256':hashlib.sha256((args.data/'dataset.npz').read_bytes()).hexdigest(),
        'features_sha256':hashlib.sha256(args.features.read_bytes()).hexdigest()}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'PROTOCOL.md').write_bytes(Path('docs/PROTOCOLO_COMPOSICAO_PARCIAL_V1.md').read_bytes())
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'models_snapshot.py').write_bytes(Path(__file__).with_name('partial_composition_models.py').read_bytes())
    ids=d['scene_object_ids'];present_np=ids>=0;safe=ids.clip(min=0)
    nq=len(d['queries']);ns=len(scenes)
    group_scene=np.repeat(np.arange(ns),nq);group_query=np.tile(np.arange(nq),ns)
    present=torch.tensor(present_np,device='cuda');phys=torch.tensor(physical[safe],device='cuda')
    query=torch.tensor(d['queries'],device='cuda');removed=torch.tensor(d['removed_index'],device='cuda')
    weak=torch.tensor(d['weak_labels'],device='cuda');case_keep=torch.tensor(d['case_keep'],device='cuda')
    gs=torch.tensor(group_scene,device='cuda');gq=torch.tensor(group_query,device='cuda')
    case_rows=np.arange(len(d['case_scenes'])*nq)
    row_cases=case_rows//nq;row_queries=case_rows%nq;row_scenes=d['case_scenes'][row_cases]
    target=d['case_targets'].reshape(-1);card=d['case_cardinality'][row_cases]
    ambiguous=(d['oracle_lower']!=d['oracle_upper']).reshape(-1)
    outputs=[]
    for fold in range(3):
        split=json.loads((args.reference_study/f'fold{fold}_split.json').read_text())
        fit=np.isin(np.arange(len(objects)),split['train_objects'])
        train_scenes=np.array([objects[i]['scene'] for i in split['train_objects']])
        val_scenes=np.array([objects[i]['scene'] for i in split['validation_objects']])
        scene_names=np.array([r['scene'] for r in scenes])
        train=np.isin(scene_names[group_scene],train_scenes)&(d['query_modes'][group_query]=='id')
        vi=np.flatnonzero(np.isin(scene_names[group_scene],val_scenes)&(d['query_modes'][group_query]=='id'))
        ti=np.flatnonzero(d['scene_folds'][row_scenes]==fold)
        available=torch.tensor(np.flatnonzero(train),device='cuda')
        if args.metadata_only:
            sp=np.empty((len(objects),0),np.float32);pm=np.empty(0,np.float32);ps=1.;pb=np.empty((0,0),np.float32)
        else:sp,pm,ps,pb=pca_fit_transform(features,fit)
        fo=np.c_[base,sp].astype(np.float32);mean=fo[fit].mean(0);sd=fo[fit].std(0);sd[sd<1e-5]=1
        context=torch.tensor(((fo-mean)/sd)[safe],device='cuda')
        np.savez_compressed(args.out/f'fold{fold}_pca.npz',mean=pm,scale=ps,basis=pb,
            context_mean=mean,context_std=sd)
        (args.out/f'fold{fold}_split.json').write_text(json.dumps(split,indent=2))
        vy=d['weak_labels'][group_scene[vi],group_query[vi]].reshape(-1).astype(float)
        for method in args.methods:
            for seed in args.seeds:
                torch.manual_seed(seed)
                model=(SetModel(fo.shape[1]) if method=='deepsets' else FactorModel(fo.shape[1],method.startswith('geometry'))).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                generator=torch.Generator(device='cuda').manual_seed(seed)
                def predict_pair(group_ids):
                    si=gs[group_ids];qi=gq[group_ids];keep=present[si]
                    without=keep.clone();without.scatter_(1,removed[si,None],False)
                    if method=='deepsets':
                        logits=torch.stack([model(context[si],query[qi],keep),model(context[si],query[qi],without)],-1)
                        return logits,None
                    factors=model(context[si],phys[si],query[qi],keep)
                    return torch.stack([or_logits(factors,keep),or_logits(factors,without)],-1),factors
                def validation():
                    probs=[]
                    with torch.no_grad():
                        for start in range(0,len(vi),512):probs.append(predict_pair(vi[start:start+512])[0].sigmoid().cpu().numpy())
                    return np.concatenate(probs)
                best_state=None;best_score=float('inf');best_step=None;history=[];start_time=time.perf_counter()
                for step in range(args.steps):
                    model.train();batch=available[torch.randint(len(available),(64,),generator=generator,device='cuda')]
                    pair_logits,factors=predict_pair(batch);observed=weak[gs[batch],gq[batch]]
                    if method.endswith('_joint'):
                        loss=joint_pair_nll(factors,present[gs[batch]],removed[gs[batch]],observed).mean()/2
                    else:loss=F.binary_cross_entropy_with_logits(pair_logits,observed.float())
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
                    if (step+1)%250==0:
                        model.eval();vp=validation();score=metrics(vy,vp.reshape(-1))['balanced_error']
                        history.append({'step':step+1,'balanced_error':score})
                        if score<best_score:
                            best_score=score;best_step=step+1
                            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if best_state is None:raise ValueError('At least 250 steps required')
                fit_seconds=time.perf_counter()-start_time
                model.load_state_dict(best_state);model.eval();vp=validation()
                threshold,calibration_score=optimal_threshold(vy,vp.reshape(-1))
                probability=[];inference_start=time.perf_counter()
                with torch.no_grad():
                    if method!='deepsets':
                        factor_cache=torch.zeros((ns,nq,ids.shape[1]),device='cuda')
                        test_groups=np.flatnonzero(d['scene_folds'][group_scene]==fold)
                        for start in range(0,len(test_groups),256):
                            g=test_groups[start:start+256];si=gs[g];qi=gq[g]
                            factor_cache[si,qi]=model(context[si],phys[si],query[qi],present[si])
                    for start in range(0,len(ti),1024):
                        i=ti[start:start+1024];ci=row_cases[i];si=row_scenes[i];qi=row_queries[i];keep=case_keep[ci]
                        logit=model(context[si],query[qi],keep) if method=='deepsets' else or_logits(factor_cache[si,qi],keep)
                        probability.append(logit.sigmoid().cpu().numpy())
                probability=np.concatenate(probability);inference_seconds=time.perf_counter()-inference_start
                masks={'original':card[ti]==0,'single':card[ti]==1,'double':card[ti]==2,'triple':card[ti]==3,
                    'compound':card[ti]>=2,'compound_ambiguous':(card[ti]>=2)&ambiguous[ti],
                    'compound_identified':(card[ti]>=2)&~ambiguous[ti]}
                for name,mask in masks.items():
                    for mode in ['id','ood']:
                        use=mask&(d['query_modes'][row_queries[ti]]==mode)
                        if not use.any():continue
                        outputs.append({'method':method,'fold':fold,'seed':seed,'view':name,'mode':mode,
                            **threshold_metrics(target[ti][use],probability[use],threshold),
                            'fixed_threshold_balanced_error':metrics(target[ti][use],probability[use])['balanced_error'],
                            'selected_step':best_step,'threshold':threshold,'validation_balanced_error':best_score,
                            'validation_calibrated_balanced_error':calibration_score,'fit_seconds':fit_seconds,
                            'inference_seconds':inference_seconds,'n_parameters':sum(p.numel() for p in model.parameters())})
                stem=f'fold{fold}_{method}_seed{seed}'
                np.savez_compressed(args.out/f'{stem}_predictions.npz',indices=ti,probability=probability,
                    threshold=np.array(threshold),validation_group_ids=vi,validation_probability=vp)
                torch.save({'state_dict':best_state},args.out/f'{stem}.pt')
                (args.out/f'{stem}_validation.json').write_text(json.dumps(history,indent=2))
                main_row=next(r for r in reversed(outputs) if r['method']==method and r['fold']==fold and r['seed']==seed and r['view']=='compound' and r['mode']=='ood')
                print(json.dumps(main_row),flush=True)
                (args.out/'results.json').write_text(json.dumps(outputs,indent=2));del model,optimizer
    def average(method,view,mode,key):
        values=[r[key] for r in outputs if r['method']==method and r['view']==view and r['mode']==mode and r[key] is not None]
        return float(np.mean(values)) if values else None
    summary={m:{v:{mode:{k:average(m,v,mode,k)
        for k in ['auc','balanced_error','false_negative_rate','false_positive_rate','fixed_threshold_balanced_error']}
        for mode in ['id','ood']} for v in masks} for m in args.methods}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
