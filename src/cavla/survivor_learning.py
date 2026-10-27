"""Matched-budget original, editor augmentation, noise and consistency pilots.

Only surviving-object appearance invariance is supervised. This is not a
physical counterfactual removal learner or a new-method novelty claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cavla.geometric_calibration import optimal_threshold, threshold_metrics
from cavla.geometric_learning import metrics, pca_fit_transform
from cavla.nuscenes_geometry import rotation
from cavla.structured_geometry_learning import GeometryHead


METHODS=['original_repeat','editor_augmentation','norm_matched_noise','paired_consistency']


def norm_matched_noise(delta_norm,dimension,generator):
    noise=torch.randn((len(delta_norm),dimension),device=delta_norm.device,generator=generator)
    return noise/torch.linalg.vector_norm(noise,dim=-1,keepdim=True).clamp(min=1e-12)*delta_norm[:,None]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-study',type=Path,required=True)
    p.add_argument('--original-features',type=Path,required=True)
    p.add_argument('--edited-features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43])
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    reference=json.loads((args.reference_study/'manifest.json').read_text())
    config=reference['config'];data=Path(config['data']);geometry_path=Path(config['geometry'])
    objects=json.loads((data/'objects.json').read_text())
    other_objects=json.loads((args.edited_features.parent/'objects.json').read_text())
    assert objects==other_objects
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    with np.load(args.original_features) as z:original=z['features'].reshape(len(objects),-1)
    with np.load(args.edited_features) as z:
        edited=z['features'].reshape(len(objects),-1);eligible=z['eligible']
    assert np.array_equal(original[~eligible],edited[~eligible])
    geometries={(r['scene'],r['stem']):r for r in json.loads((geometry_path/'records.json').read_text())}
    physical=[]
    for obj in objects:
        r=geometries[obj['scene'],obj['stem']];camera=r['input_camera'];box=r['detection_box']
        uv=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,1])
        ray=rotation(camera['rotation']).apply(np.linalg.solve(np.array(camera['camera_intrinsic']),uv))
        width,length,height=reference['class_priors'][r['label']]
        depth=np.clip(camera['camera_intrinsic'][1][1]*height/max(box[3]-box[1],1),1,180)
        physical.append([*ray[:2],*camera['translation'][:2],depth,width,length])
    physical=np.array(physical,np.float32);base=np.c_[d['metadata'],physical].astype(np.float32)
    manifest={'scope':'Exploratory survivor appearance-robustness test; not verified physical interventions',
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'methods':METHODS,'consistency_weight':1.,'consistency_loss':'Mean squared probability difference',
        'noise':'Isotropic, per-update random direction, same per-object L2 norm in unnormalized PCA space as train editor delta',
        'pca':'Rank 16 fit only on original fit objects, excludes validation and test',
        'selection':'Original ID validation BER at 0.5 every 250 updates, then original validation threshold calibration',
        'training':'Two forward inputs per query and identical batches/steps. BCE positives weighted by original train prevalence. No new independent labels.',
        'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_PARES_SOBREVIVENTES_V1.md').read_bytes()).hexdigest(),
        'edited_features_sha256':hashlib.sha256(args.edited_features.read_bytes()).hexdigest(),
        'original_features_sha256':hashlib.sha256(args.original_features.read_bytes()).hexdigest(),
        'n_survivors':int(eligible.sum()),'limitations':['Same previously explored sample',
            'Original logged nonreactive futures used for labels','No target-absence supervision or verified removal semantics']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    oid=d['object_ids'];modes=d['query_modes'];target=d['targets'];folds=d['object_folds']
    y=torch.tensor(target,device='cuda');query=torch.tensor(d['queries'],device='cuda')
    phys=torch.tensor(physical[oid],device='cuda');rows=[]
    for fold in range(3):
        split=json.loads((args.reference_study/f'fold{fold}_split.json').read_text())
        groups=[{objects[i]['log_token'] for i in split[key]} for key in ['train_objects','validation_objects','test_objects']]
        for a,b in [(0,1),(0,2),(1,2)]:assert not groups[a]&groups[b]
        fit=np.isin(np.arange(len(objects)),split['train_objects'])
        train=fit[oid]&(modes=='id');vi=np.flatnonzero(np.isin(oid,split['validation_objects'])&(modes=='id'))
        ti=np.flatnonzero(folds[oid]==fold);train_indices=np.flatnonzero(train)
        sp,pm,ps,pb=pca_fit_transform(original,fit)
        se=((edited-pm)/ps)@pb
        np.savez_compressed(args.out/f'fold{fold}_pca.npz',mean=pm,scale=ps,basis=pb)
        (args.out/f'fold{fold}_split.json').write_text(json.dumps(split,indent=2))
        fo=np.c_[base,sp].astype(np.float32);fe=np.c_[base,se].astype(np.float32)
        mean=fo[fit].mean(0);sd=fo[fit].std(0);sd[sd<1e-5]=1
        co=torch.tensor((fo[oid]-mean)/sd,device='cuda')
        ce=torch.tensor((fe[oid]-mean)/sd,device='cuda')
        delta_norm=torch.tensor(np.linalg.norm(se-sp,axis=-1)[oid],device='cuda')
        spatial_sd=torch.tensor(sd[-sp.shape[1]:],device='cuda')
        available=torch.tensor(train_indices,device='cuda')
        pw=torch.tensor((1-target[train].mean())/target[train].mean(),device='cuda')
        criterion=nn.BCEWithLogitsLoss(pos_weight=pw)
        for method in METHODS:
            for seed in args.seeds:
                torch.manual_seed(seed);model=GeometryHead(fo.shape[1]).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                batches=torch.Generator(device='cuda').manual_seed(seed)
                noise_rng=torch.Generator(device='cuda').manual_seed(seed+10000)
                def infer(indices,context):
                    probability=[]
                    with torch.no_grad():
                        for start in range(0,len(indices),2048):
                            i=indices[start:start+2048]
                            probability.append(model(context[i],phys[i],query[i]).sigmoid().cpu().numpy())
                    return np.concatenate(probability)
                best_state=None;best_score=float('inf');best_step=None;history=[]
                for step in range(args.steps):
                    model.train();batch=available[torch.randint(len(available),(256,),generator=batches,device='cuda')]
                    original_logits=model(co[batch],phys[batch],query[batch])
                    if method=='original_repeat':aug=co[batch]
                    elif method=='norm_matched_noise':
                        aug=co[batch].clone()
                        aug[:,-sp.shape[1]:]+=norm_matched_noise(delta_norm[batch],sp.shape[1],noise_rng)/spatial_sd
                    else:aug=ce[batch]
                    augmented_logits=model(aug,phys[batch],query[batch])
                    if method=='paired_consistency':
                        loss=criterion(original_logits,y[batch])+(original_logits.sigmoid()-augmented_logits.sigmoid()).square().mean()
                    else:loss=(criterion(original_logits,y[batch])+criterion(augmented_logits,y[batch]))/2
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
                    if (step+1)%250==0:
                        model.eval();score=metrics(target[vi],infer(vi,co))['balanced_error']
                        history.append({'step':step+1,'balanced_error':score})
                        if score<best_score:
                            best_score=score;best_step=step+1
                            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if best_state is None:raise ValueError('At least 250 updates required')
                model.load_state_dict(best_state);model.eval()
                vp=infer(vi,co);threshold,calibrated_score=optimal_threshold(target[vi],vp)
                po=infer(ti,co);pe=infer(ti,ce)
                for view in ['original_all','original_survivors','edited_survivors']:
                    probability=pe if view=='edited_survivors' else po
                    for mode in ['id','ood']:
                        select=modes[ti]==mode
                        if view!='original_all':select=select&eligible[oid[ti]]
                        row={'method':method,'fold':fold,'seed':seed,'view':view,'mode':mode,
                            'selected_step':best_step,'threshold':threshold,
                            'validation_balanced_error':best_score,'validation_calibrated_balanced_error':calibrated_score,
                            **threshold_metrics(target[ti][select],probability[select],threshold),
                            'fixed_threshold_balanced_error':metrics(target[ti][select],probability[select])['balanced_error']}
                        if view=='edited_survivors':row['mean_abs_probability_change']=float(np.abs(pe[select]-po[select]).mean())
                        rows.append(row);print(json.dumps(row),flush=True)
                name=f'fold{fold}_{method}_seed{seed}'
                np.savez_compressed(args.out/f'{name}_predictions.npz',indices=ti,original_probability=po,
                    edited_probability=pe,threshold=np.array(threshold),validation_indices=vi,validation_probability=vp)
                torch.save({'state_dict':best_state,'context_mean':torch.from_numpy(mean),
                    'context_std':torch.from_numpy(sd)},args.out/f'{name}.pt')
                (args.out/f'{name}_validation.json').write_text(json.dumps(history,indent=2))
                del model,optimizer
        (args.out/'results.json').write_text(json.dumps(rows,indent=2))
    summary={m:{v:{mode:{k:float(np.mean([r[k] for r in rows if r['method']==m and r['view']==v and r['mode']==mode]))
        for k in ['auc','balanced_error','false_negative_rate','false_positive_rate','fixed_threshold_balanced_error']}
        for mode in ['id','ood']} for v in ['original_all','original_survivors','edited_survivors']} for m in METHODS}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
