"""Evaluate frozen models/thresholds on reserved logs without recalibration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cavla.geometric_calibration import threshold_metrics
from cavla.geometric_learning import metrics
from cavla.nuscenes_geometry import rotation
from cavla.partial_composition_models import FactorModel, SetModel, or_logits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--metadata-study',type=Path,required=True)
    p.add_argument('--supervised-reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    config=json.loads((args.study/'manifest.json').read_text())['config']
    reference=Path(config['reference_study']);rm=json.loads((reference/'manifest.json').read_text())
    with np.load(args.data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((args.data/'objects.json').read_text());scenes=json.loads((args.data/'scenes.json').read_text())
    train_objects=json.loads((Path(config['data'])/'objects.json').read_text())
    assert not {r['log_token'] for r in objects}&{r['log_token'] for r in train_objects}
    feature_objects=json.loads((args.features.parent/'objects.json').read_text())
    assert objects==feature_objects
    with np.load(args.features) as z:features=z['features'].reshape(len(objects),-1)
    geom={(r['scene'],r['stem']):r for r in json.loads((args.geometry/'records.json').read_text())}
    physical=[];metadata=[];classes=['car','person','truck','bus','bicycle','motorcycle']
    for o in objects:
        r=geom[o['scene'],o['stem']];cam=r['input_camera'];b=r['detection_box']
        width,length,height=rm['class_priors'][r['label']]
        ray=rotation(cam['rotation']).apply(np.linalg.solve(np.array(cam['camera_intrinsic']),np.array([(b[0]+b[2])/2,(b[1]+b[3])/2,1])))
        depth=np.clip(cam['camera_intrinsic'][1][1]*height/max(b[3]-b[1],1),1,180)
        physical.append([*ray[:2],*cam['translation'][:2],depth,width,length])
        metadata.append([b[0]/1600,b[1]/900,b[2]/1600,b[3]/900,*[float(r['label']==c) for c in classes]])
    physical=np.array(physical,np.float32);base=np.c_[metadata,physical].astype(np.float32)
    ids=d['scene_object_ids'];safe=ids.clip(min=0);nq=len(d['queries']);ns=len(ids)
    present=torch.tensor(ids>=0,device='cuda');phys=torch.tensor(physical[safe],device='cuda')
    query=torch.tensor(d['queries'],device='cuda');keep_tensor=torch.tensor(d['case_keep'],device='cuda')
    all_groups=np.arange(ns*nq);indices=np.arange(len(d['case_scenes'])*nq)
    ci=indices//nq;qi=indices%nq;si=d['case_scenes'][ci];card=d['case_cardinality'][ci]
    ambiguous=(d['oracle_lower']!=d['oracle_upper']).reshape(-1);target=d['case_targets'].reshape(-1)
    sources={m:(args.study,m) for m in config['methods']};sources['geometry_joint_metadata']=(args.metadata_study,'geometry_joint')
    sources.update({m:(args.supervised_reference,m) for m in ['supervised_or','supervised_max']})
    manifest={'scope':'New-log evaluation with frozen checkpoints and original validation thresholds; no holdout fitting',
        'config':{k:str(v) for k,v in vars(args).items()},'sources':{m:[str(p),name] for m,(p,name) in sources.items()},
        'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_HOLDOUT_LOGS_COMPOSICAO.md').read_bytes()).hexdigest(),
        'dataset_sha256':hashlib.sha256((args.data/'dataset.npz').read_bytes()).hexdigest(),
        'models':{},'limitations':['Same nuScenes collection and fixed logged constraints','No compound edited images or policy outcomes',
            'New-scene subset not restricted to objects with edited pairs','No claim that backbone pretraining excluded these scenes']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    masks={'original':card==0,'single':card==1,'double':card==2,'triple':card==3,'compound':card>=2,
        'compound_ambiguous':(card>=2)&ambiguous,'compound_identified':(card>=2)&~ambiguous}
    rows=[]
    for alias,(study,method) in sources.items():
        for fold in range(3):
            for seed in [17,29,43]:
                supervised=alias.startswith('supervised_');metadata_only=alias.endswith('_metadata')
                if supervised:
                    with np.load(reference/f'fold{fold}_spatial_pca.npz') as z:sp=((features-z['mean'])/z['scale'])@z['basis']
                    weight_file=reference/f'fold{fold}_geometry_spatial_seed{seed}.pt'
                    saved=torch.load(weight_file,map_location='cpu',weights_only=True)
                    mean=saved['context_mean'].numpy();sd=saved['context_std'].numpy()
                else:
                    with np.load(study/f'fold{fold}_pca.npz') as z:
                        sp=np.empty((len(objects),0),np.float32) if metadata_only else ((features-z['mean'])/z['scale'])@z['basis']
                        mean=z['context_mean'];sd=z['context_std']
                    weight_file=study/f'fold{fold}_{method}_seed{seed}.pt'
                    saved=torch.load(weight_file,map_location='cpu',weights_only=True)
                fo=np.c_[base,sp].astype(np.float32);context=torch.tensor(((fo-mean)/sd)[safe],device='cuda')
                model=(SetModel(fo.shape[1]) if method=='deepsets' else FactorModel(fo.shape[1],supervised or method.startswith('geometry'))).cuda()
                if supervised:model.head.load_state_dict(saved['state_dict'])
                else:model.load_state_dict(saved['state_dict'])
                model.eval();threshold_file=study/f'fold{fold}_{method}_seed{seed}_predictions.npz'
                with np.load(threshold_file) as z:threshold=float(z['threshold'])
                probability=[]
                with torch.inference_mode():
                    if method!='deepsets':
                        cache=torch.zeros((ns,nq,ids.shape[1]),device='cuda')
                        for start in range(0,len(all_groups),256):
                            group=all_groups[start:start+256];sj=group//nq;qj=group%nq
                            cache[sj,qj]=model(context[sj],phys[sj],query[qj],present[sj])
                        if supervised:cache_np=cache.sigmoid().cpu().numpy().astype(float)
                    for start in range(0,len(indices),1024):
                        i=indices[start:start+1024];cj=ci[i];sj=si[i];qj=qi[i];keep=keep_tensor[cj]
                        if supervised:
                            p=cache_np[sj,qj];mask=d['case_keep'][cj]
                            probability.append(np.where(mask,p,0.).max(-1) if alias=='supervised_max' else 1-np.where(mask,1-p,1.).prod(-1))
                        else:
                            logit=model(context[sj],query[qj],keep) if method=='deepsets' else or_logits(cache[sj,qj],keep)
                            probability.append(logit.sigmoid().cpu().numpy())
                probability=np.concatenate(probability)
                for view,mask in masks.items():
                    for mode in ['id','ood']:
                        use=mask&(d['query_modes'][qi]==mode)
                        if not use.any():continue
                        rows.append({'method':alias,'fold':fold,'seed':seed,'view':view,'mode':mode,
                            **threshold_metrics(target[use],probability[use],threshold),
                            'fixed_threshold_balanced_error':metrics(target[use],probability[use])['balanced_error'],
                            'threshold':threshold})
                stem=f'fold{fold}_{alias}_seed{seed}'
                np.savez_compressed(args.out/f'{stem}_predictions.npz',indices=indices,probability=probability,threshold=np.array(threshold))
                manifest['models'][stem]={'checkpoint':str(weight_file),'checkpoint_sha256':hashlib.sha256(weight_file.read_bytes()).hexdigest(),
                    'threshold_source':str(threshold_file),'threshold_source_sha256':hashlib.sha256(threshold_file.read_bytes()).hexdigest()}
                print(json.dumps(next(r for r in reversed(rows) if r['method']==alias and r['fold']==fold and r['seed']==seed and r['view']=='compound' and r['mode']=='ood')),flush=True)
                (args.out/'results.json').write_text(json.dumps(rows,indent=2));(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
                del model
    def average(m,v,mode,key):
        values=[r[key] for r in rows if r['method']==m and r['view']==v and r['mode']==mode and r[key] is not None]
        return float(np.mean(values)) if values else None
    summary={m:{v:{mode:{k:average(m,v,mode,k) for k in ['auc','balanced_error','false_negative_rate','false_positive_rate',
        'fixed_threshold_balanced_error']} for mode in ['id','ood']} for v in masks} for m in sources}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
