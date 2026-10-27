"""Reload composition checkpoints and reconstruct all validation/test outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cavla.nuscenes_geometry import rotation
from cavla.partial_composition_models import FactorModel, SetModel, or_logits


def replay(study):
    config=json.loads((study/'manifest.json').read_text())['config'];data=Path(config['data'])
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((data/'objects.json').read_text())
    reference=json.loads((Path(config['reference_study'])/'manifest.json').read_text())
    geom={(r['scene'],r['stem']):r for r in json.loads((Path(reference['config']['geometry'])/'records.json').read_text())}
    with np.load(config['features']) as z:features=z['features'].reshape(len(objects),-1)
    physical=[];metadata=[];classes=['car','person','truck','bus','bicycle','motorcycle']
    for o in objects:
        r=geom[o['scene'],o['stem']];cam=r['input_camera'];b=r['detection_box']
        uv=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2,1]);width,length,height=reference['class_priors'][r['label']]
        ray=rotation(cam['rotation']).apply(np.linalg.solve(np.array(cam['camera_intrinsic']),uv))
        depth=np.clip(cam['camera_intrinsic'][1][1]*height/max(b[3]-b[1],1),1,180)
        physical.append([*ray[:2],*cam['translation'][:2],depth,width,length])
        metadata.append([b[0]/1600,b[1]/900,b[2]/1600,b[3]/900,*[float(r['label']==c) for c in classes]])
    physical=np.array(physical,np.float32);base=np.c_[metadata,physical].astype(np.float32)
    ids=d['scene_object_ids'];safe=ids.clip(min=0);ns=len(ids);nq=len(d['queries'])
    present=torch.tensor(ids>=0,device='cuda');phys=torch.tensor(physical[safe],device='cuda')
    query=torch.tensor(d['queries'],device='cuda');removed=torch.tensor(d['removed_index'],device='cuda')
    case_keep=torch.tensor(d['case_keep'],device='cuda');result=[]
    for fold in range(3):
        with np.load(study/f'fold{fold}_pca.npz') as z:
            sp=np.empty((len(objects),0),np.float32) if config.get('metadata_only',False) else ((features-z['mean'])/z['scale'])@z['basis']
            fo=np.c_[base,sp].astype(np.float32);context=torch.tensor(((fo-z['context_mean'])/z['context_std'])[safe],device='cuda')
        for method in config['methods']:
            for seed in config['seeds']:
                model=(SetModel(fo.shape[1]) if method=='deepsets' else FactorModel(fo.shape[1],method.startswith('geometry'))).cuda()
                stem=f'fold{fold}_{method}_seed{seed}'
                saved=torch.load(study/f'{stem}.pt',map_location='cpu',weights_only=True)
                model.load_state_dict(saved['state_dict']);model.eval()
                with np.load(study/f'{stem}_predictions.npz') as z:
                    ti=z['indices'];reference_test=z['probability'];vi=z['validation_group_ids'];reference_val=z['validation_probability']
                vp=[];pp=[]
                with torch.no_grad():
                    for start in range(0,len(vi),512):
                        group=vi[start:start+512];si=group//nq;qi=group%nq;keep=present[si]
                        without=keep.clone();without.scatter_(1,removed[si,None],False)
                        if method=='deepsets':pair=torch.stack([model(context[si],query[qi],keep),model(context[si],query[qi],without)],-1)
                        else:
                            factors=model(context[si],phys[si],query[qi],keep)
                            pair=torch.stack([or_logits(factors,keep),or_logits(factors,without)],-1)
                        vp.append(pair.sigmoid().cpu().numpy())
                    if method!='deepsets':
                        cache=torch.zeros((ns,nq,ids.shape[1]),device='cuda')
                        group=np.flatnonzero(np.repeat(d['scene_folds']==fold,nq))
                        for start in range(0,len(group),256):
                            g=group[start:start+256];si=g//nq;qi=g%nq
                            cache[si,qi]=model(context[si],phys[si],query[qi],present[si])
                    for start in range(0,len(ti),1024):
                        i=ti[start:start+1024];ci=i//nq;si=d['case_scenes'][ci];qi=i%nq;keep=case_keep[ci]
                        logit=model(context[si],query[qi],keep) if method=='deepsets' else or_logits(cache[si,qi],keep)
                        pp.append(logit.sigmoid().cpu().numpy())
                val_error=float(np.max(np.abs(np.concatenate(vp)-reference_val)))
                test_error=float(np.max(np.abs(np.concatenate(pp)-reference_test)))
                if max(val_error,test_error)>1e-5:raise RuntimeError(f'{stem}: replay errors {val_error}, {test_error}')
                row={'study':str(study),'fold':fold,'method':method,'seed':seed,'validation_maxabs':val_error,'test_maxabs':test_error}
                result.append(row);print(json.dumps(row),flush=True);del model
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--studies',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    result=[]
    for study in args.studies:result.extend(replay(study))
    summary={'replayed_models':len(result),'max_validation_error':max(r['validation_maxabs'] for r in result),
        'max_test_error':max(r['test_maxabs'] for r in result),'scope':'Saved-checkpoint consistency, not independent replication'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));(args.out/'records.json').write_text(json.dumps(result,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
