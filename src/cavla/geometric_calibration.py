"""Reload selected models; calibrate threshold using only inner validation scenes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cavla.geometric_learning import metrics
from cavla.nuscenes_geometry import rotation
from cavla.structured_geometry_learning import GeometryHead


def optimal_threshold(y,p):
    order=np.argsort(p,kind='stable');sorted_p=np.asarray(p,dtype=float)[order];ys=y[order]
    values,starts=np.unique(sorted_p,return_index=True)
    starts=np.r_[starts,len(y)];values=np.r_[values,float(sorted_p[-1])+1e-6]
    positives=np.r_[0,np.cumsum(ys)];n_pos=y.sum();n_neg=len(y)-n_pos
    errors=(positives[starts]/n_pos+(n_neg-(starts-positives[starts]))/n_neg)/2
    tied=np.flatnonzero(np.isclose(errors,errors.min(),rtol=0,atol=1e-12))
    chosen=tied[np.argmin(np.abs(values[tied]-.5))]
    return float(values[chosen]),float(errors[chosen])


def threshold_metrics(y,p,threshold):
    output=metrics(y,p);y=y.astype(bool);pred=np.asarray(p,dtype=float)>=threshold
    fn=float((~pred[y]).mean());fp=float(pred[~y].mean())
    output.update(false_negative_rate=fn,false_positive_rate=fp,balanced_error=(fn+fp)/2)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    manifest=json.loads((args.study/'manifest.json').read_text());config=manifest['config']
    data=Path(config['data']);geometry_path=Path(config['geometry']);spatial_path=Path(config['spatial_features'])
    with np.load(data/'dataset.npz') as z:d={k:z[k] for k in z.files}
    objects=json.loads((data/'objects.json').read_text())
    geometries={(r['scene'],r['stem']):r for r in json.loads((geometry_path/'records.json').read_text())}
    with np.load(spatial_path) as z:spatial=z['features'].reshape(len(objects),-1)
    priors=manifest['class_priors'];physical=[]
    for obj in objects:
        r=geometries[obj['scene'],obj['stem']];camera=r['input_camera'];box=r['detection_box']
        uv=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,1])
        ray=rotation(camera['rotation']).apply(np.linalg.solve(np.array(camera['camera_intrinsic']),uv))
        width,length,height=priors[r['label']]
        depth=np.clip(camera['camera_intrinsic'][1][1]*height/max(box[3]-box[1],1),1,180)
        physical.append([*ray[:2],*camera['translation'][:2],depth,width,length])
    physical=np.array(physical,np.float32);base=np.c_[d['metadata'],physical].astype(np.float32)
    oid=d['object_ids'];query=torch.tensor(d['queries'],device='cuda');phys=torch.tensor(physical[oid],device='cuda')
    source_rows=json.loads((args.study/'results.json').read_text());rows=[]
    for fold in range(3):
        split=json.loads((args.study/f'fold{fold}_split.json').read_text())
        validation=np.isin(oid,split['validation_objects'])&(d['query_modes']=='id');vi=np.flatnonzero(validation)
        with np.load(args.study/f'fold{fold}_spatial_pca.npz') as z:sp=((spatial-z['mean'])/z['scale'])@z['basis']
        for method in config['methods']:
            features=np.c_[base,sp] if method.endswith('_spatial') else base
            for seed in config['seeds']:
                saved=torch.load(args.study/f'fold{fold}_{method}_seed{seed}.pt',map_location='cpu',weights_only=True)
                state=saved['state_dict'];mean=saved['context_mean'].numpy();sd=saved['context_std'].numpy()
                context=torch.tensor((features[oid]-mean)/sd,device='cuda')
                structured=method.startswith('geometry_')
                if structured:model=GeometryHead(features.shape[1])
                else:
                    width=state['0.weight'].shape[0]
                    model=nn.Sequential(nn.Linear(features.shape[1]+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1))
                model.load_state_dict(state);model.cuda().eval()
                with np.load(args.study/f'fold{fold}_{method}_seed{seed}_predictions.npz') as z:
                    ti=z['indices'];reference=z['probability']
                def infer(indices):
                    output=[]
                    with torch.no_grad():
                        for start in range(0,len(indices),2048):
                            i=indices[start:start+2048]
                            logits=model(context[i],phys[i],query[i]) if structured else model(torch.cat([context[i],query[i]],-1)).squeeze(-1)
                            output.append(logits.sigmoid().cpu().numpy())
                    return np.concatenate(output)
                replay=infer(ti);error=float(np.max(np.abs(replay-reference)))
                if error>1e-5:raise RuntimeError(f'Checkpoint replay mismatch {method}, fold {fold}, seed {seed}: {error}')
                vp=infer(vi);threshold,score=optimal_threshold(d['targets'][vi],vp)
                for mode in ['id','ood']:
                    select=d['query_modes'][ti]==mode
                    row={'method':method,'fold':fold,'seed':seed,'mode':mode,'threshold':threshold,
                        'validation_balanced_error':score,'checkpoint_replay_maxabs':error,
                        **threshold_metrics(d['targets'][ti][select],replay[select],threshold)}
                    rows.append(row);print(json.dumps(row),flush=True)
                np.savez_compressed(args.out/f'fold{fold}_{method}_seed{seed}_predictions.npz',indices=ti,
                    probability=replay,threshold=np.array(threshold),validation_indices=vi,validation_probability=vp)
                del model,context
    summary={name:{mode:{k:float(np.mean([r[k] for r in rows if r['method']==name and r['mode']==mode])) for k in
        ['auc','balanced_error','false_negative_rate','false_positive_rate']} for mode in ['id','ood']} for name in config['methods']}
    (args.out/'results.json').write_text(json.dumps(rows,indent=2));(args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.out/'manifest.json').write_text(json.dumps({'study':str(args.study),'selection':'Threshold minimizing inner-validation balanced error; no test-label threshold selection',
        'ties':'Closest to 0.5 among equal validation minima','scope':'Exploratory post-training calibration of selected checkpoints'},indent=2))


if __name__=='__main__':main()
