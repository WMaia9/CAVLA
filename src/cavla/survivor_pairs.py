"""Select editor pairs whose surviving-object ROI pixels are preserved.

This is an operational appearance-robustness study. Pixel preservation neither
verifies removal identity nor validates physical counterfactual agent futures.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def overlap_area(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def changed_fraction(support,box):
    h,w=support.shape
    x0,y0=np.floor(box[:2]).astype(int);x1,y1=np.ceil(box[2:]).astype(int)
    patch=support[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]
    if not patch.size:raise ValueError('Empty image ROI')
    return float(patch.mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    objects=json.loads((args.data/'objects.json').read_text())
    geometry={(r['scene'],r['stem']):r for r in json.loads((args.geometry/'records.json').read_text())}
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')
    pairs=[];excluded=[]
    for scene in sorted({r['scene'] for r in objects}):
        ids=[i for i,r in enumerate(objects) if r['scene']==scene]
        ordered=sorted(ids,key=lambda i:hashlib.sha256((scene+'/'+objects[i]['stem']).encode()).hexdigest())
        with Image.open(root/scene/f'{scene}_front.jpg') as im:original=np.array(im.convert('RGB'))
        chosen=None
        for removed in ordered:
            stem=objects[removed]['stem'];path=root/scene/'inpainted_clean_fin'/f'{stem}_imagen.png'
            if not path.exists():continue
            with Image.open(path) as im:local=np.array(im.convert('RGB'))
            if local.shape!=original.shape:raise ValueError('Image dimension mismatch')
            support=np.any(local!=original,axis=-1);removed_box=geometry[scene,stem]['detection_box']
            survivors=[]
            for i in ids:
                if i==removed:continue
                box=geometry[scene,objects[i]['stem']]['detection_box'];fraction=changed_fraction(support,box)
                if overlap_area(removed_box,box)==0 and fraction<=.001:
                    survivors.append({'object_id':i,'roi_changed_fraction':fraction})
            if survivors:
                chosen={'scene':scene,'removed_object_id':removed,'removed_stem':stem,
                    'image':f'{stem}_imagen.png','survivors':survivors,
                    'support_fraction':float(support.mean()),'pixel_sha256':hashlib.sha256(local.tobytes()).hexdigest(),
                    'original_sha256':hashlib.sha256((root/scene/f'{scene}_front.jpg').read_bytes()).hexdigest(),
                    'local_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                break
        if chosen:pairs.append(chosen)
        else:excluded.append(scene)
    manifest={'scope':'Exploratory surviving-object appearance invariance, not verified physical removal',
        'selection':'First SHA256(scene/stem)-ordered full-future candidate with >=1 other nonoverlapping ROI and <=0.001 changed ROI pixels; no query/model outcomes used',
        'objects_sha256':hashlib.sha256((args.data/'objects.json').read_bytes()).hexdigest(),
        'scenes':len(pairs),'surviving_objects':sum(len(p['survivors']) for p in pairs),
        'excluded_scenes':excluded,'limitations':['Future labels inherited from original logged scene',
            'Preserved ROI pixels do not establish preserved semantics or reactive futures',
            'Uses previously explored scenes/logs, not independent confirmation']}
    (args.out/'pairs.json').write_text(json.dumps(pairs,indent=2))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
