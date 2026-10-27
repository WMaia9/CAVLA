"""Untrained RGB/gradient control for the cost of frozen VLM ROI features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    objects=json.loads(args.objects.read_text())
    geometry={(r['scene'],r['stem']):r for r in json.loads((args.geometry/'records.json').read_text())}
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')
    features=[];hashes={};images={}
    for obj in objects:
        scene=obj['scene'];path=root/scene/f'{scene}_front.jpg'
        if scene not in images:
            with Image.open(path) as im:images[scene]=im.convert('RGB').copy()
            hashes[scene]=hashlib.sha256(path.read_bytes()).hexdigest()
        image=images[scene];box=geometry[scene,obj['stem']]['detection_box']
        x0,y0=np.floor(box[:2]).astype(int);x1,y1=np.ceil(box[2:]).astype(int)
        crop=np.array(image.crop((max(0,x0),max(0,y0),min(image.width,x1),min(image.height,y1)))
            .resize((16,16),Image.Resampling.LANCZOS),np.float32)/255
        context=np.array(image.resize((16,16),Image.Resampling.LANCZOS),np.float32)/255
        gray=crop@np.array([.299,.587,.114],np.float32);dy,dx=np.gradient(gray)
        features.append(np.r_[crop.reshape(-1),context.reshape(-1),dy.reshape(-1),dx.reshape(-1)])
    features=np.array(features,np.float32)
    np.savez_compressed(args.out/'features.npz',features=features)
    (args.out/'objects.json').write_text(json.dumps(objects,indent=2))
    manifest={'scope':'Cheap untrained visual control, not a strong pretrained vision baseline',
        'features':'16x16 RGB crop + 16x16 RGB full-image context + 16x16 crop luminance x/y gradients',
        'shape':list(features.shape),'objects_sha256':hashlib.sha256(args.objects.read_bytes()).hexdigest(),
        'image_hashes':hashes,'reduction':'Use same rank-16 train-only PCA and geometry head as VLM study'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'shape':features.shape}))


if __name__=='__main__':main()
