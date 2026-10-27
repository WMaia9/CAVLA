"""Extract surviving-object features from existing clean editor composites."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from cavla.object_features import roi_weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--original-features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    objects=json.loads((args.data/'objects.json').read_text());pairs=json.loads(args.pairs.read_text())
    original_objects=json.loads((args.original_features.parent/'objects.json').read_text())
    assert [(r['scene'],r['stem']) for r in objects]==[(r['scene'],r['stem']) for r in original_objects]
    with np.load(args.original_features) as z:original=z['features'];layers=z['layers'].tolist()
    features=original.copy();eligible=np.zeros(len(objects),bool)
    geometry={(r['scene'],r['stem']):r for r in json.loads((args.geometry/'records.json').read_text())}
    source=Path('/mnt/datasets/Explainability/alpamayo/src');sys.path.insert(0,str(source))
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    torch.set_num_threads(2)
    checkpoint='/home/perceptron-lab/.cache/huggingface/hub/models--nvidia--Alpamayo-R1-10B/snapshots/d72c6a2757b10cacb69b688ca66ccfad3dafbf37'
    manifest={'scope':'Frozen features for surviving-object context robustness; no verified physical intervention',
        'layers':layers,'pairs_sha256':hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        'original_features_sha256':hashlib.sha256(args.original_features.read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(Path('docs/PROTOCOLO_PARES_SOBREVIVENTES_V1.md').read_bytes()).hexdigest(),
        'input_protocol':'Same archived 4x4 padded front still as original features',
        'checkpoint':checkpoint,'scenes':[]}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    model=AlpamayoR1.from_pretrained(checkpoint,dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    processor=helper.get_processor(model.tokenizer)
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')
    def extract(path,scene,ids):
        frames=torch.zeros(4,4,3,1080,1920,dtype=torch.uint8)
        with Image.open(path) as im:
            frames[3,0]=torch.from_numpy(np.array(im.convert('RGB').resize((1920,1080),Image.Resampling.LANCZOS))).permute(2,0,1)
        tokenized=processor.apply_chat_template(helper.create_message(frames.flatten(0,1)),tokenize=True,
            add_generation_prompt=False,continue_final_message=True,return_dict=True,return_tensors='pt')
        tokenized=helper.to_device(tokenized,'cuda');input_ids=tokenized.pop('input_ids')
        positions=(input_ids[0]==model.vlm.config.image_token_id).nonzero().flatten()
        grids=tokenized['image_grid_thw'].cpu().numpy();merge=model.vlm.config.vision_config.spatial_merge_size
        counts=(grids.prod(-1)//merge**2).astype(int)
        assert counts.sum()==len(positions) and grids[12,0]==1
        h,w=(grids[12,1:]//merge).astype(int);front=positions[counts[:12].sum():counts[:13].sum()]
        with np.load(root/scene/f'{scene}_data.npz') as history:
            traj={'ego_history_xyz':torch.from_numpy(history['ego_history_xyz']).float()[None,None].cuda(),
                  'ego_history_rot':torch.from_numpy(history['ego_history_rot']).float()[None,None].cuda()}
        values=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            fused=model.fuse_traj_tokens(input_ids,traj)
            output=model.vlm(input_ids=fused,**tokenized,output_hidden_states=True,use_cache=False)
            selected=torch.stack([output.hidden_states[layer][0,front].float() for layer in layers])
            for i in ids:
                box=geometry[scene,objects[i]['stem']]['detection_box']
                weight=torch.tensor(roi_weights(box,h,w),device='cuda',dtype=torch.float32)
                values.append((selected*weight[None,:,None]).sum(1).cpu().numpy())
        return np.stack(values)
    for index,pair in enumerate(pairs):
        scene=pair['scene'];ids=[s['object_id'] for s in pair['survivors']]
        assert all(objects[i]['scene']==scene for i in ids)
        if index==0:
            replay=extract(root/scene/f'{scene}_front.jpg',scene,ids)
            error=float(np.max(np.abs(replay-original[ids])))
            manifest['original_replay_maxabs']=error
            if error>1e-4:raise RuntimeError(f'Original ROI replay mismatch: {error}')
        path=root/scene/'inpainted_clean_fin'/pair['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pair['local_sha256']
        values=extract(path,scene,ids);features[ids]=values;eligible[ids]=True
        np.savez_compressed(args.out/f'{scene}.npz',object_ids=np.array(ids),features=values)
        row={'scene':scene,'survivors':len(ids),'local_sha256':pair['local_sha256']}
        manifest['scenes'].append(row);print(json.dumps(row),flush=True)
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    np.savez_compressed(args.out/'features.npz',features=features,eligible=eligible,layers=np.array(layers))
    (args.out/'objects.json').write_text(json.dumps(objects,indent=2))


if __name__=='__main__':main()
