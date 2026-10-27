"""Extract original-image object ROI features from the actual VLM prefill.

Avoids original_lm_hs.npz: that archival file contains a single object's ROI per
scene, and cannot be reused as the original ROI for every removed object.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def roi_weights(box,height,width):
    x0,y0,x1,y1=np.array(box)/np.array([1600,900,1600,900])
    left=np.arange(width)/width;right=(np.arange(width)+1)/width
    top=np.arange(height)/height;bottom=(np.arange(height)+1)/height
    dx=np.maximum(0,np.minimum(right,x1)-np.maximum(left,x0))
    dy=np.maximum(0,np.minimum(bottom,y1)-np.maximum(top,y0))
    weights=(dy[:,None]*dx[None]).reshape(-1)
    if weights.sum()<=0:raise ValueError('Empty ROI')
    return weights/weights.sum()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--blank-front',action='store_true')
    p.add_argument('--zero-ego',action='store_true')
    p.add_argument('--reuse-identical-input',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.reuse_identical_input and not (args.blank_front and args.zero_ego):
        raise ValueError('Reuse requires identical all-black frames and zero/identity ego history')
    objects=json.loads(args.objects.read_text())
    geometry={(r['scene'],r['stem']):r for r in json.loads((args.geometry/'records.json').read_text())}
    source=Path('/mnt/datasets/Explainability/alpamayo/src');sys.path.insert(0,str(source))
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    torch.set_num_threads(2)
    checkpoint='/home/perceptron-lab/.cache/huggingface/hub/models--nvidia--Alpamayo-R1-10B/snapshots/d72c6a2757b10cacb69b688ca66ccfad3dafbf37'
    model=AlpamayoR1.from_pretrained(checkpoint,dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    processor=helper.get_processor(model.tokenizer)
    layers=[8,16,24,36];features=np.zeros((len(objects),len(layers),4096),np.float32)
    manifest={'layers':layers,'active_front_image_index':12,
        'input_protocol':'Archived 4x4 padded front still, same as previous readouts',
        'roi':'Fractional pixel overlap on merged visual-token grid, weighted mean per layer',
        'objects_sha256':hashlib.sha256(args.objects.read_bytes()).hexdigest(),
        'scope':'Spatial representation baseline, no fine-tuning and no claim of causal necessity',
        'blank_front':args.blank_front,'zero_ego_history':args.zero_ego,
        'reuse_identical_input_for_all_rois':args.reuse_identical_input,
        'image_hashes':{},'scenes':{}}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')
    scenes=sorted({r['scene'] for r in objects})
    for scene in (scenes[:1] if args.reuse_identical_input else scenes):
        folder=root/scene;path=folder/f'{scene}_front.jpg'
        frames=torch.zeros(4,4,3,1080,1920,dtype=torch.uint8)
        if not args.blank_front:
            with Image.open(path) as im:frames[3,0]=torch.from_numpy(np.array(im.convert('RGB').resize((1920,1080),Image.Resampling.LANCZOS))).permute(2,0,1)
        tokenized=processor.apply_chat_template(helper.create_message(frames.flatten(0,1)),tokenize=True,
            add_generation_prompt=False,continue_final_message=True,return_dict=True,return_tensors='pt')
        tokenized=helper.to_device(tokenized,'cuda');input_ids=tokenized.pop('input_ids')
        image_positions=(input_ids[0]==model.vlm.config.image_token_id).nonzero().flatten()
        grids=tokenized['image_grid_thw'].cpu().numpy();merge=model.vlm.config.vision_config.spatial_merge_size
        counts=(grids.prod(-1)//merge**2).astype(int)
        if counts.sum()!=len(image_positions):raise RuntimeError('Visual token/grid mismatch')
        if grids[12,0]!=1:raise RuntimeError('Unexpected temporal image grid')
        h,w=(grids[12,1:]//merge).astype(int)
        front_positions=image_positions[counts[:12].sum():counts[:13].sum()]
        with np.load(folder/f'{scene}_data.npz') as history:
            traj={'ego_history_xyz':torch.from_numpy(history['ego_history_xyz']).float()[None,None].cuda(),
                  'ego_history_rot':torch.from_numpy(history['ego_history_rot']).float()[None,None].cuda()}
        if args.zero_ego:
            traj['ego_history_xyz'].zero_()
            identity=torch.eye(3,device='cuda').reshape(-1)
            traj['ego_history_rot']=identity.repeat(traj['ego_history_rot'].numel()//9).reshape_as(traj['ego_history_rot'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            fused=model.fuse_traj_tokens(input_ids,traj)
            history_positions=(fused[0]!=input_ids[0]).nonzero().flatten()
            if len(history_positions)==0:raise RuntimeError('No fused history tokens found')
            history_after_visual=bool(history_positions.min()>image_positions.max())
            output=model.vlm(input_ids=fused,**tokenized,output_hidden_states=True,use_cache=False)
            selected=torch.stack([output.hidden_states[layer][0,front_positions].float() for layer in layers])
            for i,obj in enumerate(objects):
                if obj['scene']!=scene and not args.reuse_identical_input:continue
                weights=torch.tensor(roi_weights(geometry[obj['scene'],obj['stem']]['detection_box'],h,w),device='cuda',dtype=torch.float32)
                features[i]=(selected*weights[None,:,None]).sum(1).cpu().numpy()
        manifest['image_hashes'][scene]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['scenes'][scene]={'merged_grid':[int(h),int(w)],'image_tokens':len(image_positions),'front_tokens':len(front_positions),
            'last_visual_position':int(image_positions.max()),'first_fused_history_position':int(history_positions.min()),
            'history_after_all_visual_tokens':history_after_visual}
        del output,selected,tokenized,frames,input_ids,fused,traj
        print(json.dumps({'scene':scene,'objects':sum(r['scene']==scene for r in objects)}),flush=True)
    np.savez_compressed(args.out/'features.npz',features=features,layers=np.array(layers))
    (args.out/'objects.json').write_text(json.dumps(objects,indent=2))
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
