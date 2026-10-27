"""Controlled 2x2 pixel interventions separating local and exterior changes.

The local support is defined by actual clean-composite changed pixels, not by
unverified mask ordinals. It is not a semantic-object or physical-validity label.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

from cavla.cache_transport import distance


def factorial_inputs(original, clean, raw):
    if original.shape != clean.shape or original.shape != raw.shape:
        raise ValueError('Factorial images must have identical dimensions')
    support = (clean != original).any(-1)
    exterior = np.where(support[...,None], original, raw)
    joint = np.where(support[...,None], clean, raw)
    # Two disjoint byte-level substitutions commute exactly.
    residual = joint.astype(np.int16)-clean.astype(np.int16)-exterior.astype(np.int16)+original.astype(np.int16)
    if np.any(residual):
        raise RuntimeError('Pixel factorial identity failed')
    return {'original':original,'local':clean,'exterior':exterior,'joint':joint}, support


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--language-seed',type=int,default=123)
    p.add_argument('--decoder-seeds',type=int,nargs='+',default=[9101,9102,9103,9104])
    p.add_argument('--frame-context',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    pairs=json.loads(args.pairs.read_text())
    source=Path('/mnt/datasets/Explainability/alpamayo/src')
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest={'pairs':pairs,'language_seed':args.language_seed,'decoder_seeds':args.decoder_seeds,
        'scope':'Response to four operational pixel conditions; no identified physical object effect',
        'input_protocol':'Archived front still padded in 4x4 slots',
        'support':'Pixels differing between local composite and original, any RGB channel',
        'identity':'joint - local - exterior + original = 0 exactly in integer pixel space',
        'model_source_sha256':hashlib.sha256((source/'alpamayo_r1/models/alpamayo_r1.py').read_bytes()).hexdigest(),
        'limitations':['One language draw per image; four decoder draws are not four independent scenes',
                       'Local support is not a verified object mask',
                       'Exterior alterations may change useful semantics; not assumed harmless nuisances',
                       'Joint differs from raw editor output at soft composite boundaries',
                       'No normative counterfactual actions or safety validation']}
    if args.frame_context is not None:
        context_manifest=args.frame_context/'manifest.json'
        context_info=json.loads(context_manifest.read_text())
        manifest['input_protocol']=context_info['protocol']
        manifest['frame_context_limitations']=context_info['limitations']
        manifest['latest_front_slots']=context_info['latest_front_slots']
        manifest['frame_context_manifest_sha256']=hashlib.sha256(context_manifest.read_bytes()).hexdigest()
        manifest['frame_context_path']=str(args.frame_context)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    sys.path.insert(0,str(source))
    import alpamayo_r1.models.alpamayo_r1 as module
    from alpamayo_r1 import helper
    for name in ['register_expert_hooks','register_vision_hooks','register_lm_hooks']:
        setattr(module,name,lambda *a,**kw:({},[]))
    torch.set_num_threads(2)
    checkpoint='/home/perceptron-lab/.cache/huggingface/hub/models--nvidia--Alpamayo-R1-10B/snapshots/d72c6a2757b10cacb69b688ca66ccfad3dafbf37'
    model=module.AlpamayoR1.from_pretrained(checkpoint,dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    processor=helper.get_processor(model.tokenizer);sampler=model.diffusion.sample
    rows=[]
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')
    for index,pair in enumerate(pairs):
        scene=pair['scene'];base=root/scene
        paths={'original':base/f'{scene}_front.jpg','local':base/'inpainted_clean_fin'/pair['image'],
               'raw':base/'inpainted'/pair['image']}
        arrays={k:np.array(Image.open(path).convert('RGB')) for k,path in paths.items()}
        images,support=factorial_inputs(arrays['original'],arrays['local'],arrays['raw'])
        with np.load(base/f'{scene}_data.npz') as history:
            xyz_hist=torch.from_numpy(history['ego_history_xyz']).float()[None].cuda()
            rot_hist=torch.from_numpy(history['ego_history_rot']).float()[None].cuda()
        base_frames=None
        if args.frame_context is not None:
            with np.load(args.frame_context/f'{scene}_frames.npz') as loaded:base_frames=loaded['frames']
        predictions={}
        names=list(images)+(['original_repeat'] if index==0 else [])
        for name in names:
            array=images['original' if name=='original_repeat' else name]
            frames=torch.zeros(4,4,3,1080,1920,dtype=torch.uint8) if base_frames is None else torch.from_numpy(base_frames.copy())
            resized=Image.fromarray(array).resize((1920,1080),Image.Resampling.LANCZOS)
            edited_tensor=torch.from_numpy(np.array(resized)).permute(2,0,1)
            if base_frames is None:frames[3,0]=edited_tensor
            else:
                for camera,time_index in manifest['latest_front_slots']:frames[camera,time_index]=edited_tensor
            tokenized=processor.apply_chat_template(helper.create_message(frames.flatten(0,1)),
                tokenize=True,add_generation_prompt=False,continue_final_message=True,
                return_dict=True,return_tensors='pt')
            prepared=helper.to_device({'tokenized_data':tokenized,'ego_history_xyz':xyz_hist[:,None],
                                      'ego_history_rot':rot_hist[:,None]},'cuda')
            def paired_sampler(*a,**kw):
                first=None
                for seed in args.decoder_seeds:
                    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                    action=sampler(*a,**kw)
                    xyz,_=model.action_space.action_to_traj(action,xyz_hist,rot_hist)
                    predictions[name,seed]=xyz.float().cpu().numpy().reshape(-1,3)
                    if first is None:first=action
                return first
            model.diffusion.sample=paired_sampler
            random.seed(args.language_seed);np.random.seed(args.language_seed)
            torch.manual_seed(args.language_seed);torch.cuda.manual_seed_all(args.language_seed)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                output=model.sample_trajectories_from_data_with_vlm_rollout(data=prepared,
                    top_p=.98,temperature=.6,num_traj_samples=1,max_generation_length=256)
            model.diffusion.sample=sampler
            del prepared,output,frames,tokenized
            print(json.dumps({'scene':scene,'condition':name}),flush=True)
        checks={}
        if index==0:
            checks={str(s):distance(predictions['original',s],predictions['original_repeat',s]) for s in args.decoder_seeds}
            if max(checks.values())>1e-4:raise RuntimeError(f'No-op repeat failed: {checks}')
        for seed in args.decoder_seeds:
            o,l,e,j=[predictions[k,seed] for k in ['original','local','exterior','joint']]
            row={'scene':scene,'image':pair['image'],'seed':seed,
                 'local_response_m':distance(l,o),'exterior_response_m':distance(e,o),
                 'joint_response_m':distance(j,o),'local_effect_in_exterior_m':distance(j,e),
                 'interaction_m':distance(j-l-e+o,np.zeros_like(o))}
            rows.append(row)
        np.savez_compressed(args.out/f'{scene}_predictions.npz',**{f'{k}_{s}':v for (k,s),v in predictions.items()})
        diagnostics={'support_fraction':float(support.mean()),'no_op_checks':checks,
            'joint_raw_mean_absolute_pixel_difference':float(np.abs(images['joint'].astype(float)-arrays['raw'].astype(float)).mean()),
            'image_hashes':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}}
        (args.out/f'{scene}_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        (args.out/'records.json').write_text(json.dumps(rows,indent=2))
    summary={k:float(np.mean([r[k] for r in rows])) for k in
        ['local_response_m','exterior_response_m','joint_response_m','local_effect_in_exterior_m','interaction_m']}
    summary['n_scenes']=len(pairs)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
