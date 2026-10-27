"""Frozen Qwen3-VL backbone readout from local general/VLA checkpoints.

Only visual-prefix hidden states are extracted. This does not run the planner.
Loads every backbone tensor strictly; never initializes a missing trained weight.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from cavla.object_features import roi_weights


HUB=Path('/home/perceptron-lab/.cache/huggingface/hub')


def snapshot(name):
    paths=sorted((HUB/('models--'+name.replace('/','--'))/'snapshots').glob('*'))
    if len(paths)!=1:raise ValueError(f'Expected one pinned snapshot for {name}: {paths}')
    return paths[0]


def load_backbone(name):
    path=snapshot(name); raw=json.loads((path/'config.json').read_text())
    prefix='vlm.' if name.startswith('nvidia/Alpamayo') else ''
    if prefix:
        base=raw.get('vlm_config') or json.loads((snapshot('nvidia/Cosmos-Reason2-8B')/'config.json').read_text())
        config=Qwen3VLConfig.from_dict(base)
        config.text_config.vocab_size=raw.get('vocab_size',config.text_config.vocab_size)
    else:config=Qwen3VLConfig.from_dict(raw)
    # Native wrapper resolves to SDPA in this installed Transformers version.
    config._attn_implementation='sdpa'
    with torch.device('meta'):
        model=Qwen3VLForConditionalGeneration(config).to(dtype=torch.bfloat16)
    expected=set(model.state_dict()); loaded=set(); shards=[]
    index=path/'model.safetensors.index.json'
    if index.exists():
        weight_map=json.loads(index.read_text())['weight_map']
        shards=sorted({f for k,f in weight_map.items() if not prefix or k.startswith(prefix)})
    else:shards=['model.safetensors']
    for shard in shards:
        values={}
        with safe_open(path/shard,framework='pt',device='cpu') as f:
            for key in f.keys():
                if prefix and not key.startswith(prefix):continue
                target=key[len(prefix):]
                if target not in expected:raise ValueError(f'Unexpected backbone tensor {key}')
                if target in loaded:raise ValueError(f'Duplicate tensor {target}')
                values[target]=f.get_tensor(key).to(device='cuda',dtype=torch.bfloat16);loaded.add(target)
        model.load_state_dict(values,strict=False,assign=True)
        del values
    if loaded!=expected:raise ValueError(f'Missing trained backbone tensors: {sorted(expected-loaded)}')
    # Rotary frequencies are deterministic nonpersistent buffers, absent from
    # checkpoints. Materialize them after loading parameters from the meta model.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding
    for module in model.modules():
        if isinstance(module,Qwen3VLVisionRotaryEmbedding):
            module.inv_freq=Qwen3VLVisionRotaryEmbedding(module.inv_freq.numel()*2).inv_freq.cuda()
        elif isinstance(module,Qwen3VLTextRotaryEmbedding):
            fresh=Qwen3VLTextRotaryEmbedding(module.config,device='cuda')
            module.inv_freq=fresh.inv_freq;module.original_inv_freq=fresh.original_inv_freq
            module.attention_scaling=fresh.attention_scaling
    if any(p.is_meta for p in model.parameters()):raise ValueError('Uninitialized meta tensors')
    if any(b.is_meta for b in model.buffers()):raise ValueError('Uninitialized meta buffers')
    model.eval().requires_grad_(False)
    return model,path,{'backbone_tensors':len(loaded),'strict_complete':True,'shards':shards,'checkpoint_prefix':prefix}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-scenes',type=int)
    parser.add_argument('--native-tail-length',action='store_true',help='Loader check only: native placeholder lengths with zero suffix IDs')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    objects_path=Path('artifacts/semantic_ablation_inputs_v1_20261021/objects.json')
    objects=json.loads(objects_path.read_text());scenes=sorted({o['scene'] for o in objects})
    if args.max_scenes:
        scenes=scenes[:args.max_scenes];objects=[o for o in objects if o['scene'] in scenes]
    geometry={(r['scene'],r['stem']):r for r in json.loads(Path('artifacts/semantic_ablation_inputs_v1_20261021/records.json').read_text())}
    started=time.monotonic();model,path,loading=load_backbone(args.model)
    processor=AutoProcessor.from_pretrained(snapshot('Qwen/Qwen3-VL-2B-Instruct'),local_files_only=True,
        min_pixels=163840,max_pixels=196608)
    processor.tokenizer=AutoTokenizer.from_pretrained(snapshot('Qwen/Qwen3-VL-8B-Instruct'),local_files_only=True)
    if args.native_tail_length:
        processor.tokenizer.add_tokens(['<|traj_history_start|>','<|traj_history|>','<|traj_history_end|>','<|cot_start|>'],special_tokens=True)
    sys.path.insert(0,'/mnt/datasets/Explainability/alpamayo/src')
    from alpamayo_r1.helper import create_message
    depth=model.config.text_config.num_hidden_layers
    layers=[8,16,24,36] if depth==36 else [round(depth*f) for f in [2/9,4/9,2/3,1]]
    features=np.zeros((len(objects),4,model.config.text_config.hidden_size),np.float32)
    manifest={'model':args.model,'snapshot':str(path),'loading':loading,'layers':layers,
        'scope':'Backbone visual-prefix representation only; no planner or driving-performance evaluation',
        'protocol':'Same16 image slots as earlier readout, original frontal still at12; same processor pixel bounds',
        'objects_sha256':hashlib.sha256(objects_path.read_bytes()).hexdigest(),'scenes':{},
        'parameters':sum(p.numel() for p in model.parameters()),'config_sha256':hashlib.sha256((path/'config.json').read_bytes()).hexdigest()}
    manifest['native_tail_length_check']=args.native_tail_length
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')
    for scene in scenes:
        frames=torch.zeros(16,3,1080,1920,dtype=torch.uint8)
        image_path=root/scene/f'{scene}_front.jpg'
        with Image.open(image_path) as im:frames[12]=torch.from_numpy(np.array(im.convert('RGB').resize((1920,1080),Image.Resampling.LANCZOS))).permute(2,0,1)
        tokens=processor.apply_chat_template(create_message(frames),tokenize=True,add_generation_prompt=False,
            continue_final_message=True,return_dict=True,return_tensors='pt')
        tokens={k:v.cuda() if torch.is_tensor(v) else v for k,v in tokens.items()}
        ids=tokens['input_ids'];pos=(ids[0]==model.config.image_token_id).nonzero().flatten()
        if args.native_tail_length:ids[0,int(pos.max())+1:]=0
        grids=tokens['image_grid_thw'].cpu().numpy();merge=model.config.vision_config.spatial_merge_size
        counts=(grids.prod(-1)//merge**2).astype(int)
        if len(grids)!=16 or counts.sum()!=len(pos):raise ValueError('Image/grid alignment failed')
        h,w=(grids[12,1:]//merge).astype(int);front=pos[counts[:12].sum():counts[:13].sum()]
        if (h,w)!=(10,18):raise ValueError(f'Unexpected matched grid {(h,w)}')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            # Use the same outer conditional wrapper as the original study.
            # In this Transformers version, output capture on model.model replaces
            # the final captured state with RMS-normalized last_hidden_state.
            # One token of logits avoids allocating a full sequence vocabulary.
            output=model(**tokens,output_hidden_states=True,use_cache=False,logits_to_keep=1)
            selected=torch.stack([output.hidden_states[layer][0,front].float() for layer in layers])
            for i,o in enumerate(objects):
                if o['scene']!=scene:continue
                weights=torch.tensor(roi_weights(geometry[scene,o['stem']]['detection_box'],h,w),device='cuda',dtype=torch.float32)
                features[i]=(selected*weights[None,:,None]).sum(1).cpu().numpy()
        manifest['scenes'][scene]={'grid':[int(h),int(w)],'image_tokens':len(pos),
            'visual_prefix_sha256':hashlib.sha256(ids[0,:int(pos.max())+1].cpu().numpy().tobytes()).hexdigest(),
            'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest()}
        del output,selected,tokens,frames,ids
        print(json.dumps({'model':args.model,'scene':scene,'done':len(manifest['scenes']),'total':len(scenes)}),flush=True)
    np.savez_compressed(args.out/'features.npz',features=features,layers=np.array(layers))
    (args.out/'objects.json').write_text(json.dumps(objects,indent=2))
    manifest['elapsed_seconds']=time.monotonic()-started;manifest['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
