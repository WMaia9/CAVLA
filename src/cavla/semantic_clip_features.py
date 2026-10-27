"""Frozen OpenCLIP image control using cached weights, without environment edits."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def convert_visual(weights):
    mapped = {}
    direct = {'class_embedding':'embeddings.class_embedding','conv1.weight':'embeddings.patch_embedding.weight',
              'positional_embedding':'embeddings.position_embedding.weight','ln_pre.weight':'pre_layrnorm.weight',
              'ln_pre.bias':'pre_layrnorm.bias','ln_post.weight':'post_layernorm.weight','ln_post.bias':'post_layernorm.bias'}
    for old, new in direct.items():mapped['vision_model.'+new] = weights['visual.'+old]
    mapped['visual_projection.weight'] = weights['visual.proj'].T
    for i in range(12):
        old = f'visual.transformer.resblocks.{i}.'; new = f'vision_model.encoder.layers.{i}.'
        for suffix in ['weight','bias']:
            chunks = weights[old+'attn.in_proj_'+suffix].chunk(3,dim=0)
            for name, value in zip(['q','k','v'],chunks):mapped[new+f'self_attn.{name}_proj.{suffix}'] = value
            for before, after in [('attn.out_proj','self_attn.out_proj'),('ln_1','layer_norm1'),('ln_2','layer_norm2'),('mlp.c_fc','mlp.fc1'),('mlp.c_proj','mlp.fc2')]:
                mapped[new+after+'.'+suffix] = weights[old+before+'.'+suffix]
    return mapped


def reference_forward(x, weights):
    """Independent direct evaluation of OpenCLIP state names to verify conversion."""
    w = weights
    def norm(x, prefix):return F.layer_norm(x, (768,), w[prefix+'.weight'], w[prefix+'.bias'], 1e-5)
    x = F.conv2d(x,w['visual.conv1.weight'],stride=32).flatten(2).transpose(1,2)
    x = torch.cat([w['visual.class_embedding'][None,None].expand(len(x),1,-1),x],1)
    x = norm(x+w['visual.positional_embedding'],'visual.ln_pre')
    for i in range(12):
        root = f'visual.transformer.resblocks.{i}.'
        z = norm(x,root+'ln_1')
        q,k,v = F.linear(z,w[root+'attn.in_proj_weight'],w[root+'attn.in_proj_bias']).chunk(3,-1)
        q,k,v = [a.reshape(len(x),50,12,64).transpose(1,2) for a in [q,k,v]]
        attention = F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(len(x),50,768)
        x = x+F.linear(attention,w[root+'attn.out_proj.weight'],w[root+'attn.out_proj.bias'])
        z = norm(x,root+'ln_2')
        z = F.gelu(F.linear(z,w[root+'mlp.c_fc.weight'],w[root+'mlp.c_fc.bias']))
        x = x+F.linear(z,w[root+'mlp.c_proj.weight'],w[root+'mlp.c_proj.bias'])
    return norm(x[:,0],'visual.ln_post')@w['visual.proj']


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args = p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    checkpoint = next(Path('/home/perceptron-lab/.cache/huggingface/hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots').glob('*/open_clip_model.safetensors'))
    config = json.loads(urllib.request.urlopen('https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/raw/main/open_clip_config.json',timeout=20).read())
    assert config['model_cfg']['vision_cfg'] == {'image_size':224,'layers':12,'width':768,'patch_size':32}
    weights = load_file(checkpoint,device='cpu')
    model = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=768,intermediate_size=3072,
        num_hidden_layers=12,num_attention_heads=12,image_size=224,patch_size=32,projection_dim=512,hidden_act='gelu')).eval()
    model.load_state_dict(convert_visual(weights),strict=True)
    torch.manual_seed(20261021); test=torch.randn(2,3,224,224)
    with torch.inference_mode():
        reference=reference_forward(test,weights); actual=model(test).image_embeds
    error=float((reference-actual).abs().max())
    if error>1e-4:raise RuntimeError(f'CLIP mapping verification failed: {error}')
    del weights;model.cuda()
    preprocess=transforms.Compose([transforms.Resize(224,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),
        transforms.ToTensor(),transforms.Normalize(config['preprocess_cfg']['mean'],config['preprocess_cfg']['std'])])
    root=Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene'); hashes={}
    for split, data, geometry in [('development','partial_composition_data_v2_20261021','nuscenes_geometry_v3_20261021'),
                                  ('previously_used_holdout','composition_holdout_data_v1_20261021','composition_holdout_geometry_v1_20261021')]:
        objects=json.loads((Path('artifacts')/data/'objects.json').read_text())
        geom={(r['scene'],r['stem']):r for r in json.loads((Path('artifacts')/geometry/'records.json').read_text())}
        with np.load(Path('artifacts/semantic_attribute_data_v1_20261021')/f'{split}.npz') as z:ids=z['object_indices'][z['slow_vehicle']]
        images={}; crops=[]
        for i in ids:
            obj=objects[i];scene=obj['scene'];path=root/scene/f'{scene}_front.jpg'
            if scene not in images:
                with Image.open(path) as im:images[scene]=im.convert('RGB').copy()
                hashes[scene]=hashlib.sha256(path.read_bytes()).hexdigest()
            b=geom[scene,obj['stem']]['detection_box'];box=[int(np.floor(b[0])),int(np.floor(b[1])),int(np.ceil(b[2])),int(np.ceil(b[3]))]
            crops.append(preprocess(images[scene].crop(box)))
        scene_ids=sorted(images); inputs=crops+[preprocess(images[s]) for s in scene_ids]; encoded=[]
        for start in range(0,len(inputs),32):
            with torch.inference_mode():
                output=model(torch.stack(inputs[start:start+32]).cuda()).image_embeds
                encoded.append(F.normalize(output,dim=-1).cpu().numpy())
        embeddings=np.concatenate(encoded); contexts=embeddings[len(ids):]
        context=np.stack([contexts[scene_ids.index(objects[i]['scene'])] for i in ids])
        np.savez_compressed(args.out/f'{split}.npz',object_indices=ids,crop=embeddings[:len(ids)],context=context,
                            features=np.c_[embeddings[:len(ids)],context])
        print(json.dumps({'split':split,'agents':len(ids),'scenes':len(scene_ids)}),flush=True)
    (args.out/'manifest.json').write_text(json.dumps({'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'model_config':config,'mapping_reference_maxabs':error,'activation':'GELU (OpenCLIP ViT-B-32 default)',
        'input':'Unexpanded detection-box crop and full-image context; standard resize/center crop; normalized projected embeddings',
        'scope':'Frozen ordinary visual-language encoder control, no driving fine-tuning, no shared environment changes','image_hashes':hashes},indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
