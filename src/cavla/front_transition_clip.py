"""Frozen CLIP crop/context controls on prospective frontal transition cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import CLIPVisionConfig,CLIPVisionModelWithProjection
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from cavla.semantic_clip_features import convert_visual,reference_forward


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=Path('artifacts/front_transition_data_v1_20261021');rows=json.loads((source/'objects.json').read_text())
    old=json.loads(Path('artifacts/semantic_clip_features_v1_20261021/manifest.json').read_text())
    checkpoint=Path(old['checkpoint']);weights=load_file(checkpoint,device='cpu')
    model=CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=768,intermediate_size=3072,num_hidden_layers=12,num_attention_heads=12,image_size=224,patch_size=32,projection_dim=512,hidden_act='gelu')).eval()
    model.load_state_dict(convert_visual(weights),strict=True);torch.manual_seed(20261021);sample=torch.randn(2,3,224,224)
    with torch.inference_mode():error=float((reference_forward(sample,weights)-model(sample).image_embeds).abs().max())
    assert error<1e-4;del weights;model.cuda()
    config=old['model_config']['preprocess_cfg'];pre=transforms.Compose([transforms.Resize(224,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(config['mean'],config['std'])])
    crop=np.zeros((len(rows),512),np.float32);context=np.zeros_like(crop);images={};pending=[];keys=[];hashes={}
    def flush():
        if not pending:return
        with torch.inference_mode():out=F.normalize(model(torch.stack(pending).cuda()).image_embeds,dim=-1).cpu().numpy()
        for key,value in zip(keys,out):
            kind,index=key
            if kind=='crop':crop[index]=value
            else:images[index]=value
        pending.clear();keys.clear()
    order=sorted(range(len(rows)),key=lambda i:rows[i]['image'])
    path=None;im=None
    for done,i in enumerate(order):
        r=rows[i]
        if r['image']!=path:
            path=r['image'];im=Image.open(path).convert('RGB');hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            pending.append(pre(im));keys.append(('context',path))
        box=r['projected_box'];b=[int(np.floor(box[0])),int(np.floor(box[1])),int(np.ceil(box[2])),int(np.ceil(box[3]))]
        pending.append(pre(im.crop(b)));keys.append(('crop',i))
        if len(pending)>=32:flush()
        if (done+1)%500==0:print(done+1,'/',len(rows),flush=True)
    flush()
    for i,r in enumerate(rows):context[i]=images[r['image']]
    np.savez_compressed(args.out/'features.npz',crop=crop,context=context,features=np.c_[crop,context])
    (args.out/'manifest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        mapping_reference_maxabs=error,objects_sha256=hashlib.sha256((source/'objects.json').read_bytes()).hexdigest(),image_hashes=hashes,
        input='Unexpanded annotation-projected crop and full-image context; frozen CLIP, no action head',
        model_config=old['model_config'],limitations=['Projected boxes may include occlusion','No policy execution','This is not a new method']),indent=2))
    print('complete',len(rows),'agents',len(images),'images',flush=True)

if __name__=='__main__':main()
