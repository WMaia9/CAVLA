"""Real-model cache interventions, with independently fixed language/decoder RNG.

Measures fidelity to the frozen model, never counterfactual correctness or safety.
External model sources are imported read-only. All hooks are process-local.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch


def distance(a, b):
    return float(np.linalg.norm(a[..., :2] - b[..., :2], axis=-1).mean())


def prefix_projection(delta, rank, random_basis=False):
    """Per-head feature-space orthogonal projection; exact truncated SVD via Gram.

    Random projection has the same rank, but is not energy/norm matched.
    """
    d = delta.float()
    if random_basis:
        generator = torch.Generator(device=d.device).manual_seed(7129)
        basis = torch.linalg.qr(torch.randn(*d.shape[:-2], d.shape[-1], rank,
                               device=d.device, generator=generator)).Q
    else:
        _, vectors = torch.linalg.eigh(d.transpose(-1, -2) @ d)
        basis = vectors[..., -rank:]
    return ((d @ basis) @ basis.transpose(-1, -2)).to(delta.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--decoder-seeds', nargs='+', type=int, default=[9101, 9102, 9103, 9104])
    p.add_argument('--language-seed', type=int, default=123)
    p.add_argument('--save-contexts', action='store_true')
    p.add_argument('--methods', nargs='+', default=['original', 'edited', 'prefix', 'vision', 'reasoning', 'boundary',
                   'rank4', 'rank16', 'random16', 'hybrid_rank4', 'hybrid_rank16'])
    p.add_argument('--source', type=Path, default=Path('/mnt/datasets/Explainability/alpamayo/src'))
    p.add_argument('--checkpoint', type=Path, default=Path('/home/perceptron-lab/.cache/huggingface/hub/models--nvidia--Alpamayo-R1-10B/snapshots/d72c6a2757b10cacb69b688ca66ccfad3dafbf37'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pairs = json.loads(args.pairs.read_text())
    manifest = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'pairs': pairs, 'scope': 'Frozen-model response fidelity, exploratory',
                'input_protocol': 'Archived padded front still, not native multiview video',
                'torch': torch.__version__, 'python': sys.version,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'external_source_sha256': hashlib.sha256((args.source/'alpamayo_r1/models/alpamayo_r1.py').read_bytes()).hexdigest(),
                'limitations': ['No normative counterfactual action labels',
                  'Language branch sampled once per image; decoder draws do not replicate language uncertainty',
                  'Low rank uses the full edited cache: no inference speedup claim',
                  'Pilot scenes are exploratory, not independent confirmation']}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    sys.path.insert(0, str(args.source))
    import alpamayo_r1.models.alpamayo_r1 as module
    from alpamayo_r1 import helper
    # Avoid copying large diagnostic FFN tensors. Never change shared source files.
    for name in ['register_expert_hooks', 'register_vision_hooks', 'register_lm_hooks']:
        setattr(module, name, lambda *a, **kw: ({}, []))
    torch.set_num_threads(2)
    model = module.AlpamayoR1.from_pretrained(str(args.checkpoint), dtype=torch.bfloat16,
                                            local_files_only=True).to('cuda').eval()
    processor = helper.get_processor(model.tokenizer)
    original_sample = model.diffusion.sample
    records = []
    for pair_index, pair in enumerate(pairs):
        scene = pair['scene']
        scene_dir = Path('/mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene')/scene
        with np.load(scene_dir/f'{scene}_data.npz') as data:
            history_xyz = torch.from_numpy(data['ego_history_xyz']).float()[None].to('cuda')
            history_rot = torch.from_numpy(data['ego_history_rot']).float()[None].to('cuda')
        contexts, reference, image_hashes = {}, {}, {}
        for condition, path in [('original', scene_dir/f'{scene}_front.jpg'),
                                ('edited', scene_dir/'inpainted_clean_fin'/pair['image'])]:
            image_hashes[condition] = hashlib.sha256(path.read_bytes()).hexdigest()
            frames = torch.zeros(4, 4, 3, 1080, 1920, dtype=torch.uint8)
            with Image.open(path) as im:
                frames[3, 0] = torch.from_numpy(np.array(im.convert('RGB').resize((1920, 1080), Image.Resampling.LANCZOS))).permute(2, 0, 1)
            tokenized = processor.apply_chat_template(helper.create_message(frames.flatten(0, 1)),
                tokenize=True, add_generation_prompt=False, continue_final_message=True,
                return_dict=True, return_tensors='pt')
            prefix_len = tokenized['input_ids'].shape[-1]
            image_positions = (tokenized['input_ids'][0] == model.vlm.config.image_token_id).nonzero().flatten().to('cuda')
            prepared = helper.to_device({'tokenized_data': tokenized,
                'ego_history_xyz': history_xyz[:, None], 'ego_history_rot': history_rot[:, None]}, 'cuda')

            def capture_sample(*a, **kw):
                fn = kw['step_fn']
                closure = dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__]))
                contexts[condition] = {k: copy.deepcopy(closure[k]) for k in
                    ['prompt_cache', 'position_ids', 'attention_mask', 'forward_kwargs']}
                torch.manual_seed(args.decoder_seeds[0]); torch.cuda.manual_seed_all(args.decoder_seeds[0])
                return original_sample(*a, **kw)

            model.diffusion.sample = capture_sample
            random.seed(args.language_seed); np.random.seed(args.language_seed)
            torch.manual_seed(args.language_seed); torch.cuda.manual_seed_all(args.language_seed)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=prepared, top_p=.98, temperature=.6, num_traj_samples=1,
                    max_generation_length=256)
            reference[condition] = xyz.float().cpu().numpy().reshape(-1, 3)
            contexts[condition]['prefix_len'] = prefix_len
            contexts[condition]['image_positions'] = image_positions
            model.diffusion.sample = original_sample
            del xyz, prepared, frames, tokenized
            gc.collect()
            print(json.dumps({'scene': scene, 'captured': condition, 'cache_length': contexts[condition]['prompt_cache'].get_seq_length()}), flush=True)

        def decode(context, seed):
            cache = copy.deepcopy(context['prompt_cache'])
            cache_len = cache.get_seq_length()
            n = model.action_space.get_action_space_dims()[0]
            def step(x, t):
                embeds = model.action_in_proj(x, t)
                if embeds.dim() == 2:
                    embeds = embeds.view(x.shape[0], n, -1)
                output = model.expert(inputs_embeds=embeds,
                    position_ids=context['position_ids'], past_key_values=cache,
                    attention_mask=context['attention_mask'], use_cache=True,
                    **context['forward_kwargs'])
                cache.crop(cache_len)
                return model.action_out_proj(output.last_hidden_state[:, -n:]).view(-1, *model.action_space.get_action_space_dims())
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                action = original_sample(batch_size=1, step_fn=step, device='cuda', return_all_steps=False)
                xyz, _ = model.action_space.action_to_traj(action, history_xyz, history_rot)
            return xyz.float().cpu().numpy().reshape(-1, 3)

        original, edited = contexts['original'], contexts['edited']
        assert original['prefix_len'] == edited['prefix_len']
        prefix = original['prefix_len']
        if args.save_contexts:
            serialized = {'history_xyz': history_xyz.cpu(), 'history_rot': history_rot.cpu(),
                          'pair': pair, 'image_hashes': image_hashes}
            for name, ctx in contexts.items():
                serialized[name] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in ctx.items() if k != 'prompt_cache'}
                serialized[name]['cache'] = [{f: getattr(layer, f).cpu() for f in ['keys', 'values']}
                                             for layer in ctx['prompt_cache'].layers]
            torch.save(serialized, args.out/f'{scene}_contexts.pt')
            del serialized
        outputs = {}
        diagnostics = []
        methods = args.methods
        for method in methods:
            start = time.perf_counter()
            if method in ['original', 'edited']:
                context = contexts[method]
            else:
                context = copy.deepcopy(edited if method == 'reasoning' or method.startswith('hybrid_') else original)
                for layer_index, (target, ori_layer, edit_layer) in enumerate(zip(context['prompt_cache'].layers, original['prompt_cache'].layers, edited['prompt_cache'].layers)):
                    for field in ['keys', 'values']:
                        dest = getattr(target, field)
                        ori = getattr(ori_layer, field)
                        edit = getattr(edit_layer, field)
                        if method == 'reasoning':
                            dest[..., :prefix, :] = ori[..., :prefix, :]
                        elif method == 'boundary':
                            dest[..., -1:, :] = edit[..., -1:, :]
                        elif method == 'vision':
                            pos = original['image_positions']
                            dest[..., pos, :] = edit[..., pos, :]
                        elif method == 'prefix':
                            dest[..., :prefix, :] = edit[..., :prefix, :]
                        else:
                            delta = edit[..., :prefix, :].float() - ori[..., :prefix, :].float()
                            projected = prefix_projection(delta, 4 if method.endswith('rank4') else 16, method == 'random16')
                            dest[..., :prefix, :] = (ori[..., :prefix, :].float() + projected).to(dest.dtype)
                            diagnostics.append({'method': method, 'layer': layer_index, 'field': field,
                                'relative_reconstruction_error': float((delta-projected).norm()/delta.norm().clamp_min(1e-12))})
            for seed in args.decoder_seeds:
                outputs[method, seed] = decode(context, seed)
                print(json.dumps({'scene': scene, 'method': method, 'seed': seed, 'elapsed': time.perf_counter()-start}), flush=True)
            if method not in ['original', 'edited']:
                del context
        controls = {name: distance(outputs[name, args.decoder_seeds[0]], reference[name]) for name in ['original', 'edited']}
        if max(controls.values()) > 1e-4:
            raise RuntimeError(f'Decoder replay failed: {controls}')
        for method in methods:
            for seed in args.decoder_seeds:
                record = {'scene': scene, 'image': pair['image'], 'method': method, 'seed': seed,
                    'error_to_edited_m': distance(outputs[method, seed], outputs['edited', seed]),
                    'response_from_original_m': distance(outputs[method, seed], outputs['original', seed])}
                records.append(record)
        np.savez_compressed(args.out/f'{scene}_trajectories.npz', **{f'{m}_{s}': x for (m, s), x in outputs.items()})
        (args.out/f'{scene}_diagnostics.json').write_text(json.dumps({'controls': controls,
            'image_hashes': image_hashes, 'prefix_len': prefix, 'image_positions': len(original['image_positions']),
            'projection': diagnostics}, indent=2))
        (args.out/'records.json').write_text(json.dumps(records, indent=2))
        del contexts, outputs, original, edited, reference
        gc.collect(); torch.cuda.empty_cache()
    summary = {m: {'mean_error_m': float(np.mean([r['error_to_edited_m'] for r in records if r['method']==m])),
                   'n_scenes': len(pairs)} for m in methods}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
