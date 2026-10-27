"""Repeated-input and image-edit smoke study with the locally cached Alpamayo 1.

Mirrors the archived single-front-image/zero-padding protocol deliberately. This
does NOT validate that protocol as the model's native input distribution. It
does not test instruction conditioning (unsupported in the original release).
"""
import argparse
import copy
import gc
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch

from cavla.audit_existing import trajectory_response
from cavla.distribution_effects import paired_mean_effect_squared, paired_rbf_mmd_squared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--edited-image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 124, 125, 126])
    args = p.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        p.error("Provide at least two distinct seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source))
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
    torch.set_num_threads(2)
    scene = args.scene_dir.name
    original_image = args.scene_dir/f"{scene}_front.jpg"
    with np.load(args.scene_dir/f"{scene}_data.npz", allow_pickle=False) as data:
        history_xyz, history_rot = data["ego_history_xyz"], data["ego_history_rot"]
    source_file = args.source/"alpamayo_r1/models/alpamayo_r1.py"
    source_files = sorted((args.source/"alpamayo_r1").rglob("*.py"))
    source_digest = hashlib.sha256()
    for f in source_files:
        source_digest.update(str(f.relative_to(args.source)).encode())
        source_digest.update(f.read_bytes())
    manifest = {"scope": "One-scene stochastic-response smoke study, not causal or safety validation",
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source_model_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                "source_tree_sha256": source_digest.hexdigest(),
                "checkpoint_metadata_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in [args.checkpoint/"config.json", args.checkpoint/"model.safetensors.index.json"]
                           if f.exists()},
                "image_sha256": {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in [original_image, args.edited_image]},
                "python": sys.version, "torch": torch.__version__,
                "input_protocol": "4 times x 4 camera slots; only last-time front-camera is populated",
                "sampling": {"top_p": .98, "temperature": .6, "max_generation_length": 256,
                             "num_traj_samples": 1}}
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Loading cached Alpamayo 1", flush=True)
    model = AlpamayoR1.from_pretrained(str(args.checkpoint), dtype=torch.bfloat16,
                                      local_files_only=True).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    prepared = {}
    for name, image_path in [("original", original_image), ("edited", args.edited_image)]:
        frames = torch.zeros(4, 4, 3, 1080, 1920, dtype=torch.uint8)
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS)
            frames[3, 0] = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        messages = helper.create_message(frames.flatten(0, 1))
        tokenized = processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=False, continue_final_message=True,
                    return_dict=True, return_tensors="pt")
        prepared[name] = helper.to_device({"tokenized_data": tokenized,
                         "ego_history_xyz": torch.from_numpy(history_xyz).float()[None, None],
                         "ego_history_rot": torch.from_numpy(history_rot).float()[None, None]}, "cuda")
    predictions, runs = {}, []
    conditions = [("original", seed, 0) for seed in args.seeds]
    conditions += [("original", args.seeds[0], 1)]
    conditions += [("edited", seed, 0) for seed in args.seeds]
    with torch.inference_mode():
        for name, seed, repeat in conditions:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            data = copy.deepcopy(prepared[name])  # model pops input_ids from its input dict
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=data, top_p=.98, temperature=.6, num_traj_samples=1,
                    max_generation_length=256, return_extra=True)
            torch.cuda.synchronize()
            pred = xyz.float().cpu().numpy()[0, 0]
            predictions[(name, seed, repeat)] = pred
            np.savez_compressed(args.out/f"{name}_seed{seed}_repeat{repeat}.npz", pred_xyz=pred)
            record = {"condition": name, "seed": seed, "repeat": repeat,
                      "seconds": time.perf_counter()-start,
                      "finite": bool(np.isfinite(pred).all())}
            runs.append(record)
            (args.out/"runs.json").write_text(json.dumps(runs, indent=2)+"\n")
            print(json.dumps(record), flush=True)
            del xyz, extra, data
            gc.collect()
    same_seed_noop = trajectory_response(predictions[("original", args.seeds[0], 0)],
                                         predictions[("original", args.seeds[0], 1)])[0]
    independent_original = [trajectory_response(predictions[("original", a, 0)],
                             predictions[("original", b, 0)])[0]
                            for a, b in itertools.combinations(args.seeds, 2)]
    paired_edits = [trajectory_response(predictions[("original", s, 0)],
                   predictions[("edited", s, 0)])[0] for s in args.seeds]
    original_draws = np.stack([predictions[("original", s, 0)][..., :2] for s in args.seeds])
    edited_draws = np.stack([predictions[("edited", s, 0)][..., :2] for s in args.seeds])
    summary = {"same_seed_identical_input_response_m": same_seed_noop,
               "different_seed_original_path_distances_m": independent_original,
               "same_seed_image_edit_path_distances_m": paired_edits,
               "mean_shift_diagnostic": paired_mean_effect_squared(original_draws, edited_draws),
               "paired_mmd2_fixed_bandwidths": {str(b): paired_rbf_mmd_squared(original_draws, edited_draws, b)
                                                for b in [.5, 1., 2., 5.]},
               "n_scenes": 1, "n_seeds": len(args.seeds),
               "limitations": ["One lexicographically selected archived scene and one edited image",
                               "Different-seed distances measure sample diversity, not the standard error of a mean",
                               "Same-seed original/edit generation may consume different random draws",
                               "U-statistics assume independent seed pairs; marginal stream validity is not proven by same-seed repeatability",
                               "No significance test or calibrated interval is claimed for these few-seed U-statistics",
                               "No correct physical counterfactual reference; no safety or task-success conclusion",
                               "Mirrors a padded single-camera archived protocol, not native multiview video"]}
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
