"""Matched-budget pilot: six objectives, identical images/labels/model/seeds.

All four corners have action labels in every method. Effect losses introduce no
new target information: this specifically tests their optimization bias. This
pilot is insufficient to establish VLA-scale novelty or real driving benefits.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cavla.controlled import COMMANDS, make_dataset, mixed_effect, world_effect
from cavla.statistics import cluster_bootstrap


METHODS = ("bc", "aux_effect", "policy_effect", "mixed_policy", "random_contrast", "shuffled_mixed")


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(3, 12, 3, stride=2, padding=1), nn.SiLU(),
                                    nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.SiLU(),
                                    nn.Flatten(), nn.Linear(24*6*6, 48), nn.SiLU())
        self.language = nn.Embedding(2, 8)
        self.fusion = nn.Sequential(nn.Linear(56, 48), nn.SiLU(), nn.Linear(48, 16), nn.SiLU())
        self.action = nn.Linear(16, 2)
        self.effect = nn.Linear(16, 2, bias=False)

    def forward(self, images, commands):
        batch, views = images.shape[:2]
        x = images.flatten(0, 1)
        c = commands.flatten(0, 1)
        z = self.fusion(torch.cat([self.vision(x), self.language(c)], dim=-1))
        return self.action(z).reshape(batch, views, 2), z.reshape(batch, views, 16)


def tensors(dataset, device):
    return {k: torch.from_numpy(dataset[k]).to(device) for k in ["images", "commands", "targets", "null"]}


def losses(model, data, method, permutation):
    prediction, z = model(data["images"], data["commands"])
    target = data["targets"]
    bc = F.mse_loss(prediction, target)
    if method == "bc":
        extra = bc.new_zeros(())
    elif method == "aux_effect":
        extra = F.mse_loss(model.effect(world_effect(z)), world_effect(target))
    elif method == "policy_effect":
        extra = F.mse_loss(world_effect(prediction), world_effect(target))
    elif method == "mixed_policy":
        extra = F.mse_loss(mixed_effect(prediction), mixed_effect(target))
    elif method == "shuffled_mixed":
        extra = F.mse_loss(mixed_effect(prediction), mixed_effect(target)[permutation])
    elif method == "random_contrast":
        # Generic supervised differences; same target information and quartet count.
        extra = F.mse_loss(prediction[:, 1]-prediction[permutation, 2],
                           target[:, 1]-target[permutation, 2])
    else:
        raise ValueError(method)
    return bc, extra


@torch.inference_mode()
def predict(model, data, batch_size=128):
    model.eval()
    return torch.cat([model(data["images"][i:i+batch_size], data["commands"][i:i+batch_size])[0]
                      for i in range(0, len(data["images"]), batch_size)]).cpu().numpy()


def evaluate(pred, dataset):
    target = dataset["targets"]
    per_scene_mse = ((pred-target)**2).mean((1, 2))
    effect_mse = ((world_effect(pred)-world_effect(target))**2).mean((1, 2))
    mixed_mse = ((mixed_effect(pred)-mixed_effect(target))**2).mean(1)
    null = dataset["null"]
    return {"action_mse": float(per_scene_mse.mean()), "world_effect_rmse": float(np.sqrt(effect_mse.mean())),
            "mixed_effect_rmse": float(np.sqrt(mixed_mse.mean())),
            "null_response": float(np.linalg.norm(world_effect(pred)[null], axis=-1).mean()),
            "per_scene_action_mse": per_scene_mse,
            "per_scene_world_effect_mse": effect_mse,
            "per_scene_mixed_effect_mse": mixed_mse}


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for key in sorted(dataset):
        digest.update(key.encode()); digest.update(dataset[key].tobytes())
    return digest.hexdigest()


def run(args):
    if args.steps < 1 or args.train_scenes < 1 or args.test_scenes < 8:
        raise ValueError("Invalid pilot size")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    train = make_dataset(args.train_scenes, 11001)
    validation = make_dataset(256, 22002)
    tests = {"iid": make_dataset(args.test_scenes, 33003),
             "severity": make_dataset(args.test_scenes, 44004, "severity"),
             "background": make_dataset(args.test_scenes, 55005, "background")}
    # Seeds identify exogenous scenes; no view-level train/test split.
    datasets = {"train": train, "validation": validation, **tests}
    ids = [set(x["ids"]) for x in datasets.values()]
    assert all(not a.intersection(b) for i, a in enumerate(ids) for b in ids[i+1:])
    tensor_train, tensor_validation = tensors(train, device), tensors(validation, device)
    tensor_tests = {name: tensors(d, device) for name, d in tests.items()}
    manifest = {"argv": sys.argv, "commands": COMMANDS, "torch": torch.__version__,
                "numpy": np.__version__, "python": sys.version,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset_sha256": {name: dataset_digest(d) for name, d in datasets.items()},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")},
                "scope": "Synthetic image plus two command IDs; no pretrained VLA, no physical driving validation",
                "selection": "Best in-distribution validation action MSE; test splits never select checkpoints",
                "label_budget": "All methods receive the exact same four images and action labels per scene",
                "effect_weight": args.effect_weight}
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    raw_metrics, errors = [], {}
    for seed in args.seeds:
        for method in args.methods:
            # Reset both model initialization and sample-order RNG for every method.
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device.type == "cuda": torch.cuda.manual_seed_all(seed)
            model = Policy().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            sampling = torch.Generator().manual_seed(seed+9000)
            best, best_state, best_step = float("inf"), None, 0
            learning_curve = []
            start = time.perf_counter()
            for step in range(1, args.steps+1):
                index = torch.randint(len(train["images"]), (args.batch_size,), generator=sampling).to(device)
                permutation = torch.randperm(args.batch_size, generator=sampling).to(device)
                batch = {k: v[index] for k, v in tensor_train.items()}
                model.train(); optimizer.zero_grad(set_to_none=True)
                action_loss, extra = losses(model, batch, method, permutation)
                loss = action_loss+args.effect_weight*extra
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                if step % 50 == 0 or step == args.steps:
                    val_pred = predict(model, tensor_validation)
                    value = float(((val_pred-validation["targets"])**2).mean())
                    learning_curve.append({"step": step, "train_action": action_loss.item(),
                                           "train_extra": extra.item(), "validation_action": value})
                    if value < best:
                        best, best_step = value, step
                        best_state = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            elapsed = time.perf_counter()-start
            run_data = {"seed": seed, "method": method, "best_step": best_step,
                        "validation_action_mse": best, "elapsed_seconds": elapsed, "splits": {}}
            for name, dataset in tests.items():
                pred = predict(model, tensor_tests[name])
                result = evaluate(pred, dataset)
                run_data["splits"][name] = {k: v for k, v in result.items() if not k.startswith("per_scene_")}
                for key, value in result.items():
                    if key.startswith("per_scene_"):
                        errors.setdefault((method, name, key), []).append(value)
                np.savez_compressed(args.out/f"{method}_seed{seed}_{name}.npz",
                                    prediction=pred, target=dataset["targets"], scene_ids=dataset["ids"])
            torch.save(best_state, args.out/f"{method}_seed{seed}.pt")
            (args.out/f"{method}_seed{seed}_curve.json").write_text(json.dumps(learning_curve, indent=2)+"\n")
            raw_metrics.append(run_data)
            (args.out/"runs.json").write_text(json.dumps(raw_metrics, indent=2)+"\n")
            print(json.dumps({"seed": seed, "method": method, "seconds": round(elapsed, 2),
                              "iid_mse": run_data["splits"]["iid"]["action_mse"],
                              "severity_mse": run_data["splits"]["severity"]["action_mse"]}), flush=True)
    summary = {}
    for method in args.methods:
        summary[method] = {}
        for name, dataset in tests.items():
            result = {}
            for key in ["per_scene_action_mse", "per_scene_world_effect_mse", "per_scene_mixed_effect_mse"]:
                values = np.stack(errors[(method, name, key)])
                result[key] = {"seed_means": values.mean(1).tolist(),
                               "scene_bootstrap_of_seed_mean": cluster_bootstrap(values.mean(0), dataset["ids"])}
                if "bc" in args.methods:
                    baseline = np.stack(errors[("bc", name, key)])
                    gain = (baseline-values).mean(0)
                    result[key]["paired_gain_over_bc"] = cluster_bootstrap(gain, dataset["ids"])
            summary[method][name] = result
    output = {"status": "exploratory_controlled_pilot_not_ICML_evidence",
              "methods": summary,
              "limitations": ["Hand-designed oracle and small CNN; no pretrained language/vision knowledge",
                              "One fixed training dataset; seeds vary initialization and minibatch order only",
                              "Scene CIs condition on these trained models; seed means also reported",
                              "No lambda/model-size search; one predeclared weight may favor or hurt objectives",
                              "Effect labels are algebraically redundant given the four action labels",
                              "Background/severity shifts are designed probes, not real deployment distributions",
                              "Every result is exploratory; final confirmation needs untouched data and larger policies"]}
    (args.out/"summary.json").write_text(json.dumps(output, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--train-scenes", type=int, default=768)
    p.add_argument("--test-scenes", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--effect-weight", type=float, default=1.)
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    run(p.parse_args())


if __name__ == "__main__":
    main()
