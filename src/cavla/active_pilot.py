"""Label-blind, matched-credit acquisition followed by actual CNN adaptation.

The acquisition response stream is a synthetic stochasticization of a pretrained
small ensemble. No real VLA fine-tuning or human-time efficiency is claimed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from cavla.acquisition import METHODS, ResponseBank, acquire
from cavla.controlled import make_dataset, world_effect
from cavla.pilot import Policy, dataset_digest, evaluate, predict


REGIMES = ("none", "constant", "heterogeneous")


def flatten(dataset, views=None, scenes=None):
    images, commands, targets = dataset["images"], dataset["commands"], dataset["targets"]
    if scenes is not None:
        images, commands, targets = images[scenes], commands[scenes], targets[scenes]
    if views is not None:
        images, commands, targets = images[:, views], commands[:, views], targets[:, views]
    return {"images": images.reshape(-1, 1, *images.shape[2:]),
            "commands": commands.reshape(-1, 1), "targets": targets.reshape(-1, 1, 2)}


def tensorize(dataset, device):
    return {k: torch.from_numpy(dataset[k]).to(device) for k in ["images", "commands", "targets"]}


def build_datasets(seed, train_scenes, pool_scenes, test_scenes):
    if pool_scenes % 3:
        raise ValueError("Pool size must be divisible by three")
    origin = 1_000_000+seed*100
    train = make_dataset(train_scenes, origin)
    validation = make_dataset(96, origin+1)
    pieces = [make_dataset(pool_scenes//3, origin+2+i, shift)
              for i, shift in enumerate(["iid", "severity", "background"])]
    pool = {k: np.concatenate([d[k] for d in pieces]) for k in pieces[0]}
    tests = {shift: make_dataset(test_scenes, origin+20+i, shift)
             for i, shift in enumerate(["iid", "severity", "background"])}
    all_data = {"train": train, "validation": validation, "pool": pool, **tests}
    ids = [set(d["ids"]) for d in all_data.values()]
    if any(a & b for i, a in enumerate(ids) for b in ids[i+1:]):
        raise ValueError("Scene leakage between dataset partitions")
    return train, validation, pool, tests


def train_model(model, primary, validation, steps, seed, replay=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    best_value, best_step, best_state = float("inf"), 0, None
    curve = []
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    for step in range(1, steps+1):
        n = 32 if replay is not None else 64
        ix = torch.randint(len(primary["images"]), (n,), generator=rng).to(device)
        batch = {k: v[ix] for k, v in primary.items()}
        if replay is not None:
            ix2 = torch.randint(len(replay["images"]), (32,), generator=rng).to(device)
            batch = {k: torch.cat([v, replay[k][ix2]]) for k, v in batch.items()}
        model.train(); optimizer.zero_grad(set_to_none=True)
        pred, _ = model(batch["images"], batch["commands"])
        loss = F.mse_loss(pred, batch["targets"])
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
        if step % 100 == 0 or step == steps:
            with torch.inference_mode():
                model.eval()
                value = float(np.mean((predict(model, validation)-validation["targets"].cpu().numpy())**2))
            curve.append({"step": step, "train_mse": loss.item(), "validation_factual_mse": value})
            if value < best_value:
                best_value, best_step = value, step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    if device.type == "cuda": torch.cuda.synchronize()
    return {"best_step": best_step, "validation_factual_mse": best_value,
            "seconds": time.perf_counter()-started, "curve": curve}


@torch.inference_mode()
def response_and_features(model, dataset, device):
    model.eval()
    tensor = {k: torch.from_numpy(dataset[k]).to(device) for k in ["images", "commands"]}
    prediction, z = model(tensor["images"], tensor["commands"])
    prediction, z = prediction.cpu().numpy(), z.cpu().numpy()
    response = (world_effect(prediction).reshape(len(prediction), -1)
                if prediction.shape[1] == 4 else None)
    return response, z.mean(axis=1), prediction


def test_policy(model, datasets, device):
    results, arrays = {}, {}
    for name, dataset in datasets.items():
        prediction = predict(model, tensorize(dataset, device))
        values = evaluate(prediction, dataset)
        results[name] = {k: v for k, v in values.items() if not k.startswith("per_scene_")}
        arrays[f"{name}_prediction"] = prediction
        arrays[f"{name}_scene_ids"] = dataset["ids"]
        arrays[f"{name}_action_mse"] = values["per_scene_action_mse"]
    return results, arrays


def noise_for(regime, n, seed):
    if regime == "none": return np.zeros(n)
    if regime == "constant": return np.full(n, .04)
    if regime == "heterogeneous":
        return np.where(np.random.default_rng(seed+11111).random(n) < .2, .20, .01)
    raise ValueError(regime)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def run(args):
    if len(set(args.seeds)) != len(args.seeds) or min(args.pretrain_steps, args.adapt_steps) < 1:
        raise ValueError("Distinct seeds and positive step counts required")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a new output folder to preserve the existing experiment")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    source_dir = Path(__file__).parent
    snapshot = args.out/"source_snapshot"; snapshot.mkdir()
    hashes = {}
    for source in source_dir.glob("*.py"):
        shutil.copyfile(source, snapshot/source.name)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = Path("docs/PROTOCOLO_AQUISICAO_V1.md")
    if protocol.exists(): shutil.copyfile(protocol, args.out/protocol.name)
    manifest = {"status": "exploratory_costed_replay_and_small_CNN_adaptation",
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "argv": sys.argv, "source_sha256": hashes,
                "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest() if protocol.exists() else None,
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "primary_metric": "equal-weight mean action MSE over iid/severity/background",
                "reference_access": "Only training runner retrieves selected labels, after selection completes",
                "query_costs": "Replayed synthetic response/feature=1, exact reference quartet=label_cost",
                "common_setup": {"factual_training_labels_per_replicate": args.train_scenes*2,
                                 "factual_validation_labels_per_replicate": 96*2,
                                 "ensemble_size": 3, "pretrain_steps_per_member": args.pretrain_steps,
                                 "known_feature_queries_for_diversity": args.train_scenes},
                "limitations": ["Small CNN and two command IDs, not a pretrained VLA",
                                "Additive synthetic response noise, not native generative policy stochasticity",
                                "Credit cost ratio is assumed, not measured human or GPU cost",
                                "Ensemble and pool scores remain frozen during batch acquisition",
                                "Repeated-label utility is not learned from labels within this selector",
                                "Heuristic acquisition widths have no statistical coverage guarantee",
                                "Pilot is exploratory; no independent confirmation or novelty established"]}
    save_json(args.out/"manifest.json", manifest)
    all_runs, setup = [], []
    for seed in args.seeds:
        train, validation, pool, tests = build_datasets(seed, args.train_scenes, args.pool_scenes, args.test_scenes)
        replicate = args.out/f"seed{seed}"; replicate.mkdir()
        save_json(replicate/"dataset_hashes.json", {k: dataset_digest(d) for k, d in
                   {"train": train, "validation": validation, "pool": pool, **tests}.items()})
        known = tensorize(flatten(train, views=[0, 2]), device)
        validation_t = tensorize(flatten(validation, views=[0, 2]), device)
        members, responses = [], []
        pool_features, known_features = None, None
        initial_prediction = None
        for member in range(3):
            initialization = seed*10+member
            torch.manual_seed(initialization)
            if device.type == "cuda": torch.cuda.manual_seed_all(initialization)
            model = Policy().to(device)
            bootstrap = np.random.default_rng(initialization+30000).integers(0, len(train["images"]), len(train["images"]))
            training = tensorize(flatten(train, views=[0, 2], scenes=bootstrap), device)
            information = train_model(model, training, validation_t, args.pretrain_steps, seed+member+50000)
            members.append(copy.deepcopy(model.state_dict()))
            response, features, prediction = response_and_features(model, pool, device)
            responses.append(response)
            if member == 0:
                pool_features, initial_prediction = features, prediction
                factual_view = {k: v[:, [0, 2]] if k in ["images", "commands", "targets"] else v
                                for k, v in train.items()}
                _, known_features, _ = response_and_features(model, factual_view, device)
            torch.save(members[-1], replicate/f"pretrained_member{member}.pt")
            save_json(replicate/f"pretrained_member{member}.json", information)
            print(json.dumps({"seed": seed, "stage": "pretrain", "member": member,
                              "validation": information["validation_factual_mse"]}), flush=True)
        means = np.stack(responses)
        np.savez_compressed(replicate/"frozen_responses_and_features.npz", responses=means,
                            pool_features=pool_features, known_features=known_features, pool_ids=pool["ids"])
        base = Policy().to(device); base.load_state_dict(members[0])
        base_metrics, arrays = test_policy(base, tests, device)
        np.savez_compressed(replicate/"initial_policy_test.npz", **arrays)
        continuation = Policy().to(device); continuation.load_state_dict(members[0])
        info = train_model(continuation, known, validation_t, args.adapt_steps, seed+70000)
        continuation_metrics, arrays = test_policy(continuation, tests, device)
        np.savez_compressed(replicate/"continued_without_labels_test.npz", **arrays)
        setup.append({"seed": seed, "initial": base_metrics, "continued_without_labels": continuation_metrics,
                      "continuation_training": info})
        save_json(args.out/"setup.json", setup)
        for regime in args.regimes:
            noise = noise_for(regime, args.pool_scenes, seed)
            for budget in args.budgets:
                for method in args.methods:
                    bank = ResponseBank(means, pool_features, known_features, noise,
                                        seed+80000, budget, args.label_cost)
                    start = time.perf_counter()
                    selection = acquire(bank, method)
                    selection_seconds = time.perf_counter()-start
                    selected = np.asarray(selection["selected"], dtype=int)
                    if not len(selected): raise ValueError("No labels acquired")
                    name = f"{regime}_b{budget}_{method}"
                    save_json(replicate/f"{name}_acquisition.json", selection)
                    # This is the first reference-label access for pool rows;
                    # the bank/selector neither stores nor receives targets.
                    acquired = tensorize(flatten(pool, scenes=selected), device)
                    model = Policy().to(device); model.load_state_dict(members[0])
                    training = train_model(model, acquired, validation_t, args.adapt_steps,
                                           seed+70000, replay=known)
                    metrics, arrays = test_policy(model, tests, device)
                    np.savez_compressed(replicate/f"{name}_test.npz", **arrays)
                    torch.save(model.state_dict(), replicate/f"{name}.pt")
                    save_json(replicate/f"{name}_training.json", training)
                    item = {"seed": seed, "regime": regime, "budget": budget, "method": method,
                            "spent": selection["spent"], "n_labels": selection["n_labels"],
                            "response_queries": selection["response_queries"],
                            "feature_queries": selection["feature_queries"],
                            "extra_response_queries": selection["extra_response_queries"],
                            "selection_seconds": selection_seconds, "adapt_seconds": training["seconds"],
                            "validation_factual_mse": training["validation_factual_mse"],
                            "primary_mse": float(np.mean([v["action_mse"] for v in metrics.values()])),
                            "splits": metrics,
                            "posthoc_selected_initial_reference_mse": float(np.mean((initial_prediction[selected]-pool["targets"][selected])**2))}
                    all_runs.append(item)
                    save_json(args.out/"runs.json", all_runs)
                    print(json.dumps({k: item[k] for k in ["seed", "regime", "budget", "method",
                                     "n_labels", "extra_response_queries", "primary_mse"]}), flush=True)
    save_json(args.out/"completion.json", {"complete": True, "adaptation_runs": len(all_runs),
              "pretraining_runs": 3*len(args.seeds), "continuation_runs": len(args.seeds)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    p.add_argument("--pretrain-steps", type=int, default=1500)
    p.add_argument("--adapt-steps", type=int, default=800)
    p.add_argument("--train-scenes", type=int, default=224)
    p.add_argument("--pool-scenes", type=int, default=96)
    p.add_argument("--test-scenes", type=int, default=256)
    p.add_argument("--label-cost", type=int, default=32)
    p.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    run(p.parse_args())


if __name__ == "__main__":
    main()
