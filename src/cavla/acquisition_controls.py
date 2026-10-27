"""Equal-repeat controls using frozen setup from an existing acquisition study."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from cavla.acquisition import Moments, ResponseBank
from cavla.active_pilot import (build_datasets, flatten, save_json, tensorize,
                                test_policy, train_model)
from cavla.pilot import Policy, dataset_digest


def acquire_matched(bank, extra_queries, method):
    if method not in ["matched_uniform", "matched_random"]:
        raise ValueError(method)
    if extra_queries < 0:
        raise ValueError("Negative repeat count")
    required = 2*bank.n_members*bank.n_candidates+extra_queries+bank.label_cost
    if required > bank.remaining:
        raise ValueError("Insufficient budget for matched repeats and one label")
    moments = Moments.empty(bank.n_members, bank.n_candidates, bank.dimension)
    for i in range(bank.n_candidates):
        for member in range(bank.n_members):
            for _ in range(2): moments.add(i, member, bank.query(i, member))
    rng = np.random.default_rng(bank.seed+77111)
    for draw in range(extra_queries):
        if method == "matched_uniform":
            index, member = draw % bank.n_candidates, (draw//bank.n_candidates) % bank.n_members
        else:
            index, member = int(rng.integers(bank.n_candidates)), int(rng.integers(bank.n_members))
        moments.add(index, member, bank.query(index, member))
    score = moments.estimates()["corrected"]
    tie_break = np.random.default_rng(bank.seed+91000).uniform(0, 1e-12, bank.n_candidates)
    ranked = np.argsort(-(score+tie_break), kind="stable")
    for index in ranked:
        if bank.remaining < bank.label_cost: break
        bank.label(int(index))
    return {"method": method, "selected": bank.selected.copy(), "spent": bank.spent,
            "budget": bank.budget, "label_cost": bank.label_cost, "n_labels": len(bank.selected),
            "response_queries": int(bank.counts.sum()), "feature_queries": 0,
            "extra_response_queries": extra_queries, "trace": bank.trace.copy()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    complete = json.loads((args.study/"completion.json").read_text())
    if not complete["complete"]: raise ValueError("Main study must complete first")
    if args.out.exists() and any(args.out.iterdir()): raise ValueError("Use a new output folder")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.study/"manifest.json").read_text())
    config = manifest["config"]
    snapshot = args.out/"source_snapshot"; snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"): shutil.copyfile(source, snapshot/source.name)
    addendum = Path("docs/ADENDO_CONTROLES_AQUISICAO_V1.md")
    shutil.copyfile(addendum, args.out/addendum.name)
    save_json(args.out/"manifest.json", {"source_study": str(args.study),
              "source_manifest_sha256": hashlib.sha256((args.study/"manifest.json").read_bytes()).hexdigest(),
              "addendum_sha256": hashlib.sha256(addendum.read_bytes()).hexdigest(),
              "config": config, "scope": "Post-first-replicate exploratory equal-cost controls",
              "extra_queries": "Taken only from the adaptive acquisition ledger, never test scores"})
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    runs = []
    for seed in config["seeds"]:
        train, validation, pool, tests = build_datasets(seed, config["train_scenes"],
                                                       config["pool_scenes"], config["test_scenes"])
        original = args.study/f"seed{seed}"
        expected = json.loads((original/"dataset_hashes.json").read_text())
        for key, data in {"train": train, "validation": validation, "pool": pool, **tests}.items():
            if dataset_digest(data) != expected[key]: raise ValueError(f"Dataset drift: {key}")
        destination = args.out/f"seed{seed}"; destination.mkdir()
        frozen = np.load(original/"frozen_responses_and_features.npz", allow_pickle=False)
        means = frozen["responses"]
        features, known_features = frozen["pool_features"], frozen["known_features"]
        known = tensorize(flatten(train, views=[0, 2]), device)
        val = tensorize(flatten(validation, views=[0, 2]), device)
        initial = torch.load(original/"pretrained_member0.pt", map_location=device, weights_only=True)
        from cavla.active_pilot import noise_for
        for regime in config["regimes"]:
            noise = noise_for(regime, config["pool_scenes"], seed)
            for budget in config["budgets"]:
                adaptive = json.loads((original/f"{regime}_b{budget}_adaptive_corrected_acquisition.json").read_text())
                for method in ["matched_uniform", "matched_random"]:
                    bank = ResponseBank(means, features, known_features, noise, seed+80000,
                                        budget, config["label_cost"])
                    start = time.perf_counter()
                    selection = acquire_matched(bank, adaptive["extra_response_queries"], method)
                    selection_seconds = time.perf_counter()-start
                    for key in ["spent", "n_labels", "response_queries", "extra_response_queries"]:
                        if selection[key] != adaptive[key]: raise ValueError(f"Cost mismatch: {key}")
                    name = f"{regime}_b{budget}_{method}"
                    save_json(destination/f"{name}_acquisition.json", selection)
                    selected = np.asarray(selection["selected"], dtype=int)
                    acquired = tensorize(flatten(pool, scenes=selected), device)
                    model = Policy().to(device); model.load_state_dict(initial)
                    training = train_model(model, acquired, val, config["adapt_steps"], seed+70000, replay=known)
                    metrics, arrays = test_policy(model, tests, device)
                    np.savez_compressed(destination/f"{name}_test.npz", **arrays)
                    torch.save(model.state_dict(), destination/f"{name}.pt")
                    save_json(destination/f"{name}_training.json", training)
                    item = {"seed": seed, "regime": regime, "budget": budget, "method": method,
                            "spent": selection["spent"], "n_labels": selection["n_labels"],
                            "response_queries": selection["response_queries"], "feature_queries": 0,
                            "extra_response_queries": selection["extra_response_queries"],
                            "selection_seconds": selection_seconds, "adapt_seconds": training["seconds"],
                            "primary_mse": float(np.mean([v["action_mse"] for v in metrics.values()])),
                            "splits": metrics}
                    runs.append(item); save_json(args.out/"runs.json", runs)
                    print(json.dumps({k: item[k] for k in ["seed", "regime", "budget", "method", "primary_mse"]}), flush=True)
    save_json(args.out/"completion.json", {"complete": True, "adaptation_runs": len(runs),
                                           "pretraining_runs": 0})


if __name__ == "__main__":
    main()
