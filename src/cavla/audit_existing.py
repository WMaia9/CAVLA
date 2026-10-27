"""Read-only audit of archived Alpamayo image edits; no causal-world labels.

Direct changes in sampled model trajectories are descriptive responses. They are
not correct counterfactual trajectories, safety gains, or intervention ground truth.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np
from PIL import Image

from cavla.statistics import cluster_bootstrap, correlation, grouped_folds, ridge_predict


CATEGORIES = ["car", "person", "traffic light", "truck", "bus", "bicycle", "motorcycle"]


def trajectory_response(original, edited):
    """Sample-index coupling and change of mean path, in the stored coordinates."""
    a, b = np.asarray(original), np.asarray(edited)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] < 2:
        raise ValueError(f"Incompatible trajectory shapes: {a.shape}, {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite trajectories")
    paired = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1).mean()
    mean_path = np.linalg.norm(a[..., :2].mean(0) - b[..., :2].mean(0), axis=-1).mean()
    return float(paired), float(mean_path)


def matched_box(stem, metadata, width, height):
    """Never join changed detection lists by ordinal IDs alone."""
    match = re.fullmatch(r"\d+_(.+)_conf([0-9.]+)", stem)
    if not match:
        return None
    label, conf = match.group(1), float(match.group(2))
    candidates = [o for o in metadata.get("objects", [])
                  if o.get("label") == label and abs(o.get("confidence", -1) - conf) <= .00501]
    if len(candidates) != 1:
        return None
    x0, y0, x1, y1 = candidates[0]["box_xyxy"]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    return [(x1-x0)*(y1-y0)/(width*height), (x0+x1)/(2*width), (y0+y1)/(2*height)]


def load_records(results_dir, images_dir):
    records, errors, lineage = [], [], []
    for scene_dir in sorted(results_dir.glob("scene-*")):
        table = scene_dir / f"{scene_dir.name}_comparison.csv"
        original_path = scene_dir / "trajectories/original_traj.npz"
        if not table.exists() or not original_path.exists():
            errors.append({"scene": scene_dir.name, "reason": "missing table/original"})
            continue
        content = table.read_bytes()
        lineage.append({"path": str(table), "sha256": hashlib.sha256(content).hexdigest()})
        rows = list(csv.DictReader(content.decode().splitlines()))
        originals = [r for r in rows if r["variant"] == "original"]
        if len(originals) != 1:
            raise ValueError(f"Expected one original in {table}")
        with np.load(original_path, allow_pickle=False) as data:
            orig, gt = data["pred_xyz"], data["gt_xyz"]
        orig_ade = float(np.linalg.norm(orig[..., :2] - gt[None, ..., :2], axis=-1).mean(-1).min())
        image_scene = images_dir / scene_dir.name
        metadata_path = image_scene / f"{scene_dir.name}_seg.json"
        image_path = image_scene / f"{scene_dir.name}_front.jpg"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        dimensions = Image.open(image_path).size if image_path.exists() else None
        for row in rows:
            if row["variant"] == "original":
                continue
            stem = row["variant"].removeprefix("removed_")
            traj_path = scene_dir / "trajectories" / f"{stem}_traj.npz"
            delta_path = scene_dir / "whitebox" / f"{stem}_delta.npz"
            if not traj_path.exists() or not delta_path.exists():
                errors.append({"scene": scene_dir.name, "stem": stem, "reason": "missing trajectory/features"})
                continue
            try:
                with np.load(traj_path, allow_pickle=False) as data:
                    pred, edited_gt = data["pred_xyz"], data["gt_xyz"]
                if gt.shape != edited_gt.shape or not np.allclose(gt, edited_gt):
                    raise ValueError("Logged reference differs between paired files")
                if row.get("seed") != originals[0].get("seed"):
                    raise ValueError("Original and edit have different recorded seeds")
                paired, mean_path = trajectory_response(orig, pred)
                edited_ade = float(np.linalg.norm(pred[..., :2]-gt[None, ..., :2], axis=-1).mean(-1).min())
                box = matched_box(stem, metadata, *dimensions) if dimensions else None
                label = row.get("object_label", "unknown")
                rec = {"scene": scene_dir.name, "stem": stem, "label": label,
                       "recorded_seed": row.get("seed"), "n_samples": len(pred),
                       "paired_response": paired, "mean_path_response": mean_path,
                       "abs_delta_minade": abs(edited_ade-orig_ade),
                       "original_minade": orig_ade,
                       "csv_delta_error": abs((edited_ade-orig_ade)-float(row["delta_minADE"])),
                       "box_match": box is not None,
                       "box_area": box[0] if box else None,
                       "box_x": box[1] if box else None,
                       "box_y": box[2] if box else None,
                       "trajectory_path": str(traj_path), "feature_path": str(delta_path)}
                with np.load(delta_path, allow_pickle=False) as dc:
                    rec["handoff_cosine"] = float(dc["traj_token_delta"].reshape(-1)[0])
                    for i, v in enumerate(dc["lm_global_delta"].reshape(-1)):
                        rec[f"lm_global_{i}"] = float(v)
                records.append(rec)
            except (ValueError, KeyError, OSError) as exc:
                errors.append({"scene": scene_dir.name, "stem": stem, "reason": str(exc)})
    return records, errors, lineage


def predict_grouped(records):
    groups = np.array([r["scene"] for r in records])
    target = np.array([r["mean_path_response"] for r in records])
    metadata = np.array([[float(r["label"] == cat) for cat in CATEGORIES] +
                         [r["box_area"], r["box_x"], r["box_y"], float(r["box_match"])]
                         for r in records], dtype=float)
    handoff = np.array([[r["handoff_cosine"]] for r in records])
    curves = np.array([[r[k] for k in sorted(r) if k.startswith("lm_global_")]
                       for r in records])
    designs = {"metadata": metadata, "handoff": handoff,
               "metadata_handoff": np.c_[metadata, handoff],
               "metadata_lm_curve": np.c_[metadata, curves]}
    predictions = {k: np.zeros(len(records)) for k in ["train_median", *designs]}
    fold_ids = np.full(len(records), -1)
    fold_summary = []
    for fold, (train, test) in enumerate(grouped_folds(groups)):
        fold_ids[test] = fold
        predictions["train_median"][test] = np.median(target[train])
        fold_summary.append({"fold": fold, "train_scenes": sorted(set(groups[train])),
                             "test_scenes": sorted(set(groups[test]))})
        for name, x in designs.items():
            log_pred = ridge_predict(x[train], np.log1p(target[train]), x[test], alpha=10.)
            predictions[name][test] = np.maximum(np.expm1(np.clip(log_pred, -10, 10)), 0.)
    metrics = {}
    reference_error = np.abs(predictions["metadata"]-target)
    for name, pred in predictions.items():
        error = np.abs(pred-target)
        metrics[name] = {"scene_mae": cluster_bootstrap(error, groups),
                         "pooled_spearman": correlation(pred, target),
                         "paired_scene_mae_gain_over_metadata": cluster_bootstrap(reference_error-error, groups)}
    for i, row in enumerate(records):
        row["fold"] = int(fold_ids[i])
        for name, pred in predictions.items():
            row[f"oof_{name}"] = float(pred[i])
    return metrics, fold_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, errors, lineage = load_records(args.results_dir, args.images_dir)
    if len({r["scene"] for r in records}) < 5:
        raise ValueError("Need at least five scenes")
    metrics, folds = predict_grouped(records)
    groups = np.array([r["scene"] for r in records])
    response = np.array([r["mean_path_response"] for r in records])
    proxy = np.array([r["abs_delta_minade"] for r in records])
    report = {"status": "descriptive_archived_model_response_not_causal_ground_truth",
              "n_pairs": len(records), "n_scenes": len(set(groups)),
              "sample_counts": sorted(set(r["n_samples"] for r in records)),
              "recorded_seeds": sorted(set(r["recorded_seed"] for r in records)),
              "box_matching_coverage": np.mean([r["box_match"] for r in records]).item(),
              "response_mean_path": cluster_bootstrap(response, groups),
              "proxy_vs_direct_response_spearman": correlation(proxy, response),
              "handoff_vs_direct_response_spearman": correlation([r["handoff_cosine"] for r in records], response),
              "fraction_proxy_less_than_half_response": float(np.mean(proxy < response/2)),
              "max_csv_delta_discrepancy": max(r["csv_delta_error"] for r in records),
              "grouped_oof_ridge_fixed_alpha_10": metrics,
              "exclusions": errors,
              "limitations": ["No repeated identical-input trials: stochastic noise floor unknown",
                              "Recorded same seed does not prove equal random draws after variable-length generation",
                              "Edited images have no correct physical counterfactual trajectory",
                              "No instruction quartets: task-conditioned hypothesis untested on these archives",
                              "Absolute change in logged ADE is not direct trajectory displacement",
                              "Detection metadata can drift; ambiguous label-confidence joins are missing, never ordinal joins",
                              "OOF predictions are exploratory fixed-alpha probes, not independent final confirmation"],
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}}
    for filename, data in [("summary.json", report), ("folds.json", folds), ("lineage.json", lineage)]:
        (args.out/filename).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    with (args.out/"pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps({k: report[k] for k in ["n_pairs", "n_scenes", "box_matching_coverage",
                     "proxy_vs_direct_response_spearman", "handoff_vs_direct_response_spearman"]}, indent=2))
    print(f"Saved audit: {args.out}")


if __name__ == "__main__":
    main()
