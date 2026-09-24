#!/usr/bin/env python3
"""H&E/p53 registration, staining measurements and patient-grouped supervision.

Run with --help and see P53_WORKFLOW.md. No clustering is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from p53_registration import (KEYS, extract_pair, read_pairs, register_pair,
                              select_pairs, write_json, pair_signature)

# An explicit H&E measurement vocabulary prevents labels/IDs/CellViT predictions
# or new p53-derived columns from silently entering the model as numeric features.
MORPHOLOGY = set("""area_um2 perimeter_um equivalent_diameter_um major_axis_length_um
minor_axis_length_um aspect_ratio eccentricity circularity roundness solidity
convexity concavity_count perimeter_area_ratio nuclear_membrane_irregularity_index
fractal_dimension_boundary convex_hull_area_um2 he_ratio_mean integrated_nuclear_density_h
integrated_nuclear_density_e nuclear_cytoplasm_h_contrast nuclear_cytoplasm_e_contrast
nc_area_ratio_proxy chromatin_clumping_index nucleolar_peak_count nucleolar_peak_density
nucleolar_peak_contrast_max nn_distance_mean_um nn_distance_std_um nn_distance_min_um
delaunay_degree delaunay_edge_length_mean_um delaunay_edge_length_var_um
graph_clustering_coefficient voronoi_area_um2 voronoi_polygon_irregularity
orientation_coherence_circular_variance orientation_disorder pseudostratification_index
epithelial_layer_thickness_um basal_to_luminal_position distance_from_basal_line_um
nuclear_overlap_crowding_index""".split())
STAIN_STATS = {f"{prefix}_{stat}" for prefix in ("hematoxylin", "eosin", "residual", "cytoplasm_ring_hematoxylin", "cytoplasm_ring_eosin")
               for stat in ("mean", "median", "std", "skew", "kurtosis", "min", "max", "p10", "p90")}
TEXTURE = ({f"fourier_h{i}_amp" for i in range(1, 11)} | {f"lbp_bin_{i}" for i in range(10)} |
           {f"glcm_{s}" for s in ("contrast", "correlation", "energy", "homogeneity", "entropy")} |
           {f"haar_{s}_energy" for s in ("ll", "lh", "hl", "hh")} |
           {f"gabor_f{f}_theta{t}_{s}" for f in ("0.15", "0.30") for t in range(4) for s in ("mean", "std")})
SPATIAL = {f"{s}_r{r}um" for s in ("local_density", "neighbor_count", "ripleys_k_l_proxy") for r in (20, 50, 100)}
HE_FEATURES = MORPHOLOGY | STAIN_STATS | TEXTURE | SPATIAL


def validate_keys(frame, context):
    missing = set(KEYS) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"{context}: empty table or missing cell keys {sorted(missing)}")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{context}: null or duplicate (slide, region, cell_label) keys")
    ids = pd.to_numeric(frame.cell_label, errors="coerce")
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids != ids.round()).any():
        raise ValueError(f"{context}: cell_label must be a positive integer")
    frame["cell_label"] = ids.astype("int64")


def read_features(path):
    path = Path(path)
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(p for p in path.glob("*_cell_features.csv") if p.name != "all_cell_features.csv")
        if not paths and (path / "all_cell_features.csv").exists():
            paths = [path / "all_cell_features.csv"]
    if not paths:
        raise ValueError(f"No raw Script03 feature files in {path}")
    frames = [pd.read_csv(p, dtype={"slide": str, "region": str}) for p in paths]
    frame = pd.concat(frames, ignore_index=True)
    validate_keys(frame, "H&E features")
    cols = sorted(HE_FEATURES.intersection(frame.columns))
    if not cols:
        raise ValueError("No recognized raw H&E measurements; use Script03 --stage extract output")
    return frame[KEYS + cols]


def read_annotations(path, target):
    frame = pd.read_csv(path, dtype={"slide": str, "region": str, "patient_id": str, "reviewer": str})
    validate_keys(frame, "Annotations")
    required = {target, "patient_id", "reviewer"}
    if not required <= set(frame):
        raise ValueError(f"Annotations require columns {sorted(required)}")
    values = pd.to_numeric(frame[target], errors="raise")
    if not values.isin([-1, 0, 1]).all():
        raise ValueError(f"{target} must be -1 (unknown), 0 or 1")
    for col in ("patient_id", "reviewer"):
        checked = frame.loc[values >= 0, col] if col == "reviewer" else frame[col]
        if checked.isna().any() or checked.str.strip().eq("").any():
            raise ValueError(f"Annotations require nonempty {col} for labeled rows")
    frame["label"] = values.astype(int)
    frame["label_source"] = "pathologist_reviewed"
    return frame[KEYS + ["patient_id", "label", "label_source"]]


def read_weak_labels(manifest, registration_dir):
    frames = []
    for pair in read_pairs(manifest):
        folder = Path(registration_dir) / pair["pair_id"]
        path = folder / "p53_cell_measurements.csv"
        if not path.exists():
            continue  # New pairs can be added before registration is complete.
        meta = json.loads((folder / "registration.json").read_text())
        if meta["pair_signature"] != pair_signature(pair):
            raise ValueError(f"{pair['pair_id']}: source pair changed; rerun registration and extraction")
        extraction = json.loads((folder / "extraction.json").read_text())
        review = json.loads((folder / "review.json").read_text())
        region_hash = hashlib.sha256((folder / "region_review.csv").read_bytes()).hexdigest()
        if (extraction["registration_id"] != meta["registration_id"] or extraction["review"] != review
                or extraction["region_review_sha256"] != region_hash):
            raise ValueError(f"{pair['pair_id']}: registration/review changed; rerun extract before training")
        frame = pd.read_csv(path, dtype={"slide": str, "region": str, "patient_id": str})
        validate_keys(frame, str(path))
        if not frame.registration_id.eq(meta["registration_id"]).all():
            raise ValueError("Measurements belong to a previous registration")
        if not frame.patient_id.eq(str(pair["patient_id"])).all() or set(frame.slide) != {Path(pair["he_slide"]).stem}:
            raise ValueError("Manifest patient/slide identity differs from extracted measurements")
        frame["label"] = frame.p53_label.astype(int)
        frame["label_source"] = "p53_staining_weak_label"
        frames.append(frame[KEYS + ["patient_id", "label", "label_source"]])
    if not frames:
        raise ValueError("No p53 measurements: register and extract the paired slides first")
    result = pd.concat(frames, ignore_index=True)
    validate_keys(result, "p53 measurements")
    return result


def training_table(features, labels):
    if labels.groupby("slide").patient_id.nunique().gt(1).any():
        raise ValueError("Every slide must have exactly one patient_id")
    labels = labels[labels.label >= 0].copy()
    if labels.empty:
        raise ValueError("No usable labels. Review registration/stain thresholds or supply pathologist annotations")
    joined = labels.merge(features, on=KEYS, how="left", validate="one_to_one", indicator=True)
    missing = int(joined._merge.eq("left_only").sum())
    if missing:
        print(f"Excluding {missing} labeled cells absent from Script03 features (e.g. area-filtered nuclei).", flush=True)
    joined = joined[joined._merge.eq("both")].drop(columns="_merge").reset_index(drop=True)
    if joined.empty or set(joined.label) != {0, 1}:
        raise ValueError("Training requires H&E features and labels for both positive and negative cells")
    return joined


def make_model(seed, trees, jobs):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", RandomForestClassifier(n_estimators=trees, max_depth=16, min_samples_leaf=5,
                                               max_features="sqrt", random_state=seed, n_jobs=jobs)),
    ])


def balanced_weights(frame):
    # Equal total weight for each class and each patient represented in that class.
    sizes = frame.groupby(["patient_id", "label"])["label"].transform("size").to_numpy()
    patients = frame.groupby("label").patient_id.nunique()
    weights = 1 / (sizes * frame.label.map(patients).to_numpy())
    return weights / weights.mean()


def feature_matrix(frame, columns):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Missing trained H&E features: {sorted(missing)}")
    return frame[columns].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)


def metrics(y, probability, threshold=.5):
    y, probability = np.asarray(y), np.asarray(probability)
    pred = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"cells": len(y), "positives": int(y.sum()), "threshold": threshold,
            "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "precision": float(precision[1]), "sensitivity": float(recall[1]) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None, "f1": float(f1[1]),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
            "roc_auc": float(roc_auc_score(y, probability)) if both else None,
            "average_precision": float(average_precision_score(y, probability)) if both else None}


def grouped_report(frame, probability, threshold):
    report = {"pooled_cells": metrics(frame.label, probability, threshold), "per_patient": {}}
    for patient, indices in frame.groupby("patient_id").indices.items():
        report["per_patient"][str(patient)] = metrics(frame.iloc[indices].label, probability[indices], threshold)
    values = [v["balanced_accuracy"] for v in report["per_patient"].values() if v["balanced_accuracy"] is not None]
    report["macro_patient_balanced_accuracy"] = float(np.mean(values)) if values else None
    return report


def train(args):
    features = read_features(args.features)
    if args.target == "tumor" and args.labels is None:
        raise ValueError("Tumor training requires --labels with reviewed tumor=0/1 annotations; p53 absence is not a benign label")
    labels = read_annotations(args.labels, args.target) if args.labels else read_weak_labels(args.manifest, args.registration_dir)
    frame = training_table(features, labels)
    columns = sorted(HE_FEATURES.intersection(features.columns))
    x = feature_matrix(frame, columns)
    patients = frame.patient_id.nunique()
    report = {"target": args.target, "label_sources": sorted(frame.label_source.unique()),
              "training_patients": int(patients), "training_cells": len(frame), "feature_count": len(columns),
              "note": "Staining-derived labels evaluate a p53 proxy, not independently established tumor accuracy."}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame[KEYS + ["patient_id", "label", "label_source"]].to_csv(out / "training_labels.csv", index=False)
    oof = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=int)
    folds = []
    if patients >= 2:
        splitter = GroupKFold(n_splits=min(args.folds, patients))
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, frame.label, frame.patient_id)):
            fold_info = {"fold": fold, "train_patients": sorted(frame.iloc[train_idx].patient_id.unique()),
                         "validation_patients": sorted(frame.iloc[test_idx].patient_id.unique())}
            if frame.iloc[train_idx].label.nunique() < 2:
                fold_info["status"] = "not_evaluable: training fold has only one class"
            else:
                model = make_model(args.seed, args.trees, args.jobs)
                model.fit(x.iloc[train_idx], frame.iloc[train_idx].label,
                          classifier__sample_weight=balanced_weights(frame.iloc[train_idx]))
                oof[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]
                fold_ids[test_idx] = fold
                fold_info["status"] = "evaluated"
            folds.append(fold_info)
    report["folds"] = folds
    complete = np.isfinite(oof).all()
    report["validation_status"] = "patient_grouped_cross_validation" if complete else "insufficient_patients_or_class_coverage"
    report["evaluated_cells"] = int(np.isfinite(oof).sum())
    # Do not pool only easy/evaluable folds into a misleading headline metric.
    if complete:
        report["cross_validation"] = grouped_report(frame, oof, args.threshold)
    predictions = frame[KEYS + ["patient_id", "label", "label_source"]].copy()
    predictions["fold"] = fold_ids
    predictions["oof_probability"] = oof
    predictions.to_csv(out / "cross_validation_predictions.csv", index=False)
    final_model = make_model(args.seed, args.trees, args.jobs)
    final_model.fit(x, frame.label, classifier__sample_weight=balanced_weights(frame))
    bundle = {"schema_version": 1, "model": final_model, "features": columns, "target": args.target,
              "threshold": args.threshold, "training_patients": sorted(frame.patient_id.unique()),
              "training_slides": sorted(frame.slide.unique()), "label_sources": report["label_sources"],
              "seed": args.seed, "validation_status": report["validation_status"]}
    joblib.dump(bundle, out / "model.joblib")
    write_json(out / "training_report.json", report)
    write_json(out / "model_metadata.json", {k: v for k, v in bundle.items() if k != "model"})
    print(f"Saved {args.target} model from {patients} patients / {len(frame)} cells. Validation: {report['validation_status']}")


def predict(args):
    bundle = joblib.load(args.model)
    features = read_features(args.features)
    if args.slide:
        features = features[features.slide.isin(args.slide)].copy()
    if features.empty:
        raise ValueError("No cells for the requested slide(s)")
    x = feature_matrix(features, bundle["features"])
    probability = bundle["model"].predict_proba(x)[:, 1]
    output = features[KEYS].copy()
    output["target"] = bundle["target"]
    output["probability"] = probability
    output["prediction"] = (probability >= bundle["threshold"]).astype(int)
    output["seen_training_slide"] = output.slide.isin(bundle["training_slides"])
    output["missing_feature_fraction"] = x.isna().mean(axis=1).to_numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = output.groupby("slide").agg(cells=("prediction", "size"), positive_cells=("prediction", "sum"),
                                           positive_fraction=("prediction", "mean"), mean_score=("probability", "mean"))
    summary.to_csv(Path(args.output).with_suffix(".slides.csv"))
    print(f"Wrote {len(output)} H&E-only predictions -> {args.output}. These are predictions, not validation labels.")


def evaluate(args):
    bundle = joblib.load(args.model)
    annotations = read_annotations(args.labels, bundle["target"])
    annotations = annotations[annotations.label >= 0].copy()
    if annotations.empty or annotations.groupby("slide").patient_id.nunique().gt(1).any():
        raise ValueError("Evaluation needs labels with one patient_id per slide")
    if set(annotations.patient_id) & set(bundle["training_patients"]) or set(annotations.slide) & set(bundle["training_slides"]):
        raise ValueError("Independent evaluation cannot include training patients or training slides")
    frame = annotations.merge(read_features(args.features), on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not frame._merge.eq("both").all():
        raise ValueError("Some evaluation annotations have no H&E feature row; reconcile keys explicitly")
    probability = bundle["model"].predict_proba(feature_matrix(frame, bundle["features"]))[:, 1]
    report = {"target": bundle["target"], "evaluation": "independent_reviewed_labels",
              **grouped_report(frame, probability, bundle["threshold"])}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"Independent evaluation of {len(frame)} cells saved -> {args.output}")


def init_manifest(args):
    output = Path(args.manifest).resolve()
    if output.exists():
        raise ValueError(f"Manifest already exists: {output}; edit it to add pairs")
    p53_dir = Path(args.p53_dir).resolve()
    paths = sorted(p for p in p53_dir.iterdir() if p.suffix.lower() == ".ndpi")
    if not paths:
        raise ValueError(f"No NDPI slides in {p53_dir}")
    pairs = []
    for i, path in enumerate(paths, 1):
        try:
            value = str(path.relative_to(output.parent))
        except ValueError:
            value = str(path)
        pairs.append({"pair_id": f"pair_{i:03d}", "patient_id": "SET_PATIENT_ID", "he_slide": None,
                      "p53_slide": value.replace("\\", "/"), "section_relationship": "unknown",
                      "he_mpp": None, "p53_mpp": None, "landmarks": None, "qc_landmarks": None})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"pairs": pairs})
    print(f"Created {output}; fill he_slide, patient_id and section_relationship before registration")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Inventory p53 slides and write a manifest template")
    init.add_argument("--p53-dir", type=Path, default=Path("p53"))
    init.add_argument("--manifest", type=Path, default=Path("p53_pairs.json"))
    for command in ("register", "extract"):
        cmd = commands.add_parser(command)
        cmd.add_argument("--manifest", type=Path, default=Path("p53_pairs.json"))
        cmd.add_argument("--pair", help="Process one pair_id (useful for Slurm jobs)")
        cmd.add_argument("--output", type=Path, default=Path("p53_registration"))
        cmd.add_argument("--threads", type=int, default=4)
        if command == "register":
            cmd.add_argument("--size", type=int, default=2048, help="Maximum registration thumbnail dimension")
            cmd.add_argument("--iterations", type=int, default=100)
            cmd.add_argument("--deformable", action="store_true", help="Refine affine registration with a B-spline")
        else:
            cmd.add_argument("--cellvit-root", type=Path, default=Path("cellvit_output"))
            cmd.add_argument("--features", type=Path, default=Path("feature_data"), help="Raw H&E feature directory for joined output")
            cmd.add_argument("--extract-he-features", action="store_true", help="Run Script03 extraction for each paired H&E slide first")
            cmd.add_argument("--stain-ref-dir", type=Path, default=Path("stain_references"))
            cmd.add_argument("--allow-missing-stain-ref", action="store_true")
            cmd.add_argument("--tile-size", type=int, default=512)
            cmd.add_argument("--max-error-um", type=float, default=2.)
            cmd.add_argument("--allow-serial-weak-labels", action="store_true",
                             help="Permit reviewed serial-section staining proxies; these are not exact cell matches")
    cmd = commands.add_parser("train")
    cmd.add_argument("--features", type=Path, default=Path("feature_data"))
    cmd.add_argument("--manifest", type=Path, default=Path("p53_pairs.json"))
    cmd.add_argument("--registration-dir", type=Path, default=Path("p53_registration"))
    cmd.add_argument("--labels", type=Path, help="Reviewed annotation CSV; overrides staining-derived labels")
    cmd.add_argument("--target", choices=["tumor", "p53_positive"], default="tumor")
    cmd.add_argument("--output", type=Path, default=Path("p53_supervised"))
    cmd.add_argument("--folds", type=int, default=5)
    cmd.add_argument("--trees", type=int, default=300)
    cmd.add_argument("--jobs", type=int, default=4)
    cmd.add_argument("--seed", type=int, default=42)
    cmd.add_argument("--threshold", type=float, default=.5)
    for command in ("predict", "evaluate"):
        cmd = commands.add_parser(command)
        cmd.add_argument("--model", type=Path, default=Path("p53_supervised/model.joblib"))
        cmd.add_argument("--features", type=Path, default=Path("feature_data"))
        if command == "predict":
            cmd.add_argument("--slide", action="append", help="Exact H&E slide stem; repeat for multiple slides")
            cmd.add_argument("--output", type=Path, default=Path("p53_supervised/predictions.csv"))
        else:
            cmd.add_argument("--labels", type=Path, required=True)
            cmd.add_argument("--output", type=Path, default=Path("p53_supervised/independent_evaluation.json"))
    return root


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        for key in ("threads", "size", "iterations", "tile_size", "max_error_um", "trees"):
            if hasattr(args, key) and (not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0):
                raise ValueError(f"--{key.replace('_', '-')} must be positive")
        if args.command == "train" and (args.folds < 2 or not 0 < args.threshold < 1 or args.jobs == 0):
            raise ValueError("Require --folds >= 2, 0 < --threshold < 1 and --jobs != 0")
        if args.command in ("register", "extract"):
            import SimpleITK as sitk
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
            for pair in select_pairs(args.manifest, args.pair):
                if pair["patient_id"] == "SET_PATIENT_ID":
                    raise ValueError("Replace SET_PATIENT_ID with the actual patient identifier")
                if args.command == "register":
                    register_pair(pair, args.output, args.size, args.iterations, args.deformable)
                else:
                    if not pair.get("he_slide"):
                        raise ValueError(f"Set he_slide for {pair['pair_id']} in the manifest")
                    if args.extract_he_features:
                        meta = json.loads((args.output / pair["pair_id"] / "registration.json").read_text())
                        if not np.isclose(*meta["he_mpp"], rtol=.01):
                            raise ValueError("Script03 assumes square pixels; resample anisotropic H&E before extracting features")
                        command = [sys.executable, str(Path(__file__).with_name("03_feature_extraction.py")),
                                   "--stage", "extract", "--slide-name", Path(pair["he_slide"]).stem,
                                   "--cellvit-root", str(args.cellvit_root), "--out-dir", str(args.features),
                                   "--stain-ref-dir", str(args.stain_ref_dir), "--mpp", str(meta["he_mpp"][0])]
                        if args.allow_missing_stain_ref:
                            command.append("--allow-missing-stain-ref")
                        subprocess.run(command, check=True)
                    he_features = read_features(args.features)
                    he_features = he_features[he_features.slide.eq(Path(pair["he_slide"]).stem)]
                    if he_features.empty:
                        raise ValueError("No H&E feature rows for this pair; add --extract-he-features or provide --features")
                    measured = extract_pair(pair, args.output, args.cellvit_root, args.tile_size, args.allow_serial_weak_labels, args.max_error_um)
                    combined = he_features.merge(measured, on=KEYS, how="left", validate="one_to_one", indicator=True)
                    if not combined._merge.eq("both").all():
                        raise ValueError("Some H&E feature cells lack p53 measurements; check input mask/coordinate versions")
                    combined.drop(columns="_merge").to_csv(args.output / pair["pair_id"] / "co_registered_cell_features.csv", index=False)
        else:
            {"init": init_manifest, "train": train, "predict": predict, "evaluate": evaluate}[args.command](args)
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
