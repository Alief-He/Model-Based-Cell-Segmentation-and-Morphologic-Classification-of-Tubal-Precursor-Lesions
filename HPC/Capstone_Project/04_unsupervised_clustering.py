"""GMM clustering of ROIs represented by selected per-cell features.

Examples (run from the project directory)::

    python 04_unsupervised_clustering.py train --input-dir feature_data
    python 04_unsupervised_clustering.py train --input-dir feature_data \
        --feature-table feature_data_cleaned/feature_group_selection_table.csv
    python 04_unsupervised_clustering.py predict --input-dir new_roi.csv \
        --model clustering_results/gmm_model.joblib --output-dir predictions
    python 04_unsupervised_clustering.py evaluate --input-dir held_out_features \
        --model clustering_results/gmm_model.joblib --output-dir evaluation

Input: one cell per row. `slide` + `region` identifies an ROI. Without `region`,
each CSV is treated as one ROI. An anonymous single-ROI CSV needs only the saved
feature columns. Train on raw Script03 measurements by default: the feature
selection method in variable_cleaning is fitted on training cells only.

--feature-table reuses an existing selected-feature list on raw measurements.
If that list was selected using validation data, evaluation is exploratory.
--input-space cleaned also accepts already-scaled variable_cleaning outputs,
but future inputs MUST use that SAME upstream transformation; this script cannot
recover its scaling parameters. Prefer raw input for reproducible deployment.

Labels are parsed only for splitting, post-fit cluster naming and evaluation.
Train/evaluate skip ROIs without a recognized label and save skipped_unlabeled_rois.csv.
Predict accepts unlabeled ROIs. Conflicting labels still raise an error.
predict_rois() never reads labels; filenames/identifiers are not model features.
Three clusters do not guarantee three biological classes. Component probabilities
are not calibrated diagnostic probabilities. Only load trusted joblib models.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

import variable_cleaning as cleaning


CLASSES = ("STIC", "HGSC", "NFT")
LABEL_PATTERN = re.compile(r"(?<![A-Z0-9])(STIC|HGSC|NFT)(?![A-Z0-9])", re.I)
EXTRA_METADATA = {
    "patient", "patient_id", "subject", "subject_id", "roi", "roi_id",
    "true_label", "predicted_label", "split", "type_prob",
}
SCHEMA_VERSION = 1


def warn(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=2)


def csv_paths(path: Path, input_space: str) -> list[Path]:
    if path.is_file():
        return [path]
    suffix = "_cell_features_cleaned.csv" if input_space == "cleaned" else "_cell_features.csv"
    paths = sorted(p for p in path.glob(f"*{suffix}") if p.name != f"all{suffix}")
    if paths:
        return paths
    aggregate = path / f"all{suffix}"
    if aggregate.is_file():
        return [aggregate]
    # Anonymous ROI CSV files are also accepted. Exclude generated reports.
    paths = sorted(p for p in path.glob("*.csv") if not any(
        token in p.stem for token in ("matrix", "selection", "predictions", "manifest", "confusion")
    ))
    if input_space == "raw":
        paths = [p for p in paths if not p.stem.endswith("_cleaned")]
    if not paths:
        raise ValueError(f"No feature CSV files found in {path}")
    return paths


def read_cells(path: Path, input_space: str, group_column: str = "slide") -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, records, seen = [], [], set()
    for csv_path in csv_paths(path, input_space):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise ValueError(f"Empty feature CSV: {csv_path}")
        if any(str(c).startswith("__") for c in frame.columns):
            raise ValueError("Input columns beginning with '__' are reserved.")
        if "region" not in frame:
            frame["region"] = csv_path.stem
        if "slide" not in frame:
            frame["slide"] = csv_path.stem
        if frame[["slide", "region"]].isna().any().any():
            raise ValueError(f"Missing slide/region identifiers in {csv_path}")
        frame["slide"] = frame["slide"].astype(str)
        frame["region"] = frame["region"].astype(str)
        for (slide, region), cells in frame.groupby(["slide", "region"], sort=True):
            key = (slide, region)
            if key in seen:
                raise ValueError(f"Duplicate ROI {key}; do not mix aggregate and per-slide CSVs.")
            seen.add(key)
            if group_column not in cells or cells[group_column].isna().any():
                raise ValueError(f"Missing grouping column {group_column!r} in {csv_path}")
            groups = cells[group_column].astype(str).unique()
            if len(groups) != 1:
                raise ValueError(f"ROI {key} belongs to multiple {group_column} groups.")
            index = len(records)
            records.append(dict(roi_index=index, slide=slide, region=region,
                                source_file=csv_path.name, group=groups[0], n_cells=len(cells)))
            cells = cells.copy()
            cells["__roi_index"] = index
            frames.append(cells)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(records).set_index("roi_index")


def labels_from_names(rois: pd.DataFrame) -> pd.Series:
    labels = []
    for row in rois.itertuples():
        matches = set(LABEL_PATTERN.findall(f"{row.region} {row.source_file}".upper()))
        if len(matches) > 1:
            raise ValueError(
                f"Expected one STIC/HGSC/NFT label in ROI/file name, got {sorted(matches)}: "
                f"{row.region} / {row.source_file}. Resolve conflicting labels."
            )
        labels.append(matches.pop() if matches else pd.NA)
    return pd.Series(labels, index=rois.index, name="true_label")


def keep_labeled_rois(
    cells: pd.DataFrame, rois: pd.DataFrame, output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude unlabeled ROIs and all their cells before training/evaluation."""
    rois = rois.copy()
    rois["true_label"] = labels_from_names(rois)
    labeled = rois.true_label.notna()
    skipped = rois.loc[~labeled].copy()
    skipped["reason"] = "No STIC/HGSC/NFT label in ROI or file name"
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped.to_csv(output_dir / "skipped_unlabeled_rois.csv", index_label="roi_index")
    if not skipped.empty:
        print(f"Skipped {len(skipped)} unlabeled ROI(s), {int(skipped.n_cells.sum())} cells; "
              f"details: {output_dir / 'skipped_unlabeled_rois.csv'}", flush=True)
    if not labeled.any():
        raise ValueError("No labeled ROIs remain after excluding ROIs without STIC/HGSC/NFT labels.")
    rois = rois.loc[labeled].copy()
    return cells.loc[cells["__roi_index"].isin(rois.index)].copy(), rois


def split_rois(rois: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    counts = rois.true_label.value_counts().reindex(CLASSES, fill_value=0)
    if (counts < 2).any():
        raise ValueError(f"Need at least two ROIs per class for train/validation: {counts.to_dict()}")
    rng = np.random.default_rng(args.seed)
    if args.split_unit == "roi":
        warn("ROI split can share patients/slides across train and validation; results are exploratory.")
        validation = []
        for label in CLASSES:
            indices = rois.index[rois.true_label == label].to_numpy().copy()
            rng.shuffle(indices)
            n_val = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * args.validation_fraction))))
            validation.extend(indices[:n_val])
        validation = np.array(validation)
    else:
        if rois.group.nunique() < 2:
            raise ValueError(
                "Group split needs multiple independent groups. Supply more slides/patients, "
                "or explicitly use --split-unit roi for an exploratory single-slide run."
            )
        splitter = GroupShuffleSplit(n_splits=500, test_size=args.validation_fraction, random_state=args.seed)
        best = None
        for train_pos, val_pos in splitter.split(rois, groups=rois.group):
            train_counts = rois.iloc[train_pos].true_label.value_counts().reindex(CLASSES, fill_value=0)
            val_counts = rois.iloc[val_pos].true_label.value_counts().reindex(CLASSES, fill_value=0)
            if (train_counts == 0).any() or (val_counts == 0).any():
                continue
            score = float(np.abs(val_counts / counts - args.validation_fraction).mean())
            if best is None or score < best[0]:
                best = (score, rois.index[val_pos].to_numpy())
        if best is None:
            raise ValueError(
                "Could not find a group-disjoint split containing all three classes on both sides. "
                "Check class/group coverage, change --validation-fraction, or supply more groups."
            )
        validation = best[1]
    split = pd.Series("unused_training_group", index=rois.index, name="split")
    split.loc[validation] = "validation"
    available = rois.loc[split != "validation"]
    quota = int(available.true_label.value_counts().reindex(CLASSES).min())
    if args.max_train_rois_per_class:
        quota = min(quota, args.max_train_rois_per_class)
    for label in CLASSES:
        indices = available.index[available.true_label == label].to_numpy()
        split.loc[rng.choice(indices, quota, replace=False)] = "train"
    if quota < 5:
        warn(f"Only {quota} training ROI(s) per class; use results to check the pipeline, not model quality.")
    return split


def selected_columns(table_path: Path) -> list[str]:
    table = pd.read_csv(table_path)
    if "feature" in table:
        values = table.feature.dropna().astype(str).tolist()
    elif "selected_features" in table:
        values = [name.strip() for text in table.selected_features.dropna().astype(str)
                  for name in text.split(";") if name.strip()]
    else:
        raise ValueError("Feature table needs a 'feature' or 'selected_features' column.")
    columns = list(dict.fromkeys(values))
    if not columns:
        raise ValueError("Selected-feature table is empty.")
    return columns


def numeric_features(cells: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(cells.columns))
    if missing:
        raise ValueError(f"Missing required cell features: {missing}")
    return cells[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def fit_feature_selection(cells: pd.DataFrame, args: argparse.Namespace) -> tuple[list[str], dict]:
    # A fixed cap per training ROI prevents large ROIs dominating feature selection.
    rng = np.random.default_rng(args.seed)
    sampled = []
    for _, group in cells.groupby("__roi_index", sort=True):
        sampled.append(group.loc[rng.choice(group.index, min(len(group), args.selection_cells_per_roi), replace=False)])
    training = pd.concat(sampled, ignore_index=True)
    forbidden = EXTRA_METADATA | {args.group_column.lower()}
    candidates = [c for c in cleaning.feature_columns(training)
                  if c.lower() not in forbidden and not c.startswith("__")]
    if args.feature_table:
        columns = selected_columns(args.feature_table)
        invalid = sorted(set(columns) - set(candidates))
        if invalid:
            raise ValueError(f"Selected columns are absent, nonnumeric or excluded metadata: {invalid}")
        warn("Using an external feature list. If selected on all data, validation is exploratory.")
        return columns, {"method": "external_selected_features", "table": str(args.feature_table),
                         "selected_features": columns}
    if args.input_space == "cleaned":
        raise ValueError("Cleaned input requires --feature-table to exclude passthrough metadata.")
    _, summary = cleaning.clean_variables(
        numeric_features(training, candidates), args.max_features,
        args.feature_groups, args.features_per_group,
    )
    summary["method"] = "variable_cleaning_fitted_on_training_cells_only"
    return summary["selected_features"], summary


def aggregate_rois(cells: pd.DataFrame, rois: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = numeric_features(cells, columns)
    numeric["__roi_index"] = cells["__roi_index"].to_numpy()
    grouped = numeric.groupby("__roi_index", sort=True)
    medians = grouped[columns].median().add_suffix("__median")
    iqrs = (grouped[columns].quantile(.75) - grouped[columns].quantile(.25)).add_suffix("__iqr")
    result = pd.concat([medians, iqrs], axis=1).reindex(rois.index)
    if result.isna().all(axis=1).any():
        bad = result.index[result.isna().all(axis=1)].tolist()
        raise ValueError(f"ROIs have no usable selected measurements: {bad}")
    return result


def fit_preprocessing(features: pd.DataFrame, pca_components: int) -> tuple[np.ndarray, dict]:
    usable = features.columns[features.nunique(dropna=True) > 1].tolist()
    if not usable:
        raise ValueError("No varying ROI-level features remain in training data.")
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    values = scaler.fit_transform(imputer.fit_transform(features[usable]))
    pca = None
    if pca_components:
        components = min(pca_components, len(features) - 1, len(usable))
        pca = PCA(n_components=components, svd_solver="full")
        values = pca.fit_transform(values)
    if np.unique(values, axis=0).shape[0] < 3:
        raise ValueError("At least three distinct training ROI vectors are required.")
    return values, dict(roi_feature_columns=usable, imputer=imputer, scaler=scaler, pca=pca)


def transform_rois(features: pd.DataFrame, model: dict) -> np.ndarray:
    values = model["scaler"].transform(model["imputer"].transform(features[model["roi_feature_columns"]]))
    if model["pca"] is not None:
        values = model["pca"].transform(values)
    return values


def predict_rois(cells: pd.DataFrame, rois: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Predict using cell measurements only; ROI names are copied for output identification."""
    features = aggregate_rois(cells, rois, model["cell_feature_columns"])
    values = transform_rois(features, model)
    probabilities = model["gmm"].predict_proba(values)
    clusters = probabilities.argmax(axis=1)
    result = rois[["slide", "region", "source_file", "n_cells"]].copy()
    result["cluster"] = clusters
    result["predicted_label"] = [model["cluster_to_label"][int(c)] for c in clusters]
    for cluster, label in model["cluster_to_label"].items():
        result[f"prob_{label}"] = probabilities[:, cluster]
    result["max_component_probability"] = probabilities.max(axis=1)
    return result


def metrics_for(predictions: pd.DataFrame) -> dict:
    truth, predicted = predictions.true_label, predictions.predicted_label
    return {
        "n_rois": len(predictions),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=list(CLASSES), average="macro", zero_division=0)),
        "adjusted_rand_index": float(adjusted_rand_score(truth, predictions.cluster)),
        "classification_report": classification_report(
            truth, predicted, labels=list(CLASSES), output_dict=True, zero_division=0),
        "confusion_matrix_label_order": list(CLASSES),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(CLASSES)).tolist(),
    }


def write_json(path: Path, data: dict) -> None:
    # variable_cleaning includes missing partner distances in feature-selection metadata.
    text = json.dumps(data, indent=2, ensure_ascii=False, default=lambda x: x.item())
    data = json.loads(text, parse_constant=lambda _: None)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_metrics(predictions: pd.DataFrame, output: Path, stem: str) -> dict:
    metrics = metrics_for(predictions)
    write_json(output / f"{stem}_metrics.json", metrics)
    pd.DataFrame(metrics["classification_report"]).T.to_csv(output / f"{stem}_classification_report.csv")
    matrix = np.array(metrics["confusion_matrix"])
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
        output / f"{stem}_confusion_matrix.csv", index_label="true_label")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted label", ylabel="True label", title=f"GMM: {stem}")
    fig.tight_layout()
    fig.savefig(output / f"{stem}_confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"{stem}: ROIs={len(predictions)}, accuracy={metrics['accuracy']:.4f}, "
          f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, macro_F1={metrics['macro_f1']:.4f}, "
          f"STIC_recall={metrics['classification_report']['STIC']['recall']:.4f}")
    return metrics


def train(args: argparse.Namespace) -> None:
    cells, rois = read_cells(args.input_dir, args.input_space, args.group_column)
    cells, rois = keep_labeled_rois(cells, rois, args.output_dir)
    rois["split"] = split_rois(rois, args)
    train_ids = rois.index[rois.split == "train"]
    training_cells = cells.loc[cells["__roi_index"].isin(train_ids)]
    columns, selection = fit_feature_selection(training_cells, args)
    features = aggregate_rois(cells, rois, columns)
    values, preprocessing = fit_preprocessing(features.loc[train_ids], args.pca_components)
    gmm = GaussianMixture(n_components=3, covariance_type=args.covariance_type,
                          reg_covar=args.reg_covar, n_init=args.n_init,
                          max_iter=args.max_iter, random_state=args.seed)
    gmm.fit(values)  # No true labels are passed to the clustering estimator.
    if not gmm.converged_:
        raise ValueError("GMM did not converge; increase --max-iter or --reg-covar.")
    clusters = gmm.predict(values)
    if len(np.unique(clusters)) != 3:
        raise ValueError("GMM has fewer than three occupied training clusters; inspect features or regularization.")
    contingency = np.zeros((3, 3), dtype=int)
    for cluster, label in zip(clusters, rois.loc[train_ids, "true_label"]):
        contingency[cluster, CLASSES.index(label)] += 1
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(row): CLASSES[int(col)] for row, col in zip(rows, cols)}
    if any(contingency[row, col] == 0 for row, col in zip(rows, cols)):
        warn("One-to-one mapping includes a class with zero supporting training ROIs; inspect cluster_label_counts.csv.")
    limitations = ["Cluster names use training labels; probabilities are not calibrated diagnostic probabilities."]
    if args.feature_table:
        limitations.append("External feature selection may have seen held-out data; verify its provenance.")
    if args.input_space == "cleaned":
        limitations.append("Upstream scaling is external and may include held-out data; reuse exactly that scaling at inference.")
        warn(limitations[-1])
    if args.split_unit == "roi":
        limitations.append("ROI split may share patients/slides; not an independent patient/slide evaluation.")
    quota = int((rois.split == "train").sum() // 3)
    if quota < 5:
        limitations.append(f"Very small training set: {quota} ROI(s) per class; exploratory results only.")
    model = dict(schema_version=SCHEMA_VERSION, sklearn_version=sklearn.__version__,
                 input_space=args.input_space, cell_feature_columns=columns,
                 cluster_to_label=mapping, gmm=gmm, **preprocessing)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "gmm_model.joblib")
    rois.to_csv(output / "split_manifest.csv", index_label="roi_index")
    pd.DataFrame(contingency, columns=CLASSES).to_csv(output / "cluster_label_counts.csv", index_label="cluster")
    pd.DataFrame({"feature": columns}).to_csv(output / "selected_cell_features.csv", index=False)
    write_json(output / "feature_selection.json", selection)
    predictions = predict_rois(cells, rois, model)
    predictions["split"] = rois.split
    predictions["true_label"] = rois.true_label
    predictions["matches_label"] = predictions.predicted_label == predictions.true_label
    predictions.to_csv(output / "roi_predictions.csv", index_label="roi_index")
    for subset in ("train", "validation"):
        selected = predictions.loc[predictions.split == subset]
        selected.to_csv(output / f"{subset}_predictions.csv", index_label="roi_index")
        save_metrics(selected, output, subset)
    write_json(output / "training_summary.json", {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "class_counts_by_split": pd.crosstab(rois.split, rois.true_label).to_dict(orient="index"),
        "training_rois_per_class": quota, "selected_cell_features": columns,
        "model_dimensions": values.shape[1], "cluster_to_label": mapping,
        "converged": bool(gmm.converged_), "iterations": int(gmm.n_iter_),
        "limitations": limitations,
    })
    print(f"Saved model and reports to {output.resolve()}")


def infer(args: argparse.Namespace) -> None:
    model = joblib.load(args.model)
    if model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported model file schema.")
    if model["sklearn_version"] != sklearn.__version__:
        warn(f"Model used sklearn {model['sklearn_version']}; current version is {sklearn.__version__}.")
    if args.input_space != model["input_space"]:
        raise ValueError(f"Model expects --input-space {model['input_space']}; do not mix raw and cleaned measurements.")
    cells, rois = read_cells(args.input_dir, args.input_space)
    if args.command == "evaluate":
        cells, rois = keep_labeled_rois(cells, rois, args.output_dir)
    # predict accepts unlabeled inputs; labels never enter predict_rois's feature matrix.
    predictions = predict_rois(cells, rois, model)
    if args.command == "evaluate":
        predictions["true_label"] = rois.true_label
        predictions["matches_label"] = predictions.predicted_label == predictions.true_label
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "roi_predictions.csv", index_label="roi_index")
    if args.command == "evaluate":
        save_metrics(predictions, args.output_dir, "evaluation")
    print(predictions[["region", "predicted_label", "max_component_probability"]].to_string(index=False))
    print(f"Saved predictions to {(args.output_dir / 'roi_predictions.csv').resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "predict", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--input-dir", type=Path, required=True, help="Feature CSV directory, or a single CSV.")
        sub.add_argument("--output-dir", type=Path, default=Path("clustering_results") if command == "train" else Path(command + "_results"))
        sub.add_argument("--input-space", choices=("raw", "cleaned"), default="raw",
                         help="Measurement space; inference must match training. Default: raw.")
        if command != "train":
            sub.add_argument("--model", type=Path, required=True)
            continue
        sub.add_argument("--feature-table", type=Path, help="Optional existing variable_cleaning selection table.")
        sub.add_argument("--split-unit", choices=("group", "roi"), default="group")
        sub.add_argument("--group-column", default="slide", help="Use patient_id when available; default: slide.")
        sub.add_argument("--validation-fraction", type=float, default=.3)
        sub.add_argument("--max-train-rois-per-class", type=int, default=0, help="0 uses the minority training class count.")
        sub.add_argument("--selection-cells-per-roi", type=int, default=500)
        sub.add_argument("--max-features", type=int, default=40)
        sub.add_argument("--feature-groups", type=int, default=20)
        sub.add_argument("--features-per-group", type=int, default=2)
        sub.add_argument("--pca-components", type=int, default=10, help="Capped by training size; 0 disables PCA.")
        sub.add_argument("--covariance-type", choices=("diag", "tied", "full", "spherical"), default="diag")
        sub.add_argument("--reg-covar", type=float, default=1e-4)
        sub.add_argument("--n-init", type=int, default=20)
        sub.add_argument("--max-iter", type=int, default=500)
        sub.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "train":
        if not 0 < args.validation_fraction < 1:
            parser.error("--validation-fraction must be between 0 and 1.")
        if not 0 < args.max_features < 50:
            parser.error("--max-features must be between 1 and 49.")
        if min(args.selection_cells_per_roi, args.feature_groups, args.features_per_group, args.n_init, args.max_iter) < 1:
            parser.error("Feature selection limits, --n-init and --max-iter must be positive.")
        if min(args.max_train_rois_per_class, args.pca_components, args.seed) < 0 or not np.isfinite(args.reg_covar) or args.reg_covar <= 0:
            parser.error("Counts/seed must be nonnegative and --reg-covar must be finite and positive.")
        if args.group_column.lower() in {"region", "roi", "roi_id", "label", "true_label", "class"}:
            parser.error("Use a patient/slide identifier as --group-column; use --split-unit roi for ROI splitting.")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.command == "train":
            train(args)
        else:
            infer(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
