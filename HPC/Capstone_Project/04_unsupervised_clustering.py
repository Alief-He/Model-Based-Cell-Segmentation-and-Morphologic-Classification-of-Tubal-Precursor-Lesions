"""Cell-level GMM: 40 selected features, 5 ROIs per STIC/HGSC/NFT, 20 components.

    python 04_unsupervised_clustering.py --input-dir cellvit_output/feature_data
    python 04_unsupervised_clustering.py --input-dir feature_data --rois-per-class 5

Read the 40 feature names from Script03's feature_group_selection_table.csv,
then use only those columns from the raw per-cell measurements. By default the
table is in the input folder's sibling <input-folder>_cleaned directory;
--feature-table can specify another table. Never fall back to all features.
ROI diagnoses are used ONLY to select ROIs. All cells in those ROIs enter one
pooled GMM. There is no ROI aggregation or cluster-to-diagnosis mapping.
Median imputation and standardization are fitted on these selected cells only.
PCA is for display only; GMM sees the 40 standardized selected features.
Sample the same number of ROIs from each class using --rois-per-class (default 5).
Cell identifiers are not validated; all feature rows in selected ROIs are used.

The only output is cell_clusters_pca_3d.html, showing all selected cells in
PC1/PC2/PC3 space, colored by their GMM cluster. Open it in a browser to rotate,
zoom and toggle clusters. Plotly is embedded so the file works offline.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


CLASSES = ("STIC", "HGSC", "NFT")
LABEL_PATTERN = re.compile(r"(?<![A-Z0-9])(STIC|HGSC|NFT)(?![A-Z0-9])", re.I)
# Identifiers and annotations must not enter clustering, even via a feature list.
METADATA = set("""id index target type type_name type_prob label class diagnosis
    group category phenotype slide region region_type source_json source_file
    cell_label cell_id cell_type patient patient_id subject subject_id roi roi_id
    true_label predicted_label split mpp center contour cx_roi cy_roi cx_wsi cy_wsi
    cx_wsi_um cy_wsi_um centroid_x_px centroid_y_px centroid_x_um centroid_y_um
    roi_class cluster cluster_id confidence""".split())


def csv_paths(path: Path) -> list[Path]:
    if path.is_file():
        if "cleaned" in path.stem:
            raise ValueError("Use raw *_cell_features.csv; --feature-table supplies the 40 selected feature names.")
        return [path]
    files = sorted(p for p in path.glob("*_cell_features.csv")
                   if p.name != "all_cell_features.csv")
    if files:
        return files  # Never load the aggregate alongside its constituent files.
    aggregate = path / "all_cell_features.csv"
    if aggregate.is_file():
        return [aggregate]
    raise ValueError(f"No raw *_cell_features.csv files found in {path}")


def read_cells(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, records, seen = [], [], set()
    for source in csv_paths(path):
        frame = pd.read_csv(source)
        if frame.empty:
            raise ValueError(f"Empty feature table: {source}")
        if any(str(c).startswith("__") for c in frame):
            raise ValueError("Input columns starting with '__' are reserved.")
        for column in ("slide", "region"):
            if column not in frame:
                frame[column] = source.stem
        if frame[["slide", "region"]].isna().any().any():
            raise ValueError(f"Missing slide/region identifiers: {source}")
        frame["__source_row"] = np.arange(len(frame))
        frame["__source_file"] = str(source.resolve())
        for (slide, region), group in frame.groupby(["slide", "region"], sort=True):
            key = (str(slide), str(region))
            if key in seen:
                raise ValueError(f"Duplicate ROI {key}; do not mix duplicate feature files.")
            seen.add(key)
            matches = set(LABEL_PATTERN.findall(str(region).upper()))
            # Per-slide tables may contain multiple diagnoses: use filename only
            # when the region itself has no diagnosis, never override a region.
            if not matches:
                matches = set(LABEL_PATTERN.findall(source.stem.upper()))
            if len(matches) > 1:
                raise ValueError(f"Ambiguous ROI diagnosis: {key}")
            roi_index = len(records)
            records.append(dict(roi_index=roi_index, slide=key[0], region=key[1],
                                source_file=str(source.resolve()), n_cells=len(group),
                                roi_class=next(iter(matches), None)))
            group = group.copy()
            group["__roi_index"] = roi_index
            frames.append(group)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(records).set_index("roi_index")


def select_rois(rois: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    counts = rois.roi_class.value_counts().reindex(CLASSES, fill_value=0)
    if (counts < per_class).any():
        raise ValueError(f"Need {per_class} ROIs per class; available: {counts.to_dict()}. "
                         "Supply more raw features or explicitly lower --rois-per-class.")
    rng = np.random.default_rng(seed)
    selected = []
    for label in CLASSES:
        candidates = rois.loc[rois.roi_class == label].sort_values(["slide", "region"])
        selected.extend(rng.choice(candidates.index.to_numpy(), per_class, replace=False))
    result = rois.loc[selected].sort_values(["roi_class", "slide", "region"]).copy()
    return result


def selected_features(table_path: Path) -> list[str]:
    """Reuse the existing selection; do not select features again on these ROIs."""
    table = pd.read_csv(table_path)
    if "selected_features" in table:
        columns = [name.strip() for value in table.selected_features.dropna().astype(str)
                   for name in value.split(";") if name.strip()]
    elif "feature" in table:
        columns = [name.strip() for name in table.feature.dropna().astype(str) if name.strip()]
    else:
        raise ValueError("Feature table needs a 'selected_features' or 'feature' column.")
    if len(columns) != 40 or len(set(columns)) != 40:
        raise ValueError(f"Expected exactly 40 distinct selected features in {table_path}; "
                         f"got {len(columns)} entries and {len(set(columns))} distinct names.")
    forbidden = [column for column in columns if column.lower() in METADATA
                 or column.lower().startswith(("_", "bbox_", "unnamed:", "prob_"))
                 or column.lower().endswith(("_id", "_label", "_class"))]
    if forbidden:
        raise ValueError(f"Feature table contains metadata/labels: {forbidden}")
    return columns


def feature_matrix(cells: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Use only the supplied 40 features, preserving the saved selection order."""
    missing = [column for column in columns if column not in cells]
    if missing:
        raise ValueError(f"Selected features missing from input: {missing}")
    raw = cells.loc[:, columns]
    features = raw.apply(pd.to_numeric, errors="coerce")
    invalid = (raw.notna() & features.isna()).any()
    if invalid.any():
        raise ValueError(f"Non-numeric values in selected features: {invalid.index[invalid].tolist()}")
    features = features.replace([np.inf, -np.inf], np.nan)
    empty = features.isna().all()
    if empty.any():
        raise ValueError(f"Selected features have no finite values in these ROIs: {empty.index[empty].tolist()}")
    if features.isna().all(axis=1).any():
        raise ValueError("Some cells have no observed features; repair the input before clustering.")
    # A feature may be constant within the sampled ROIs; retain it so the saved
    # selection remains exactly 40 columns. StandardScaler safely centers it.
    return features


def fit_cells(features: pd.DataFrame, args: argparse.Namespace):
    """Accept only the feature matrix: no ROI labels, names or coordinates."""
    if len(features) < args.n_components:
        raise ValueError(f"Need at least {args.n_components} cells, got {len(features)}.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputer.fit_transform(features))
    if np.unique(scaled, axis=0).shape[0] < args.n_components:
        raise ValueError("Fewer distinct feature vectors than requested GMM components.")
    gmm = GaussianMixture(n_components=args.n_components, covariance_type=args.covariance_type,
                          reg_covar=args.reg_covar, n_init=args.n_init,
                          max_iter=args.max_iter, random_state=args.seed)
    gmm.fit(scaled)
    probabilities = gmm.predict_proba(scaled)
    model = dict(schema_version=2, unit="cell", feature_columns=features.columns.tolist(),
                 imputer=imputer, scaler=scaler, gmm=gmm,
                 cluster_id_convention="GMM component index + 1; no diagnosis mapping")
    return model, scaled, probabilities


def plot_pca_3d(scaled: np.ndarray, clusters: np.ndarray, n_components: int, output: Path):
    """Save an offline, rotatable 3D plot of all cells using the selected features."""
    if min(scaled.shape) < 3:
        raise ValueError("PC1/PC2/PC3 requires at least three cells and three usable features.")
    pca = PCA(n_components=3, svd_solver="full")
    xyz = pca.fit_transform(scaled)
    cmap = colormaps["tab20" if n_components <= 20 else "hsv"]
    colors = [to_hex(cmap(i if n_components <= 20 else i / n_components))
              for i in range(n_components)]
    fig = go.Figure()
    for cluster, color in enumerate(colors):
        mask = clusters == cluster
        points = xyz[mask]
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            name=f"C{cluster + 1:02d} ({mask.sum():,} cells)",
            marker=dict(size=2.5, color=color, opacity=.7),
            hovertemplate=(f"C{cluster + 1:02d}<br>PC1: %{{x:.3f}}<br>"
                           "PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>"),
        ))
    axis_titles = [f"PC{i + 1} ({ratio:.1%})"
                   for i, ratio in enumerate(pca.explained_variance_ratio_)]
    fig.update_layout(
        title=dict(text=f"Cell GMM: {n_components} clusters | {len(xyz):,} cells | "
                        f"{scaled.shape[1]} selected features<br>"
                        f"PC1-PC3: {pca.explained_variance_ratio_.sum():.1%} explained variance"),
        scene=dict(xaxis_title=axis_titles[0], yaxis_title=axis_titles[1],
                   zaxis_title=axis_titles[2], dragmode="orbit", aspectmode="cube"),
        template="plotly_white", margin=dict(l=0, r=0, t=85, b=40),
        legend=dict(title="Clusters", itemsizing="constant"),
        annotations=[dict(text="Drag to rotate | Scroll to zoom | Click legend to toggle clusters",
                          x=.5, y=0, xref="paper", yref="paper", yshift=-30,
                          showarrow=False)],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True, full_html=True,
                   default_width="100%", default_height="100vh",
                   config=dict(responsive=True, scrollZoom=True, displaylogo=False))


def run(args: argparse.Namespace):
    input_folder = args.input_dir.parent if args.input_dir.is_file() else args.input_dir
    table_path = args.feature_table or (input_folder.with_name(input_folder.name + "_cleaned")
                                       / "feature_group_selection_table.csv")
    columns = selected_features(table_path)
    cells, rois = read_cells(args.input_dir)
    selected = select_rois(rois, args.rois_per_class, args.seed)
    cells = cells.loc[cells.__roi_index.isin(selected.index)].reset_index(drop=True)
    features = feature_matrix(cells, columns)
    if min(features.shape) < 3:
        raise ValueError("PC1/PC2/PC3 requires at least three cells and three usable features.")
    print(f"Fitting {args.n_components}-component GMM on {len(cells):,} cells, "
          f"{len(selected)} ROIs, {features.shape[1]} features; ROI labels excluded.", flush=True)
    model, scaled, probabilities = fit_cells(features, args)
    if not model["gmm"].converged_:
        warnings.warn("GMM did not converge; increase --max-iter before interpretation.")
    output = args.output_dir / "cell_clusters_pca_3d.html"
    plot_pca_3d(scaled, probabilities.argmax(axis=1), args.n_components, output)
    print(f"Saved: {output.resolve()}", flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("train",), help="Optional legacy train alias")
    parser.add_argument("--input-dir", type=Path, default=Path("cellvit_output/feature_data"))
    parser.add_argument("--feature-table", type=Path,
                        help="Existing 40-feature selection table; default: <input-folder>_cleaned/feature_group_selection_table.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("clustering_results/cell_gmm_20"))
    parser.add_argument("--rois-per-class", type=int, default=5,
                        help="Number of ROIs to sample from EACH of STIC, HGSC and NFT (default: 5)")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--covariance-type", choices=("diag", "full", "tied", "spherical"), default="diag",
                        help="Diagonal covariance is the default for the 40 selected features.")
    parser.add_argument("--reg-covar", type=float, default=1e-5)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args(argv)
    for name in ("rois_per_class", "n_components", "n_init", "max_iter"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.reg_covar) or args.reg_covar <= 0:
        parser.error("--reg-covar must be finite and positive")
    return args


if __name__ == "__main__":
    run(parse_args())
