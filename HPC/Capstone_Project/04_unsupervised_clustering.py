"""Cell-level GMM: 40 selected features, configurable number of clusters.

    python 04_unsupervised_clustering.py --input-dir cellvit_output/feature_data
    python 04_unsupervised_clustering.py --input-dir feature_data --cells-per-class 1000
    python 04_unsupervised_clustering.py --n-components 30

Read the 40 feature names from Script03's feature_group_selection_table.csv,
then use only those columns from the raw per-cell measurements. By default the
table is in the input folder's sibling <input-folder>_cleaned directory;
--feature-table can specify another table. Never fall back to all features.
Each cell inherits its sampling class from its ROI diagnosis (STIC/HGSC/NFT).
Use --cells-per-class as a target cell count for EACH class (default 1000).
Randomly order each class's ROIs without replacement, then include complete
ROIs until their cumulative cell count reaches or exceeds the target. Keep
ALL cells in the final ROI, so actual class sizes may differ and exceed the
target, including when a single ROI already exceeds it. Never trim an ROI.
The same --seed reproduces the selection for unchanged inputs.
All three classes must have enough cells to reach the target; lower
--cells-per-class if necessary. Unclassified ROIs are excluded with a warning.
The sampled cells enter one pooled GMM; diagnoses are never model features.
Median imputation and standardization are fitted on these selected cells only.
PCA is for display only; GMM sees the 40 standardized selected features.
All input cells from every selected ROI enter GMM and its outputs.
ROI overlays match cell_label to
the original Script02 instance mask (unique positive labels within each ROI).
Set --n-components to any positive cluster count supported by the data (default
20). Results default to clustering_results/cell_gmm_<n-components>; override
with --output-dir if needed.

Outputs include cell_clusters_pca_3d.html, showing all selected cells in
PC1/PC2/PC3 space, colored by their GMM cluster. Open it in a browser to rotate,
zoom and toggle clusters. Plotly is embedded so the file works offline.
cell_cluster_feature_heatmap.png has one row per selected feature and one
column per GMM cluster (40 x 6 when --n-components 6). Each entry is the mean
of the same imputed, standardized feature values used by GMM. Before plotting,
min-max normalize each feature's cluster means to [-1, 1]: -1 is the lowest
cluster mean, +1 is the highest. Constant rows map to 0; empty clusters are
gray. These are descriptive feature profiles, not model feature importance.
Save the normalized matrix to cell_cluster_feature_heatmap.csv and the means
before normalization to cell_cluster_feature_means.csv, in selection-table order.
roi_cluster_overlays/ contains original-resolution H&E images with only cell
boundaries colored by cluster, plus cluster_legend.png and an ROI manifest.
Cluster colors use a vivid categorical palette shared by overlays, their
legend, PCA and assignments; outlines are painted at full opacity.
cell_cluster_assignments.csv records each fitted cell's ROI class, cluster and color.
roi_sampling_summary.csv records each input ROI's available and sampled cell
counts, so sampling coverage can be checked: each ROI has either all its
available cells sampled or zero cells sampled. requested_cells_per_class is
the target, not a cap on the actual sampled count.
roi_cluster_counts.csv summarizes those same fitted cells: one row per ROI
(identified by slide and region), and one integer count column per cluster,
C01 through C<n-components>. Absent clusters are recorded as zero. This CSV
is also saved when --skip-roi-overlays is used.
--cellvit-root points to Script02 outputs (default: cellvit_output). Exact
region_rgb/region_mask pairs are preferred, with rgb/mask pairs as fallback.
Use --outline-width for boundary thickness in pixels, or --skip-roi-overlays
to run clustering when the original images/masks are unavailable.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import binary_erosion
from skimage.measure import regionprops
import tifffile
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


CLASSES = ("STIC", "HGSC", "NFT")
DEFAULT_N_COMPONENTS = 20
DEFAULT_CELLS_PER_CLASS = 1000
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


def sample_cells(cells: pd.DataFrame, rois: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Reach each class's cell target using whole ROIs, including the final ROI."""
    roi_classes = cells.__roi_index.map(rois.roi_class)
    counts = roi_classes.value_counts().reindex(CLASSES, fill_value=0)
    if per_class < 1:
        raise ValueError("--cells-per-class must be positive.")
    if (counts < per_class).any():
        raise ValueError(f"Need {per_class} cells per ROI class; available: {counts.to_dict()}. "
                         f"The largest target supported by all classes is {int(counts.min())}. "
                         "Lower --cells-per-class or supply more cells from the missing class; "
                         "sampling does not duplicate cells.")
    unknown = ~roi_classes.isin(CLASSES)
    if unknown.any():
        warnings.warn(f"Excluded {int(unknown.sum()):,} cells with unrecognized ROI diagnosis from sampling.")
    rng = np.random.default_rng(seed)
    roi_cell_counts = cells.__roi_index.value_counts().reindex(rois.index, fill_value=0)
    selected_rois = []
    for label in CLASSES:
        candidates = rois.index[(rois.roi_class == label) & roi_cell_counts.gt(0)].to_numpy()
        total = 0
        for roi_index in rng.permutation(candidates):
            selected_rois.append(roi_index)
            total += int(roi_cell_counts.loc[roi_index])
            if total >= per_class:
                break
    # Consolidate before adding metadata, including when input ROI tables have
    # different column layouts. Original source rows and mask IDs remain intact.
    result = cells.loc[cells.__roi_index.isin(selected_rois)].reset_index(drop=True).copy()
    result["roi_class"] = result.__roi_index.map(rois.roi_class)
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


def cluster_colors(n_components: int) -> list[str]:
    """Use distinct saturated colors instead of tab20's light/dark pairs."""
    palette = (
        "#0066FF", "#00CC44", "#FF7A00", "#00D9E6", "#FFD600", "#D500F9",
        "#FF1744", "#004D40", "#7C4DFF", "#76FF03", "#8C2F00", "#008C99",
        "#B89B00", "#000080", "#A00060", "#4A6600", "#00FF99", "#FF4081",
        "#6500A8", "#263238",
    )
    if n_components <= len(palette):
        return list(palette[:n_components])
    cmap = colormaps["hsv"]
    return [to_hex(cmap(i / n_components)) for i in range(n_components)]


def roi_image_paths(root: Path, slide: str, region: str) -> tuple[Path, Path]:
    """Resolve an exact image/mask pair; never substitute a tile or other ROI."""
    tag = f"{slide}_{region}"
    if any(char in tag for char in ("/", "\\")):
        raise ValueError(f"Slide/region must not contain path separators: {tag}")
    for rgb_dir, mask_dir in (("region_rgb", "region_mask"), ("rgb", "mask")):
        rgb = root / rgb_dir / f"rgb_{tag}.tif"
        mask = root / mask_dir / f"mask_{tag}.tif"
        if rgb.is_file() and mask.is_file():
            return rgb, mask
    raise FileNotFoundError(
        f"No matching ROI RGB/mask pair for {tag} under {root}. "
        "Set --cellvit-root to the Script02 output directory, or explicitly "
        "use --skip-roi-overlays to produce clustering plots only.")


def validate_overlay_cells(cells: pd.DataFrame) -> None:
    """Prevent ambiguous assignments before painting any cell."""
    if "cell_label" not in cells:
        raise ValueError("ROI overlays require cell_label from Script03's raw feature table.")
    labels = pd.to_numeric(cells.cell_label, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(labels) & (labels > 0) & (labels == np.floor(labels))):
        raise ValueError("ROI overlays require finite positive integer cell_label values.")
    keys = cells[["slide", "region"]].copy()
    keys["cell_label"] = labels
    if keys.duplicated().any():
        raise ValueError("Duplicate cell_label within a slide/region; ROI cluster mapping is ambiguous.")


def draw_cluster_outlines(rgb: np.ndarray, labels: np.ndarray,
                          label_clusters: dict[int, int], colors: list[str],
                          width: int) -> np.ndarray:
    """Paint an inner boundary band; all remaining RGB pixels stay identical."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("ROI image must be an original uint8 RGB image.")
    if (labels.ndim != 2 or labels.shape != rgb.shape[:2]
            or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)):
        raise ValueError("ROI mask must be a nonnegative integer label image matching the RGB dimensions.")
    if width < 1:
        raise ValueError("Outline width must be positive.")
    props = {int(prop.label): prop for prop in regionprops(labels)}
    missing = sorted(set(label_clusters) - props.keys())
    if missing:
        raise ValueError(f"Feature cell_label values missing from the ROI mask: {missing[:10]}")
    palette = [tuple(bytes.fromhex(color.lstrip("#"))) for color in colors]
    overlay = rgb.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    for label, cluster in label_clusters.items():
        prop = props[label]
        r0, c0, r1, c1 = prop.bbox
        binary = (labels[r0:r1, c0:c1] == label).astype(np.uint8)
        interior = binary_erosion(binary, structure=kernel, iterations=width, border_value=0)
        boundary = (binary != 0) & (interior == 0)
        overlay[r0:r1, c0:c1][boundary] = palette[cluster]
    return overlay


def save_roi_overlays(cells: pd.DataFrame, clusters: np.ndarray, n_components: int,
                      root: Path, output_dir: Path, width: int) -> Path:
    """Export full-resolution outlines for only the cells included in this fit."""
    validate_overlay_cells(cells)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = cluster_colors(n_components)
    assignments = cells.copy()
    assignments["__cluster"] = clusters
    records = []
    for index, ((slide, region), group) in enumerate(
            assignments.groupby(["slide", "region"], sort=True), start=1):
        rgb_path, mask_path = roi_image_paths(root, str(slide), str(region))
        label_clusters = dict(zip(group.cell_label.astype(int), group.__cluster.astype(int)))
        rgb = tifffile.imread(rgb_path)
        labels = tifffile.imread(mask_path)
        try:
            overlay = draw_cluster_outlines(rgb, labels, label_clusters, colors, width)
        except ValueError as exc:
            raise ValueError(f"{slide} / {region}: {exc}") from exc
        tag = re.sub(r'[^\w. -]', "_", f"{slide}_{region}")[:160]
        destination = output_dir / f"{index:03d}_{tag}_clusters.png"
        Image.fromarray(overlay).save(destination)
        counts = np.bincount(group.__cluster.to_numpy(), minlength=n_components)
        records.append(dict(slide=slide, region=region, n_cells=len(group),
                            rgb_path=str(rgb_path.resolve()), mask_path=str(mask_path.resolve()),
                            overlay_path=str(destination.resolve()), outline_width_px=width,
                            **{f"C{i + 1:02d}": int(n) for i, n in enumerate(counts)}))
        print(f"Saved ROI overlay ({len(group):,} cells): {destination.resolve()}", flush=True)
        del rgb, labels, overlay
    pd.DataFrame(records).to_csv(output_dir / "roi_overlay_manifest.csv", index=False)
    counts = np.bincount(clusters, minlength=n_components)
    handles = [Patch(facecolor="none", edgecolor=color, linewidth=2,
                     label=f"C{i + 1:02d} (n={counts[i]:,})") for i, color in enumerate(colors)]
    ncols = min(5, n_components)
    nrows = (n_components + ncols - 1) // ncols
    fig = Figure(figsize=(3 * ncols, 1.4 + .4 * nrows))
    FigureCanvasAgg(fig)
    fig.legend(handles=handles, loc="center", ncol=ncols, frameon=False)
    fig.suptitle(f"GMM: {n_components} clusters | colors shared across all fitted ROIs", fontsize=11)
    fig.text(.5, .05, f"Cell outlines only ({width} px); interiors retain original H&E. "
             "Counts: all fitted cells.", ha="center", fontsize=9)
    fig.savefig(output_dir / "cluster_legend.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.clear()
    return output_dir


def cluster_feature_profiles(scaled: np.ndarray, clusters: np.ndarray,
                             feature_names: list[str], n_components: int):
    """Return feature x cluster means, row-normalized means, and cell counts."""
    means = np.full((len(feature_names), n_components), np.nan)
    counts = np.bincount(clusters, minlength=n_components)
    for cluster in range(n_components):
        if counts[cluster]:
            means[:, cluster] = scaled[clusters == cluster].mean(axis=0)
    low = np.nanmin(means, axis=1, keepdims=True)
    high = np.nanmax(means, axis=1, keepdims=True)
    span = high - low
    # Avoid amplifying numerical roundoff when cluster means are equal.
    constant = np.isclose(span, 0, atol=1e-12, rtol=0)
    normalized = 2 * (means - low) / np.where(constant, 1, span) - 1
    normalized = np.where(constant & np.isfinite(means), 0, normalized)
    normalized = np.clip(normalized, -1, 1)
    index = pd.Index(feature_names, name="feature")
    columns = [f"C{i + 1:02d}" for i in range(n_components)]
    return (pd.DataFrame(means, index=index, columns=columns),
            pd.DataFrame(normalized, index=index, columns=columns), counts)


def plot_feature_heatmap(scaled: np.ndarray, clusters: np.ndarray,
                         feature_names: list[str], n_components: int, output: Path):
    """Plot relative feature levels within each hard-assigned GMM cluster."""
    means, normalized, counts = cluster_feature_profiles(scaled, clusters, feature_names, n_components)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output.with_suffix(".csv"))
    means.to_csv(output.parent / "cell_cluster_feature_means.csv")
    cmap = colormaps["coolwarm"].copy()
    cmap.set_bad("#dedede")
    fig = Figure(figsize=(max(10, 5 + .65 * n_components), max(5, .3 * len(feature_names) + 2.5)),
                 constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    values = normalized.to_numpy()
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
                      cmap=cmap, vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xticks(np.arange(n_components))
    ax.set_xticklabels([f"C{i + 1:02d}\n(n={counts[i]:,})" for i in range(n_components)],
                       rotation=0 if n_components <= 12 else 90, fontsize=8)
    ax.set_xlabel("GMM cluster")
    ax.set_ylabel("Feature")
    ax.set_title(f"Feature profiles by GMM cluster | {len(feature_names)} features x {n_components} clusters\n"
                 f"{len(scaled):,} fitted cells | per-feature min-max normalization", fontsize=12, pad=12)
    ax.set_xticks(np.arange(n_components + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(feature_names) + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    if n_components <= 12:
        for row in range(len(feature_names)):
            for column in range(n_components):
                value = values[row, column]
                label = f"{value:.2f}" if np.isfinite(value) else "-"
                ax.text(column, row, label, ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and abs(value) > .65 else "#222222")
    fig.colorbar(image, ax=ax, shrink=.65, ticks=[-1, -.5, 0, .5, 1],
                 label="Normalized cluster mean")
    fig.supxlabel("Within each feature: -1 = lowest mean, +1 = highest mean. Constant rows = 0; empty clusters = gray.\n"
                  "Descriptive feature profiles; normalization does not measure model feature importance.", fontsize=8)
    fig.savefig(output, dpi=180, facecolor="white")
    fig.clear()


def plot_pca_3d(scaled: np.ndarray, clusters: np.ndarray, n_components: int, output: Path):
    """Save an offline, rotatable 3D plot of all cells using the selected features."""
    if min(scaled.shape) < 3:
        raise ValueError("PC1/PC2/PC3 requires at least three cells and three usable features.")
    pca = PCA(n_components=3, svd_solver="full")
    xyz = pca.fit_transform(scaled)
    colors = cluster_colors(n_components)
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


def save_roi_cluster_counts(assignments: pd.DataFrame, n_components: int, output: Path):
    """Count hard-assigned fitted cells per ROI, preserving all cluster columns."""
    counts = (assignments.groupby(["slide", "region", "cluster_id"], sort=True)
              .size().unstack("cluster_id", fill_value=0)
              .reindex(columns=range(1, n_components + 1), fill_value=0)
              .astype(np.int64))
    counts.columns = [f"C{i + 1:02d}" for i in range(n_components)]
    result = counts.reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def run(args: argparse.Namespace):
    input_folder = args.input_dir.parent if args.input_dir.is_file() else args.input_dir
    table_path = args.feature_table or (input_folder.with_name(input_folder.name + "_cleaned")
                                       / "feature_group_selection_table.csv")
    columns = selected_features(table_path)
    cells, rois = read_cells(args.input_dir)
    cells = sample_cells(cells, rois, args.cells_per_class, args.seed)
    selected = rois.loc[sorted(cells.__roi_index.unique())]
    features = feature_matrix(cells, columns)
    if not args.skip_roi_overlays:
        validate_overlay_cells(cells)
        for row in selected.itertuples():
            roi_image_paths(args.cellvit_root, row.slide, row.region)
    if min(features.shape) < 3:
        raise ValueError("PC1/PC2/PC3 requires at least three cells and three usable features.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampling = rois.rename(columns={"n_cells": "available_cells"}).copy()
    sampling["sampled_cells"] = cells.__roi_index.value_counts().reindex(rois.index, fill_value=0)
    sampling["requested_cells_per_class"] = args.cells_per_class
    sampling["sampling_seed"] = args.seed
    sampling_output = args.output_dir / "roi_sampling_summary.csv"
    sampling.to_csv(sampling_output)
    print(f"Cell target per ROI class: {args.cells_per_class:,}; selecting complete ROIs "
          f"without replacement (seed={args.seed}).", flush=True)
    for label in CLASSES:
        group = cells.loc[cells.roi_class == label]
        print(f"  {label}: {len(group):,} cells from {group.__roi_index.nunique()} complete ROIs "
              f"({len(group) - args.cells_per_class:,} cells above target).", flush=True)
    print(f"Saved: {sampling_output.resolve()}", flush=True)
    print(f"Fitting {args.n_components}-component GMM on {len(cells):,} cells, "
          f"{len(selected)} ROIs, {features.shape[1]} features; ROI labels excluded.", flush=True)
    model, scaled, probabilities = fit_cells(features, args)
    if not model["gmm"].converged_:
        warnings.warn("GMM did not converge; increase --max-iter before interpretation.")
    output = args.output_dir / "cell_clusters_pca_3d.html"
    clusters = probabilities.argmax(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_columns = [name for name in ("slide", "region", "roi_class", "cell_label", "cx_roi", "cy_roi",
                                   "cx_wsi", "cy_wsi", "__source_file", "__source_row")
                  if name in cells]
    assignments = cells[id_columns].rename(columns={"__source_file": "source_file",
                                                    "__source_row": "source_row"}).copy()
    assignments["cluster_id"] = clusters + 1
    assignments["cluster_confidence"] = probabilities.max(axis=1)
    assignments["cluster_color"] = np.asarray(cluster_colors(args.n_components))[clusters]
    assignment_output = args.output_dir / "cell_cluster_assignments.csv"
    assignments.to_csv(assignment_output, index=False)
    print(f"Saved: {assignment_output.resolve()}", flush=True)
    roi_counts_output = args.output_dir / "roi_cluster_counts.csv"
    save_roi_cluster_counts(assignments, args.n_components, roi_counts_output)
    print(f"Saved: {roi_counts_output.resolve()}", flush=True)
    plot_pca_3d(scaled, clusters, args.n_components, output)
    print(f"Saved: {output.resolve()}", flush=True)
    feature_output = args.output_dir / "cell_cluster_feature_heatmap.png"
    plot_feature_heatmap(scaled, clusters, columns, args.n_components, feature_output)
    print(f"Saved: {feature_output.resolve()}", flush=True)
    if not args.skip_roi_overlays:
        save_roi_overlays(cells, clusters, args.n_components, args.cellvit_root,
                          args.output_dir / "roi_cluster_overlays", args.outline_width)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("train",), help="Optional legacy train alias")
    parser.add_argument("--input-dir", type=Path, default=Path("cellvit_output/feature_data"))
    parser.add_argument("--feature-table", type=Path,
                        help="Existing 40-feature selection table; default: <input-folder>_cleaned/feature_group_selection_table.csv")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: clustering_results/cell_gmm_<n-components>")
    parser.add_argument("--cellvit-root", type=Path, default=Path("cellvit_output"),
                        help="Script02 output root containing region_rgb/region_mask (or rgb/mask)")
    parser.add_argument("--outline-width", type=int, default=2,
                        help="Width of the colored inner cell boundary in original ROI pixels (default: 2)")
    parser.add_argument("--skip-roi-overlays", action="store_true",
                        help="Skip ROI images when original RGB/masks are unavailable")
    parser.add_argument("--cells-per-class", type=int, default=DEFAULT_CELLS_PER_CLASS,
                        help=f"Target cells per ROI diagnosis class STIC/HGSC/NFT (default: {DEFAULT_CELLS_PER_CLASS}); "
                             "include whole ROIs until the target is reached; retain all cells in the last ROI")
    parser.add_argument("--rois-per-class", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS,
                        help=f"Number of GMM clusters, no fixed upper limit (default: {DEFAULT_N_COMPONENTS})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--covariance-type", choices=("diag", "full", "tied", "spherical"), default="diag",
                        help="Diagonal covariance is the default for the 40 selected features.")
    parser.add_argument("--reg-covar", type=float, default=1e-5)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args(argv)
    if args.rois_per_class is not None:
        parser.error("--rois-per-class is no longer used. Set --cells-per-class instead; "
                     "its value is a CELL count per ROI diagnosis class.")
    for name in ("cells_per_class", "n_components", "n_init", "max_iter", "outline_width"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.reg_covar) or args.reg_covar <= 0:
        parser.error("--reg-covar must be finite and positive")
    if args.output_dir is None:
        args.output_dir = Path("clustering_results") / f"cell_gmm_{args.n_components}"
    return args


if __name__ == "__main__":
    run(parse_args())
