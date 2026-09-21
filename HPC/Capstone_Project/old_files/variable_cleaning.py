from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


DEFAULT_INPUT_DIR = "HPC/Capstone_Project/feature_data"
DEFAULT_OUTPUT_DIR = "HPC/Capstone_Project/feature_data_cleaned"
DEFAULT_FEATURE_GROUPS = 20
DEFAULT_FEATURES_PER_GROUP = 2
DEFAULT_MAX_FEATURES = DEFAULT_FEATURE_GROUPS * DEFAULT_FEATURES_PER_GROUP

METADATA_COLUMNS = {
    "id",
    "index",
    "target",
    "type",
    "type_name",
    "label",
    "class",
    "diagnosis",
    "group",
    "category",
    "phenotype",
    "slide",
    "region",
    "region_type",
    "source_json",
    "source_file",
    "cell_label",
    "cell_id",
    "cell_type",
    "centroid_x_px",
    "centroid_y_px",
    "centroid_x_um",
    "centroid_y_um",
    "mpp",
    "cx_roi",
    "cy_roi",
    "cx_wsi",
    "cy_wsi",
    "cx_wsi_um",
    "cy_wsi_um",
    "center",
    "contour",
    "orientation_rad",
    "orientation_deg",
    "mitotic_figure_density_local",
    "tufting_budding_index_proxy",
}

DROP_OUTPUT_COLUMNS = {
    "type",
    "type_name",
}

EXCLUDED_FEATURE_PREFIXES = (
    "bbox_",
)

EXCLUDED_FEATURE_SUFFIXES = (
    "_px",
    "_px2",
)


def feature_csv_paths(input_dir: Path) -> list[Path]:
    csv_paths = sorted(
        path
        for path in input_dir.glob("*.csv")
        if not path.name.endswith("_cleaned.csv")
    )
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in input folder: {input_dir}")

    split_feature_paths = [
        path
        for path in csv_paths
        if path.name.endswith("_cell_features.csv") and path.name != "all_cell_features.csv"
    ]
    if split_feature_paths:
        return split_feature_paths
    return csv_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove non-informative features, robust-scale numeric features, then "
            "select diverse columns by hierarchical clustering of features."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Input folder containing feature CSV files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Folder for cleaned CSV files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=DEFAULT_MAX_FEATURES,
        help="Maximum selected feature count. Must be less than 50; default: 40.",
    )
    parser.add_argument(
        "--feature-groups",
        type=int,
        default=DEFAULT_FEATURE_GROUPS,
        help="Number of hierarchical feature groups to form; default: 20.",
    )
    parser.add_argument(
        "--features-per-group",
        type=int,
        default=DEFAULT_FEATURES_PER_GROUP,
        help="Number of mutually different features to select per group; default: 2.",
    )
    return parser.parse_args()


def feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = {col for col in data.columns if col.lower() in METADATA_COLUMNS}

    numeric_cols = data.select_dtypes(include="number").columns
    return [
        col
        for col in numeric_cols
        if col not in excluded
        and not col.startswith(EXCLUDED_FEATURE_PREFIXES)
        and not col.endswith(EXCLUDED_FEATURE_SUFFIXES)
    ]


def drop_zero_variance_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    variances = features.var(axis=0, skipna=True)
    non_constant = variances.gt(0) & variances.notna()
    dropped = variances.index[~non_constant].tolist()
    return features.loc[:, non_constant], dropped


def robust_scale_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    q1 = features.quantile(0.25)
    q2 = features.quantile(0.50)
    q3 = features.quantile(0.75)
    iqr = q3 - q1

    scalable = iqr.gt(0) & iqr.notna()
    dropped = iqr.index[~scalable].tolist()
    scaled = (features.loc[:, scalable] - q2.loc[scalable]) / iqr.loc[scalable]
    return scaled, dropped


def impute_scaled_features(features: pd.DataFrame) -> pd.DataFrame:
    medians = features.median(axis=0, skipna=True)
    return features.fillna(medians).fillna(0.0)


def feature_correlation_matrices(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputed = impute_scaled_features(features)
    corr = imputed.corr(method="spearman").clip(lower=-1.0, upper=1.0)
    if not np.isfinite(corr.to_numpy()).all():
        raise ValueError("Spearman correlation requires non-constant, finite feature columns.")
    distance = 1.0 - corr.abs()
    return corr, distance


def feature_correlation_distance(features: pd.DataFrame) -> pd.DataFrame:
    return feature_correlation_matrices(features)[1]


def hierarchical_feature_groups(
    features: pd.DataFrame,
    n_groups: int,
) -> tuple[dict[int, list[str]], pd.DataFrame]:
    if n_groups < 1:
        raise ValueError("--feature-groups must be positive.")

    distance = feature_correlation_distance(features)
    columns = list(features.columns)
    if len(columns) <= n_groups:
        return {idx + 1: [col] for idx, col in enumerate(columns)}, distance

    condensed = squareform(distance.to_numpy(), checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=n_groups, criterion="maxclust")

    raw_groups: dict[int, list[str]] = {}
    for col, label in zip(columns, labels):
        raw_groups.setdefault(int(label), []).append(col)

    ordered_groups = sorted(
        raw_groups.values(),
        key=lambda group: min(columns.index(col) for col in group),
    )
    return {idx + 1: group for idx, group in enumerate(ordered_groups)}, distance


def select_different_features_from_group(
    group_cols: list[str],
    distance: pd.DataFrame,
    n_features: int,
) -> list[str]:
    if n_features < 1:
        return []
    if len(group_cols) <= n_features:
        return list(group_cols)
    if n_features == 1:
        group_distance = distance.loc[group_cols, group_cols]
        mean_distance = group_distance.mean(axis=1)
        return [mean_distance.sort_values(ascending=False, kind="mergesort").index[0]]

    best_pair = None
    best_distance = -np.inf
    for i, first in enumerate(group_cols):
        for second in group_cols[i + 1 :]:
            pair_distance = float(distance.loc[first, second])
            if pair_distance > best_distance:
                best_distance = pair_distance
                best_pair = [first, second]

    selected = best_pair or group_cols[:1]
    while len(selected) < n_features and len(selected) < len(group_cols):
        candidates = [col for col in group_cols if col not in selected]
        min_distances = {
            col: float(distance.loc[col, selected].min())
            for col in candidates
        }
        next_col = max(candidates, key=lambda col: (min_distances[col], -group_cols.index(col)))
        selected.append(next_col)
    return selected


def select_relevant_features(
    features: pd.DataFrame,
    max_features: int,
    n_groups: int,
    features_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]]]:
    groups, distance = hierarchical_feature_groups(features, n_groups)

    selected_cols = []
    selection_rows = []
    for group_id, group_cols in groups.items():
        remaining = max_features - len(selected_cols)
        if remaining <= 0:
            break
        group_quota = min(features_per_group, remaining)
        group_selected = select_different_features_from_group(
            group_cols,
            distance,
            group_quota,
        )
        selected_cols.extend(group_selected)
        for rank, col in enumerate(group_selected, start=1):
            partner_distances = [
                float(distance.loc[col, other])
                for other in group_selected
                if other != col
            ]
            selection_rows.append(
                {
                    "feature": col,
                    "feature_group": group_id,
                    "group_size": len(group_cols),
                    "within_group_rank": rank,
                    "mean_distance_to_selected_group_partner": (
                        float(np.mean(partner_distances)) if partner_distances else np.nan
                    ),
                }
            )

    if len(selected_cols) < max_features:
        remaining_cols = [col for col in features.columns if col not in selected_cols]
        while remaining_cols and len(selected_cols) < max_features:
            candidate_distances = {
                col: float(distance.loc[col, selected_cols].min()) if selected_cols else 1.0
                for col in remaining_cols
            }
            next_col = max(
                remaining_cols,
                key=lambda col: (candidate_distances[col], -list(features.columns).index(col)),
            )
            selected_cols.append(next_col)
            group_id = next(
                group_id
                for group_id, group_cols in groups.items()
                if next_col in group_cols
            )
            selection_rows.append(
                {
                    "feature": next_col,
                    "feature_group": group_id,
                    "group_size": len(groups[group_id]),
                    "within_group_rank": sum(
                        row["feature_group"] == group_id
                        for row in selection_rows
                    )
                    + 1,
                    "mean_distance_to_selected_group_partner": candidate_distances[next_col],
                }
            )
            remaining_cols.remove(next_col)

    selected_cols = selected_cols[:max_features]
    selection = pd.DataFrame(selection_rows)
    selection = selection[selection["feature"].isin(selected_cols)].reset_index(drop=True)
    return features.loc[:, selected_cols], selection, groups


def feature_group_table(
    feature_groups: dict[int, list[str]],
    selected_feature_groups: list[dict[str, object]],
) -> pd.DataFrame:
    selected_by_group: dict[int, list[str]] = {}
    for row in selected_feature_groups:
        group_id = int(row["feature_group"])
        selected_by_group.setdefault(group_id, []).append(str(row["feature"]))

    rows = []
    for group_id, group_features in feature_groups.items():
        selected_features = selected_by_group.get(group_id, [])
        rows.append(
            {
                "feature_group": group_id,
                "group_size": len(group_features),
                "all_features": "; ".join(group_features),
                "selected_features": "; ".join(selected_features),
                "selected_feature_1": selected_features[0] if len(selected_features) > 0 else "",
                "selected_feature_2": selected_features[1] if len(selected_features) > 1 else "",
            }
        )
    return pd.DataFrame(rows)


def save_feature_group_chart(group_table: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if group_table.empty:
        return

    rows = []
    for row in group_table.itertuples(index=False):
        all_text = textwrap.fill(str(row.all_features), width=78)
        selected_text = textwrap.fill(str(row.selected_features), width=44)
        line_count = max(
            all_text.count("\n") + 1,
            selected_text.count("\n") + 1,
            1,
        )
        rows.append(
            {
                "feature_group": int(row.feature_group),
                "group_size": int(row.group_size),
                "all_features": all_text,
                "selected_features": selected_text,
                "line_count": line_count,
            }
        )

    row_heights = [0.34 + 0.24 * row["line_count"] for row in rows]
    fig_height = max(10.0, 1.5 + sum(row_heights))
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Hierarchical Feature Groups and Selected Features",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    left = 0.02
    top = 0.96
    table_width = 0.96
    header_height = 0.035
    col_widths = [0.055, 0.045, 0.62, 0.28]
    col_x = [left]
    for width in col_widths[:-1]:
        col_x.append(col_x[-1] + width * table_width)

    total_row_height = sum(row_heights)
    scale = (top - 0.03 - header_height) / total_row_height
    row_heights = [height * scale for height in row_heights]

    headers = ["Group", "N", "All Features in Group", "Selected Features"]
    y = top - header_height
    for idx, header in enumerate(headers):
        width = col_widths[idx] * table_width
        ax.add_patch(
            Rectangle(
                (col_x[idx], y),
                width,
                header_height,
                facecolor="#1f2937",
                edgecolor="#cccccc",
                linewidth=0.8,
            )
        )
        ax.text(
            col_x[idx] + 0.006,
            y + header_height / 2,
            header,
            ha="left",
            va="center",
            fontsize=8.5,
            color="white",
            fontweight="bold",
        )

    y -= row_heights[0] if row_heights else 0
    for row_idx, row in enumerate(rows):
        if row_idx > 0:
            y -= row_heights[row_idx]
        facecolor = "#f7f7f7" if row_idx % 2 else "white"
        values = [
            f"{row['feature_group']:02d}",
            str(row["group_size"]),
            row["all_features"],
            row["selected_features"],
        ]
        for col_idx, value in enumerate(values):
            width = col_widths[col_idx] * table_width
            cell_color = "#eaf2ff" if col_idx == 3 else facecolor
            ax.add_patch(
                Rectangle(
                    (col_x[col_idx], y),
                    width,
                    row_heights[row_idx],
                    facecolor=cell_color,
                    edgecolor="#cccccc",
                    linewidth=0.8,
                )
            )
            ax.text(
                col_x[col_idx] + 0.006,
                y + row_heights[row_idx] - 0.008,
                value,
                ha="left",
                va="top",
                fontsize=7.8,
                color="#111827",
                linespacing=1.18,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def clean_variables(
    data: pd.DataFrame,
    max_features: int,
    n_groups: int,
    features_per_group: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if max_features >= 50:
        raise ValueError("--max-features must be less than 50.")
    if max_features < 1:
        raise ValueError("--max-features must be positive.")
    if features_per_group < 1:
        raise ValueError("--features-per-group must be positive.")

    original_feature_cols = feature_columns(data)
    if not original_feature_cols:
        raise ValueError("No numeric feature columns were found.")

    raw_features = data[original_feature_cols].replace([np.inf, -np.inf], np.nan)
    variable_features, zero_variance_dropped = drop_zero_variance_features(raw_features)
    scaled_features, zero_iqr_dropped = robust_scale_features(variable_features)
    if scaled_features.empty:
        raise ValueError("No feature columns remain after constant and zero-IQR filtering.")
    selected_features, selection, feature_groups = select_relevant_features(
        scaled_features,
        max_features,
        n_groups,
        features_per_group,
    )

    passthrough_cols = [
        col
        for col in data.columns
        if col not in original_feature_cols and col.lower() not in DROP_OUTPUT_COLUMNS
    ]
    cleaned = pd.concat([data[passthrough_cols], selected_features], axis=1)

    summary = {
        "rows": len(data),
        "input_feature_count": len(original_feature_cols),
        "zero_variance_removed": len(zero_variance_dropped),
        "zero_iqr_removed": len(zero_iqr_dropped),
        "feature_group_count": len(feature_groups),
        "features_per_group": features_per_group,
        "selected_feature_count": selected_features.shape[1],
        "scoring_method": "average-linkage hierarchical clustering; D = 1 - |Spearman M|",
        "selected_features": selection["feature"].tolist(),
        "selected_feature_groups": selection.to_dict(orient="records"),
        "feature_group_members": [
            {
                "feature_group": group_id,
                "group_size": len(group_cols),
                "all_features": group_cols,
            }
            for group_id, group_cols in feature_groups.items()
        ],
    }
    return cleaned, summary


def read_feature_folder(input_dir: Path) -> tuple[pd.DataFrame, dict[Path, pd.Index]]:
    csv_paths = feature_csv_paths(input_dir)

    frames = []
    row_indices_by_path: dict[Path, pd.Index] = {}
    next_index = 0

    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if "source_file" in frame.columns:
            frame["source_file"] = csv_path.name
        else:
            frame.insert(0, "source_file", csv_path.name)
        frame.index = pd.RangeIndex(next_index, next_index + len(frame))
        row_indices_by_path[csv_path] = frame.index
        next_index += len(frame)
        frames.append(frame)

    return pd.concat(frames, axis=0), row_indices_by_path


def write_cleaned_outputs(
    cleaned: pd.DataFrame,
    row_indices_by_path: dict[Path, pd.Index],
    output_dir: Path,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_path, row_indices in row_indices_by_path.items():
        output_path = output_dir / f"{csv_path.stem}_cleaned.csv"
        cleaned.loc[row_indices].to_csv(output_path, index=False)

    cleaned.to_csv(output_dir / "all_cell_features_cleaned.csv", index=False)

    feature_groups = {
        int(row["feature_group"]): list(row["all_features"])
        for row in summary["feature_group_members"]
    }
    group_table = feature_group_table(
        feature_groups,
        summary["selected_feature_groups"],
    )
    group_table.to_csv(output_dir / "feature_group_selection_table.csv", index=False)
    save_feature_group_chart(
        group_table,
        output_dir / "feature_group_selection_chart.png",
    )


def save_matrix_heatmap(
    matrix: pd.DataFrame,
    group_ids: list[int],
    output_path: Path,
    *,
    is_distance: bool,
) -> None:
    import matplotlib.pyplot as plt

    count = len(matrix)
    side = max(12.0, count * 0.22 + 5.0)
    fig, ax = plt.subplots(figsize=(side, side))
    plot = ax.imshow(
        matrix.to_numpy(),
        cmap="viridis" if is_distance else "RdBu_r",
        vmin=0.0 if is_distance else -1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    labels = [f"G{group_id:02d} | {col}" for group_id, col in zip(group_ids, matrix.columns)]
    font_size = 7 if count <= 50 else 5
    ax.set_xticks(range(count), labels=labels, rotation=90, fontsize=font_size)
    ax.set_yticks(range(count), labels=labels, fontsize=font_size)
    ax.tick_params(length=0)
    for idx in range(1, count):
        if group_ids[idx] != group_ids[idx - 1]:
            ax.axhline(idx - 0.5, color="#777777", linewidth=0.5)
            ax.axvline(idx - 0.5, color="#777777", linewidth=0.5)
    title = "Distance D = 1 - |Spearman M|" if is_distance else "Spearman Correlation M"
    ax.set_title(f"{title} ({count} features, ordered by group)", pad=16)
    colorbar = fig.colorbar(plot, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Distance D" if is_distance else "Spearman rho")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_feature_matrices(
    data: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    group_by_feature = {
        col: int(row["feature_group"])
        for row in summary["feature_group_members"]
        for col in row["all_features"]
    }
    columns = list(group_by_feature)
    # Match the scaling and median imputation used for hierarchical clustering.
    scaled, _ = robust_scale_features(data[columns].replace([np.inf, -np.inf], np.nan))
    correlation, distance = feature_correlation_matrices(scaled)
    selected = set(summary["selected_features"])
    for scope, names in (
        ("all", columns),
        ("selected", [col for col in columns if col in selected]),
    ):
        group_ids = [group_by_feature[col] for col in names]
        for kind, matrix, is_distance in (
            ("spearman_correlation", correlation, False),
            ("distance", distance, True),
        ):
            subset = matrix.loc[names, names]
            stem = f"{scope}_feature_{kind}"
            subset.to_csv(output_dir / f"{stem}_matrix.csv", index_label="feature")
            save_matrix_heatmap(
                subset, group_ids, output_dir / f"{stem}_heatmap.png",
                is_distance=is_distance,
            )


def main() -> None:
    args = parse_args()
    data, row_indices_by_path = read_feature_folder(args.input_dir)

    cleaned, summary = clean_variables(
        data,
        args.max_features,
        args.feature_groups,
        args.features_per_group,
    )
    write_cleaned_outputs(cleaned, row_indices_by_path, args.output_dir, summary)
    write_feature_matrices(data, summary, args.output_dir)

    print(f"Saved cleaned files to: {args.output_dir}")
    print(f"Input CSV files: {len(row_indices_by_path)}")
    print(f"Rows: {summary['rows']}")
    print(f"Input numeric feature count: {summary['input_feature_count']}")
    print(f"Zero-variance features removed: {summary['zero_variance_removed']}")
    print(f"Zero-IQR features removed before robust scaling: {summary['zero_iqr_removed']}")
    print(f"Feature groups: {summary['feature_group_count']}")
    print(f"Features requested per group: {summary['features_per_group']}")
    print(f"Selected feature count: {summary['selected_feature_count']}")
    print(f"Feature scoring: {summary['scoring_method']}")
    print("Selected features:")
    for row in summary["selected_feature_groups"]:
        print(f"  - group {row['feature_group']:02d}: {row['feature']}")


if __name__ == "__main__":
    main()
