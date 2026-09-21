#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_feature_extraction.py

Extract per-cell morphology, stain, texture, neighborhood, and approximate
fallopian-tube epithelial architecture features from Script02 CellViT outputs.

Inputs expected under --cellvit-root:
  spatial_data/*_cell_coords.csv
  region_mask/mask_<slide>_<region>.tif
  region_rgb/rgb_<slide>_<region>.tif
  ./stain_references/<slide>_stain_ref.npz by default, or --stain-ref-dir

Outputs:
  feature_data/*_cell_features.csv
  feature_data/all_cell_features.csv
  feature_data/feature_extraction_summary.json

By default, extraction is followed by global variable cleaning and feature selection.
Cleaned CSVs, the selection table, and correlation/distance charts are written to
feature_data_cleaned/ (override with --cleaned-out-dir).

For HPC array jobs, use --stage extract --slide-name NAME. After every task
succeeds, run --stage clean once across the raw feature directory. Script03.sh
submits these two stages with an afterok dependency.

Examples:
  python 03_feature_extraction.py --out-dir ./feature_data
  python 03_feature_extraction.py --stage clean --out-dir ./feature_data
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree
from scipy.stats import kurtosis, skew
from skimage.color import rgb2hed
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, peak_local_max
from skimage.filters import gabor
from skimage.io import imread
from skimage.measure import find_contours, perimeter as sk_perimeter, regionprops
from skimage.morphology import binary_dilation, disk

try:
    import cv2
except ImportError:  # OpenCV is useful but not required for this script.
    cv2 = None


CELLVIT_ROOT = "./cellvit_output"
STAIN_REF_DIR = "./stain_references"
OUT_SUBDIR = "feature_data"
MPP = 0.2215
FOURIER_HARMONICS = 10
GLCM_LEVELS = 32
K_NEIGHBORS = 6
RADIUS_UM = (20.0, 50.0, 100.0)
CYTO_RING_UM = 2.0
MIN_CELL_AREA_PX = 200
EPS = 1e-9


DEFAULT_INPUT_DIR = "./feature_data"
DEFAULT_OUTPUT_DIR = "./feature_data_cleaned"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-cell features, then clean and select features across slides."
    )
    parser.add_argument(
        "--stage", choices=("all", "extract", "clean"), default="all",
        help="all: extract then clean (default); extract: array task; clean: existing CSVs.",
    )
    parser.add_argument("--cellvit-root", default=CELLVIT_ROOT)
    parser.add_argument(
        "--slide-name",
        default=None,
        help="Only process this slide. Intended for SLURM array jobs.",
    )
    parser.add_argument(
        "--stain-ref-dir",
        default=STAIN_REF_DIR,
        help="Directory containing slide-specific SlideName_stain_ref.npz files.",
    )
    parser.add_argument(
        "--allow-missing-stain-ref",
        action="store_true",
        help="Allow rgb2hed fallback for texture when a slide-specific stain reference is missing.",
    )
    parser.add_argument(
        "--out-dir", "--input-dir", dest="out_dir", default=OUT_SUBDIR,
        help="Raw feature output directory; also the input directory for --stage clean.",
    )
    parser.add_argument(
        "--cleaned-out-dir", "--output-dir", dest="cleaned_out_dir",
        type=Path, default=Path(DEFAULT_OUTPUT_DIR),
        help="Output directory for cleaned CSVs, feature selection table and charts.",
    )
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--feature-groups", type=int, default=DEFAULT_FEATURE_GROUPS)
    parser.add_argument("--features-per-group", type=int, default=DEFAULT_FEATURES_PER_GROUP)
    parser.add_argument("--mpp", type=float, default=MPP, help="Microns per level-0 pixel.")
    parser.add_argument("--fourier-harmonics", type=int, default=FOURIER_HARMONICS)
    parser.add_argument("--glcm-levels", type=int, default=GLCM_LEVELS)
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    parser.add_argument(
        "--radius-um",
        type=float,
        nargs="+",
        default=list(RADIUS_UM),
        help="Neighborhood radii in microns.",
    )
    parser.add_argument("--cyto-ring-um", type=float, default=CYTO_RING_UM)
    parser.add_argument(
        "--max-regions",
        type=int,
        default=None,
        help="Optional debugging limit on processed regions.",
    )
    parser.add_argument(
        "--max-cells-per-region",
        type=int,
        default=None,
        help="Optional debugging limit on processed cells per region.",
    )
    parser.add_argument(
        "--skip-texture",
        action="store_true",
        help="Skip GLCM/LBP/Gabor/wavelet features for faster smoke tests.",
    )
    args = parser.parse_args()
    if args.slide_name and args.stage != "extract":
        parser.error("--slide-name requires --stage extract; run --stage clean after all slides finish.")
    if args.stage != "extract":
        if not 1 <= args.max_features < 50:
            parser.error("--max-features must be between 1 and 49.")
        if args.feature_groups < 1 or args.features_per_group < 1:
            parser.error("--feature-groups and --features-per-group must be positive.")
        if Path(args.out_dir).resolve() == args.cleaned_out_dir.resolve():
            parser.error("Raw and cleaned output directories must be different.")
    return args


def safe_float(value) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def nan_dict(keys: Iterable[str]) -> dict[str, float]:
    return {key: np.nan for key in keys}


def stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    keys = [
        "mean",
        "median",
        "std",
        "skew",
        "kurtosis",
        "min",
        "max",
        "p10",
        "p90",
    ]
    if values.size == 0:
        return nan_dict(f"{prefix}_{key}" for key in keys)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_skew": float(skew(values, bias=False)) if values.size > 2 else np.nan,
        f"{prefix}_kurtosis": float(kurtosis(values, bias=False)) if values.size > 3 else np.nan,
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def region_tag(slide: str, region: str) -> str:
    return f"{slide}_{region}"


def mask_path(cellvit_root: Path, slide: str, region: str) -> Path:
    return cellvit_root / "region_mask" / f"mask_{region_tag(slide, region)}.tif"


def rgb_path(cellvit_root: Path, slide: str, region: str) -> Path:
    return cellvit_root / "region_rgb" / f"rgb_{region_tag(slide, region)}.tif"


def filter_small_area_noise(
    labels: np.ndarray,
    region_df: pd.DataFrame,
    min_area_px: int = MIN_CELL_AREA_PX,
) -> tuple[np.ndarray, pd.DataFrame]:
    area_px = pd.to_numeric(region_df["area_px"], errors="coerce")
    cell_labels = pd.to_numeric(region_df["cell_label"], errors="coerce")
    noise_mask = area_px < float(min_area_px)
    removed_labels = cell_labels[noise_mask].dropna().astype(int).unique()
    filtered_df = region_df.loc[~noise_mask].copy()

    if removed_labels.size == 0:
        return labels, filtered_df

    filtered = labels.copy()
    filtered[np.isin(filtered, removed_labels)] = 0
    return filtered, filtered_df


def find_cell_csvs(cellvit_root: Path, slide_name: str | None = None) -> list[Path]:
    spatial_dir = cellvit_root / "spatial_data"
    if slide_name:
        exact_path = spatial_dir / f"{slide_name}_cell_coords.csv"
        if exact_path.exists():
            return [exact_path]
        return sorted(spatial_dir.glob(f"{slide_name}*_cell_coords.csv"))
    return sorted(spatial_dir.glob("*_cell_coords.csv"))


def load_all_cell_tables(cellvit_root: Path, slide_name: str | None = None) -> pd.DataFrame:
    frames = []
    for csv_path in find_cell_csvs(cellvit_root, slide_name):
        df = pd.read_csv(csv_path)
        if slide_name and "slide" in df.columns:
            df = df[df["slide"].astype(str) == str(slide_name)]
        if not df.empty:
            frames.append(df)
    if not frames:
        scope = f" for slide {slide_name!r}" if slide_name else ""
        raise FileNotFoundError(
            f"No cell coordinate CSVs found{scope} in {cellvit_root / 'spatial_data'}"
        )
    df_all = pd.concat(frames, ignore_index=True)
    required = {"slide", "region", "cell_label", "cx_roi", "cy_roi", "area_px"}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Cell coordinate CSV is missing required columns: {sorted(missing)}")
    return df_all


def contour_from_mask(binary: np.ndarray) -> np.ndarray:
    contours = find_contours(binary.astype(np.uint8), level=0.5)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    contour = max(contours, key=len)
    return np.column_stack([contour[:, 1], contour[:, 0]]).astype(np.float64)


def cv2_contour_from_mask(binary: np.ndarray) -> np.ndarray | None:
    if cv2 is None:
        return None
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def resample_closed_contour(contour_xy: np.ndarray, n_points: int = 128) -> np.ndarray:
    if contour_xy.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.vstack([contour_xy, contour_xy[0]])
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total <= EPS:
        return np.empty((0, 2), dtype=np.float64)
    samples = np.linspace(0, total, n_points, endpoint=False)
    x = np.interp(samples, cumulative, pts[:, 0])
    y = np.interp(samples, cumulative, pts[:, 1])
    return np.column_stack([x, y])


def fourier_descriptors(contour_xy: np.ndarray, harmonics: int) -> dict[str, float]:
    keys = [f"fourier_h{k}_amp" for k in range(1, harmonics + 1)]
    sampled = resample_closed_contour(contour_xy)
    if sampled.size == 0:
        return nan_dict(keys)
    complex_pts = sampled[:, 0] + 1j * sampled[:, 1]
    complex_pts = complex_pts - np.mean(complex_pts)
    coeffs = np.fft.fft(complex_pts)
    scale = abs(coeffs[1]) if len(coeffs) > 1 and abs(coeffs[1]) > EPS else 1.0
    out = {}
    for k in range(1, harmonics + 1):
        out[f"fourier_h{k}_amp"] = float(abs(coeffs[k]) / scale) if k < len(coeffs) else np.nan
    return out


def concavity_count_and_convexity(binary: np.ndarray, perimeter: float) -> tuple[float, float, float]:
    contour_xy = contour_from_mask(binary)
    if contour_xy.shape[0] < 4:
        return np.nan, np.nan, np.nan

    hull_perimeter = np.nan
    if cv2 is not None:
        contour = cv2_contour_from_mask(binary)
        if contour is not None and len(contour) >= 4:
            hull = cv2.convexHull(contour, returnPoints=True)
            hull_perimeter = float(cv2.arcLength(hull, True))
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is not None and len(hull_indices) >= 3:
                try:
                    defects = cv2.convexityDefects(contour, hull_indices)
                    if defects is not None:
                        depths = defects[:, 0, 3].astype(np.float64) / 256.0
                        significant = depths > 1.0
                        concavity_count = int(np.sum(significant))
                        mean_depth = (
                            float(np.mean(depths[significant])) if np.any(significant) else 0.0
                        )
                        convexity = hull_perimeter / (perimeter + EPS) if perimeter > 0 else np.nan
                        return float(concavity_count), float(mean_depth), convexity
                except cv2.error:
                    pass

    try:
        hull = ConvexHull(contour_xy)
        hull_pts = contour_xy[hull.vertices]
        hull_closed = np.vstack([hull_pts, hull_pts[0]])
        hull_perimeter = float(
            np.sum(np.sqrt(np.sum(np.diff(hull_closed, axis=0) ** 2, axis=1)))
        )
    except Exception:
        hull_perimeter = np.nan

    centroid = np.mean(contour_xy, axis=0)
    radial = np.sqrt(np.sum((contour_xy - centroid) ** 2, axis=1))
    if radial.size < 8:
        concavity_count = np.nan
        mean_depth = np.nan
    else:
        smooth = ndi.uniform_filter1d(radial, size=max(5, radial.size // 20), mode="wrap")
        depth = smooth - radial
        threshold = max(1.0, float(np.std(depth)))
        dips = depth > threshold
        starts = np.flatnonzero(dips & ~np.roll(dips, 1))
        concavity_count = float(len(starts))
        mean_depth = float(np.mean(depth[dips])) if np.any(dips) else 0.0

    convexity = hull_perimeter / (perimeter + EPS) if perimeter > 0 and np.isfinite(hull_perimeter) else np.nan
    return float(concavity_count), float(mean_depth), convexity


def box_counting_dimension(binary_boundary: np.ndarray) -> float:
    if not np.any(binary_boundary):
        return np.nan
    h, w = binary_boundary.shape
    max_size = min(h, w)
    if max_size < 8:
        return np.nan
    sizes = []
    box = 2
    while box <= max_size // 2:
        sizes.append(box)
        box *= 2
    if len(sizes) < 2:
        return np.nan

    counts = []
    for size in sizes:
        pad_h = int(math.ceil(h / size) * size - h)
        pad_w = int(math.ceil(w / size) * size - w)
        padded = np.pad(binary_boundary, ((0, pad_h), (0, pad_w)), mode="constant")
        blocks = padded.reshape(
            padded.shape[0] // size,
            size,
            padded.shape[1] // size,
            size,
        )
        counts.append(np.sum(blocks.any(axis=(1, 3))))
    counts = np.asarray(counts, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    valid = counts > 0
    if np.sum(valid) < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(1.0 / sizes[valid]), np.log(counts[valid]), 1)
    return float(slope)


def morphometric_features(prop, binary: np.ndarray, harmonics: int) -> dict[str, float]:
    area = float(prop.area)
    perimeter = float(prop.perimeter)
    if perimeter <= 0:
        perimeter = float(sk_perimeter(binary, neighborhood=8))
    major = float(prop.major_axis_length)
    minor = float(prop.minor_axis_length)
    convex_area = float(prop.convex_area)
    equivalent_diameter = float(prop.equivalent_diameter)

    concavity_count, concavity_depth_mean, convexity = concavity_count_and_convexity(
        binary, perimeter
    )
    contour_xy = contour_from_mask(binary)
    boundary = binary ^ ndi.binary_erosion(binary)

    features = {
        "area_px": area,
        "area_um2": area,
        "perimeter_px": perimeter,
        "perimeter_um": perimeter,
        "equivalent_diameter_px": equivalent_diameter,
        "equivalent_diameter_um": equivalent_diameter,
        "major_axis_length_px": major,
        "major_axis_length_um": major,
        "minor_axis_length_px": minor,
        "minor_axis_length_um": minor,
        "aspect_ratio": major / (minor + EPS) if minor > 0 else np.nan,
        "eccentricity": float(prop.eccentricity),
        "circularity": (4.0 * math.pi * area) / ((perimeter**2) + EPS),
        "roundness": (4.0 * area) / (math.pi * (major**2) + EPS) if major > 0 else np.nan,
        "solidity": float(prop.solidity),
        "convex_hull_area_px": convex_area,
        "convexity": convexity,
        "concavity_count": concavity_count,
        "concavity_depth_mean_px": concavity_depth_mean,
        "perimeter_area_ratio": perimeter / (area + EPS),
        "nuclear_membrane_irregularity_index": 1.0 / (convexity + EPS)
        if np.isfinite(convexity) and convexity > 0
        else np.nan,
        "fractal_dimension_boundary": box_counting_dimension(boundary),
    }
    features.update(fourier_descriptors(contour_xy, harmonics))
    return features


def scale_morphometry_to_um(features: dict[str, float], mpp: float) -> None:
    for key in list(features):
        if key.endswith("_um"):
            px_key = key[:-3] + "_px"
            if px_key in features and np.isfinite(features[px_key]):
                features[key] = float(features[px_key] * mpp)
    if np.isfinite(features.get("area_px", np.nan)):
        features["area_um2"] = float(features["area_px"] * (mpp**2))
    if np.isfinite(features.get("convex_hull_area_px", np.nan)):
        features["convex_hull_area_um2"] = float(features["convex_hull_area_px"] * (mpp**2))


def crop_with_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    minr, minc, maxr, maxc = bbox
    return arr[minr:maxr, minc:maxc]


def masked_channel_values(channel: np.ndarray, binary: np.ndarray) -> np.ndarray:
    if channel.shape != binary.shape:
        raise ValueError("Channel and mask shapes do not match.")
    return channel[binary]


class StainReference(NamedTuple):
    path: Path
    stain_matrix: np.ndarray
    max_c: np.ndarray
    metadata: dict


def convert_rgb_to_od(img_rgb: np.ndarray) -> np.ndarray:
    img = img_rgb[:, :, :3].astype(np.float64)
    img[img <= 0] = 1.0
    return -np.log(img / 255.0)


def available_stain_reference_names(stain_ref_dir: Path) -> list[str]:
    return sorted(
        path.name.removesuffix("_stain_ref.npz")
        for path in stain_ref_dir.glob("*_stain_ref.npz")
    )


def load_stain_reference(stain_ref_dir: Path, slide: str) -> StainReference | None:
    ref_path = stain_ref_dir / f"{slide}_stain_ref.npz"
    if not ref_path.exists():
        return None
    with np.load(ref_path, allow_pickle=False) as payload:
        stain_matrix = np.asarray(payload["stain_matrix"], dtype=np.float64)
        max_c = np.asarray(payload["max_c"], dtype=np.float64).reshape(-1)
        metadata_json = str(payload["metadata_json"].item()) if "metadata_json" in payload else "{}"
    if stain_matrix.shape != (2, 3) or max_c.size < 2:
        raise ValueError(
            f"Invalid stain reference in {ref_path}: expected stain_matrix (2, 3) and max_c with 2 values."
        )
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata_json in {ref_path}: {exc}") from exc
    metadata_slide = str(metadata.get("slide_name", slide))
    if metadata_slide != slide:
        raise ValueError(
            f"Stain reference filename matches {slide!r}, but metadata slide_name is {metadata_slide!r}."
        )
    max_c = np.maximum(max_c[:2], EPS)
    return StainReference(ref_path, stain_matrix, max_c, metadata)


def normalized_hematoxylin_channel(
    rgb_crop: np.ndarray,
    stain_reference: StainReference | None,
) -> np.ndarray:
    if stain_reference is None:
        return rgb2hed(rgb_crop[:, :, :3])[:, :, 0]

    od = convert_rgb_to_od(rgb_crop).reshape((-1, 3))
    concentrations = np.linalg.lstsq(
        stain_reference.stain_matrix.T,
        od.T,
        rcond=None,
    )[0].T
    concentrations = np.clip(concentrations, 0, None)
    h_channel = concentrations[:, 0] / stain_reference.max_c[0]
    return h_channel.reshape(rgb_crop.shape[:2])


def cytoplasmic_ring_mask(
    label_crop: np.ndarray,
    cell_label: int,
    binary: np.ndarray,
    radius_px: int,
) -> np.ndarray:
    if radius_px <= 0:
        return np.zeros_like(binary, dtype=bool)
    dilated = binary_dilation(binary, footprint=disk(radius_px))
    ring = dilated & ~binary
    occupied_by_other = (label_crop != 0) & (label_crop != cell_label)
    return ring & ~occupied_by_other


def intensity_features(
    rgb_crop: np.ndarray,
    label_crop: np.ndarray,
    cell_label: int,
    binary: np.ndarray,
    area_px: float,
    cyto_ring_px: int,
) -> dict[str, float]:
    if rgb_crop.ndim != 3 or rgb_crop.shape[2] < 3:
        return {}
    hed = rgb2hed(rgb_crop[:, :, :3])
    hematoxylin = hed[:, :, 0]
    eosin = hed[:, :, 1]
    residual = hed[:, :, 2]

    h_vals = masked_channel_values(hematoxylin, binary)
    e_vals = masked_channel_values(eosin, binary)
    r_vals = masked_channel_values(residual, binary)
    ring = cytoplasmic_ring_mask(label_crop, cell_label, binary, cyto_ring_px)
    h_ring = hematoxylin[ring]
    e_ring = eosin[ring]

    h_mean = float(np.mean(h_vals)) if h_vals.size else np.nan
    e_mean = float(np.mean(e_vals)) if e_vals.size else np.nan
    h_ring_mean = float(np.mean(h_ring)) if h_ring.size else np.nan
    e_ring_mean = float(np.mean(e_ring)) if e_ring.size else np.nan

    features = {}
    features.update(stats("hematoxylin", h_vals))
    features.update(stats("eosin", e_vals))
    features.update(stats("residual", r_vals))
    features.update(stats("cytoplasm_ring_hematoxylin", h_ring))
    features.update(stats("cytoplasm_ring_eosin", e_ring))
    features.update(
        {
            "he_ratio_mean": h_mean / (e_mean + EPS) if np.isfinite(h_mean) and np.isfinite(e_mean) else np.nan,
            "integrated_nuclear_density_h": h_mean * area_px if np.isfinite(h_mean) else np.nan,
            "integrated_nuclear_density_e": e_mean * area_px if np.isfinite(e_mean) else np.nan,
            "nuclear_cytoplasm_h_contrast": h_mean - h_ring_mean
            if np.isfinite(h_mean) and np.isfinite(h_ring_mean)
            else np.nan,
            "nuclear_cytoplasm_e_contrast": e_mean - e_ring_mean
            if np.isfinite(e_mean) and np.isfinite(e_ring_mean)
            else np.nan,
            "cytoplasm_ring_area_px": float(np.sum(ring)),
            "nc_area_ratio_proxy": area_px / (float(np.sum(ring)) + EPS) if np.sum(ring) > 0 else np.nan,
        }
    )
    return features


def quantize_image(values: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    vals = values[mask]
    if vals.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + EPS
    scaled = np.clip((values - lo) / (hi - lo), 0, 1)
    return np.floor(scaled * (levels - 1)).astype(np.uint8)


def glcm_features(h_channel: np.ndarray, binary: np.ndarray, levels: int) -> dict[str, float]:
    keys = [
        "glcm_contrast",
        "glcm_correlation",
        "glcm_energy",
        "glcm_homogeneity",
        "glcm_entropy",
    ]
    if np.sum(binary) < 8:
        return nan_dict(keys)
    q = quantize_image(h_channel, binary, levels)
    fill = int(np.median(q[binary]))
    q_masked = np.where(binary, q, fill).astype(np.uint8)
    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    matrix = graycomatrix(
        q_masked,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    entropy = -np.sum(matrix * np.log2(matrix + EPS), axis=(0, 1))
    return {
        "glcm_contrast": float(np.mean(graycoprops(matrix, "contrast"))),
        "glcm_correlation": float(np.mean(graycoprops(matrix, "correlation"))),
        "glcm_energy": float(np.mean(graycoprops(matrix, "energy"))),
        "glcm_homogeneity": float(np.mean(graycoprops(matrix, "homogeneity"))),
        "glcm_entropy": float(np.mean(entropy)),
    }


def lbp_features(h_channel: np.ndarray, binary: np.ndarray) -> dict[str, float]:
    bins = 10
    keys = [f"lbp_bin_{i}" for i in range(bins)]
    if np.sum(binary) < 8:
        return nan_dict(keys)
    vals = h_channel[binary]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + EPS
    image = np.clip((h_channel - lo) / (hi - lo), 0, 1)
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp[binary], bins=np.arange(bins + 1), range=(0, bins), density=True)
    return {f"lbp_bin_{i}": float(hist[i]) for i in range(bins)}


def gabor_features(h_channel: np.ndarray, binary: np.ndarray) -> dict[str, float]:
    features = {}
    if np.sum(binary) < 8:
        for freq in (0.15, 0.3):
            for theta_idx in range(4):
                features[f"gabor_f{freq:.2f}_theta{theta_idx}_mean"] = np.nan
                features[f"gabor_f{freq:.2f}_theta{theta_idx}_std"] = np.nan
        return features
    vals = h_channel[binary]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        hi = lo + EPS
    image = np.clip((h_channel - lo) / (hi - lo), 0, 1)
    image = np.where(binary, image, np.median(image[binary]))
    for freq in (0.15, 0.3):
        for theta_idx, theta in enumerate((0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
            real, imag = gabor(image, frequency=freq, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)
            mvals = magnitude[binary]
            features[f"gabor_f{freq:.2f}_theta{theta_idx}_mean"] = float(np.mean(mvals))
            features[f"gabor_f{freq:.2f}_theta{theta_idx}_std"] = float(np.std(mvals))
    return features


def haar_wavelet_energy(h_channel: np.ndarray, binary: np.ndarray) -> dict[str, float]:
    keys = ["haar_ll_energy", "haar_lh_energy", "haar_hl_energy", "haar_hh_energy"]
    if np.sum(binary) < 16:
        return nan_dict(keys)
    vals = h_channel[binary]
    fill = float(np.median(vals))
    image = np.where(binary, h_channel, fill)
    h, w = image.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    if h2 < 2 or w2 < 2:
        return nan_dict(keys)
    image = image[:h2, :w2]
    a = image[0::2, 0::2]
    b = image[0::2, 1::2]
    c = image[1::2, 0::2]
    d = image[1::2, 1::2]
    ll = (a + b + c + d) / 4.0
    lh = (a - b + c - d) / 4.0
    hl = (a + b - c - d) / 4.0
    hh = (a - b - c + d) / 4.0
    return {
        "haar_ll_energy": float(np.mean(ll**2)),
        "haar_lh_energy": float(np.mean(lh**2)),
        "haar_hl_energy": float(np.mean(hl**2)),
        "haar_hh_energy": float(np.mean(hh**2)),
    }


def chromatin_clumping_index(h_channel: np.ndarray, binary: np.ndarray) -> float:
    if np.sum(binary) < 16:
        return np.nan
    values = np.where(binary, h_channel, 0.0)
    weights = binary.astype(np.float64)
    mean = ndi.uniform_filter(values, size=5)
    mean_sq = ndi.uniform_filter(values**2, size=5)
    local_var = np.maximum(mean_sq - mean**2, 0)
    return float(np.mean(local_var[weights > 0]))


def nucleolar_proxy_features(h_channel: np.ndarray, binary: np.ndarray) -> dict[str, float]:
    if np.sum(binary) < 16:
        return {
            "nucleolar_peak_count": np.nan,
            "nucleolar_peak_density": np.nan,
            "nucleolar_peak_contrast_max": np.nan,
        }
    vals = h_channel[binary]
    threshold = np.percentile(vals, 90)
    smoothed = ndi.gaussian_filter(np.where(binary, h_channel, np.min(vals)), sigma=1.0)
    coords = peak_local_max(
        smoothed,
        min_distance=2,
        threshold_abs=threshold,
        labels=binary.astype(np.uint8),
        exclude_border=False,
    )
    if len(coords) == 0:
        max_contrast = 0.0
    else:
        max_contrast = float(np.max(smoothed[coords[:, 0], coords[:, 1]]) - np.median(vals))
    return {
        "nucleolar_peak_count": float(len(coords)),
        "nucleolar_peak_density": float(len(coords) / (np.sum(binary) + EPS)),
        "nucleolar_peak_contrast_max": max_contrast,
    }


def texture_features(h_channel: np.ndarray, binary: np.ndarray, levels: int) -> dict[str, float]:
    features = {}
    features.update(glcm_features(h_channel, binary, levels))
    features.update(lbp_features(h_channel, binary))
    features.update(gabor_features(h_channel, binary))
    features.update(haar_wavelet_energy(h_channel, binary))
    features.update(nucleolar_proxy_features(h_channel, binary))
    features["chromatin_clumping_index"] = chromatin_clumping_index(h_channel, binary)
    return features


def graph_features(points: np.ndarray, k: int, radii_px: list[float]) -> dict[int, dict[str, float]]:
    n = len(points)
    out = {i: {} for i in range(n)}
    if n == 0:
        return out
    tree = cKDTree(points)
    max_k = min(k + 1, n)
    dists, idxs = tree.query(points, k=max_k)
    if max_k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    for i in range(n):
        neighbor_dists = dists[i, 1:] if max_k > 1 else np.array([])
        out[i].update(
            {
                "nn_distance_mean_px": float(np.mean(neighbor_dists)) if neighbor_dists.size else np.nan,
                "nn_distance_std_px": float(np.std(neighbor_dists)) if neighbor_dists.size else np.nan,
                "nn_distance_min_px": float(np.min(neighbor_dists)) if neighbor_dists.size else np.nan,
            }
        )
        for radius in radii_px:
            count = len(tree.query_ball_point(points[i], r=radius)) - 1
            area = math.pi * radius * radius
            out[i][f"local_density_r{radius:.1f}px"] = float(count / (area + EPS))
            out[i][f"neighbor_count_r{radius:.1f}px"] = float(count)
            expected = n * area / (np.ptp(points[:, 0]) * np.ptp(points[:, 1]) + EPS)
            out[i][f"ripleys_k_l_proxy_r{radius:.1f}px"] = float(count - expected)

    edges: set[tuple[int, int]] = set()
    if n >= 3:
        try:
            tri = Delaunay(points)
            for simplex in tri.simplices:
                for a, b in ((0, 1), (1, 2), (0, 2)):
                    i, j = sorted((int(simplex[a]), int(simplex[b])))
                    edges.add((i, j))
        except Exception:
            edges = set()
    if not edges and n > 1:
        for i in range(n):
            for j in np.atleast_1d(idxs[i, 1:]):
                edges.add(tuple(sorted((i, int(j)))))

    adjacency = defaultdict(set)
    edge_lengths = defaultdict(list)
    for i, j in edges:
        dist = float(np.linalg.norm(points[i] - points[j]))
        adjacency[i].add(j)
        adjacency[j].add(i)
        edge_lengths[i].append(dist)
        edge_lengths[j].append(dist)

    for i in range(n):
        lengths = np.asarray(edge_lengths[i], dtype=np.float64)
        neigh = adjacency[i]
        possible = len(neigh) * (len(neigh) - 1) / 2
        links = 0
        if possible > 0:
            neigh_list = list(neigh)
            for a_idx in range(len(neigh_list)):
                for b_idx in range(a_idx + 1, len(neigh_list)):
                    if neigh_list[b_idx] in adjacency[neigh_list[a_idx]]:
                        links += 1
        out[i].update(
            {
                "delaunay_degree": float(len(neigh)),
                "delaunay_edge_length_mean_px": float(np.mean(lengths)) if lengths.size else np.nan,
                "delaunay_edge_length_var_px": float(np.var(lengths)) if lengths.size else np.nan,
                "graph_clustering_coefficient": float(links / possible) if possible > 0 else np.nan,
            }
        )

    if n >= 4:
        try:
            vor = Voronoi(points)
            for i, region_idx in enumerate(vor.point_region):
                vertices = vor.regions[region_idx]
                if not vertices or -1 in vertices:
                    out[i]["voronoi_area_px2"] = np.nan
                    out[i]["voronoi_polygon_irregularity"] = np.nan
                    continue
                polygon = vor.vertices[vertices]
                x, y = polygon[:, 0], polygon[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
                perim = np.sum(np.sqrt(np.sum(np.diff(np.vstack([polygon, polygon[0]]), axis=0) ** 2, axis=1)))
                out[i]["voronoi_area_px2"] = float(area)
                out[i]["voronoi_polygon_irregularity"] = float((perim**2) / (4 * math.pi * area + EPS))
        except Exception:
            for i in range(n):
                out[i]["voronoi_area_px2"] = np.nan
                out[i]["voronoi_polygon_irregularity"] = np.nan

    return out


def add_spatial_unit_scaling(features: dict[str, float], mpp: float, radius_px_to_um: dict[str, float]) -> None:
    for key in list(features):
        if key.endswith("_px") and np.isfinite(features[key]):
            features[key[:-3] + "_um"] = float(features[key] * mpp)
        elif key.endswith("_px2") and np.isfinite(features[key]):
            features[key[:-4] + "_um2"] = float(features[key] * (mpp**2))
        elif key.startswith("local_density_r") and key.endswith("px"):
            radius_px = key.split("_r", 1)[1][:-2]
            radius_um = radius_px_to_um.get(radius_px)
            if radius_um is not None and np.isfinite(features[key]):
                features[f"local_density_r{radius_um:g}um"] = float(features[key] / (mpp**2))
        elif key.startswith("neighbor_count_r") and key.endswith("px"):
            radius_px = key.split("_r", 1)[1][:-2]
            radius_um = radius_px_to_um.get(radius_px)
            if radius_um is not None:
                features[f"neighbor_count_r{radius_um:g}um"] = features[key]
        elif key.startswith("ripleys_k_l_proxy_r") and key.endswith("px"):
            radius_px = key.split("_r", 1)[1][:-2]
            radius_um = radius_px_to_um.get(radius_px)
            if radius_um is not None:
                features[f"ripleys_k_l_proxy_r{radius_um:g}um"] = features[key]


def circular_variance(angles_rad: np.ndarray) -> float:
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    angles_rad = angles_rad[np.isfinite(angles_rad)]
    if angles_rad.size == 0:
        return np.nan
    resultant = abs(np.mean(np.exp(1j * 2.0 * angles_rad)))
    return float(1.0 - resultant)


def architecture_features(
    points: np.ndarray,
    orientations: np.ndarray,
    k: int,
) -> dict[int, dict[str, float]]:
    n = len(points)
    out = {i: {} for i in range(n)}
    if n == 0:
        return out
    centered = points - np.mean(points, axis=0)
    if n >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        epithelial_axis = vh[0]
        normal_axis = np.array([-epithelial_axis[1], epithelial_axis[0]])
        along = centered @ epithelial_axis
        perp = centered @ normal_axis
    else:
        along = np.zeros(n)
        perp = np.zeros(n)

    total_thickness = float(np.max(perp) - np.min(perp)) if n > 1 else np.nan
    normalized_position = (perp - np.min(perp)) / (total_thickness + EPS) if np.isfinite(total_thickness) else np.full(n, np.nan)
    tree = cKDTree(points)
    max_k = min(k + 1, n)
    _, idxs = tree.query(points, k=max_k)
    if max_k == 1:
        idxs = idxs[:, None]

    for i in range(n):
        neigh = np.asarray(idxs[i], dtype=int)
        local_perp = perp[neigh]
        local_orient = orientations[neigh]
        local_thickness = float(np.max(local_perp) - np.min(local_perp)) if len(local_perp) > 1 else np.nan
        out[i].update(
            {
                "orientation_coherence_circular_variance": circular_variance(local_orient),
                "orientation_disorder": circular_variance(local_orient),
                "pseudostratification_index": float(np.std(local_perp)) if len(local_perp) > 1 else np.nan,
                "epithelial_layer_thickness_px": local_thickness,
                "basal_to_luminal_position": float(normalized_position[i]),
                "distance_from_basal_line_px": float(perp[i] - np.min(local_perp))
                if len(local_perp) > 0
                else np.nan,
            }
        )
    return out


def assign_types_from_raw_cellvit(
    cellvit_root: Path,
    slide: str,
    region_df: pd.DataFrame,
) -> dict[int, dict[str, float | str]]:
    detection_json = cellvit_root / "raw_cellvit" / slide / "cell_detection.json"
    if not detection_json.exists():
        return {}
    try:
        with open(detection_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    cells = payload.get("cells", [])
    type_map = payload.get("type_map", {})
    if not cells:
        return {}

    roi_points = region_df[["cx_wsi", "cy_wsi"]].to_numpy(dtype=float)
    if roi_points.size == 0 or not np.all(np.isfinite(roi_points)):
        return {}
    cell_points = []
    cell_payload = []
    for cell in cells:
        centroid = cell.get("centroid")
        if centroid and len(centroid) == 2:
            cell_points.append([float(centroid[0]), float(centroid[1])])
            cell_payload.append(cell)
    if not cell_points:
        return {}
    tree = cKDTree(np.asarray(cell_points, dtype=float))
    dists, idxs = tree.query(roi_points, k=1)
    out = {}
    for row_idx, (_, row) in enumerate(region_df.iterrows()):
        if dists[row_idx] > 2.0:
            continue
        cell = cell_payload[int(idxs[row_idx])]
        type_id = cell.get("type")
        out[int(row["cell_label"])] = {
            "type": type_id,
            "type_name": type_map.get(str(type_id), f"type_{type_id}"),
            "type_prob": cell.get("type_prob", np.nan),
        }
    return out


def heterotypic_features(
    points: np.ndarray,
    type_values: list,
    k: int,
) -> dict[int, dict[str, float]]:
    n = len(points)
    out = {i: {"heterotypic_neighbor_fraction": np.nan, "inflammatory_neighbor_fraction": np.nan} for i in range(n)}
    if n <= 1:
        return out
    max_k = min(k + 1, n)
    tree = cKDTree(points)
    _, idxs = tree.query(points, k=max_k)
    if max_k == 1:
        idxs = idxs[:, None]
    lowered = [str(t).lower() if t is not None and not pd.isna(t) else "" for t in type_values]
    for i in range(n):
        neigh = [int(j) for j in np.atleast_1d(idxs[i, 1:])]
        if not neigh:
            continue
        own = lowered[i]
        out[i]["heterotypic_neighbor_fraction"] = float(
            np.mean([lowered[j] != own and lowered[j] != "" and own != "" for j in neigh])
        )
        out[i]["inflammatory_neighbor_fraction"] = float(
            np.mean([
                ("inflam" in lowered[j])
                or ("immune" in lowered[j])
                or ("lymph" in lowered[j])
                for j in neigh
            ])
        )
    return out


def apoptotic_body_proxy(row_features: dict[str, float]) -> float:
    area = row_features.get("area_px", np.nan)
    h_mean = row_features.get("hematoxylin_mean", np.nan)
    circularity = row_features.get("circularity", np.nan)
    irregularity = row_features.get("nuclear_membrane_irregularity_index", np.nan)
    if not all(np.isfinite(v) for v in (area, h_mean, circularity, irregularity)):
        return np.nan
    return float(area < 80 and h_mean > 0.1 and (circularity < 0.55 or irregularity > 1.2))


def process_region(
    cellvit_root: Path,
    slide: str,
    region: str,
    region_df: pd.DataFrame,
    args: argparse.Namespace,
    stain_reference: StainReference | None,
) -> pd.DataFrame:
    m_path = mask_path(cellvit_root, slide, region)
    r_path = rgb_path(cellvit_root, slide, region)
    if not m_path.exists():
        raise FileNotFoundError(f"Missing mask: {m_path}")
    if not r_path.exists():
        raise FileNotFoundError(f"Missing RGB image: {r_path}")

    labels = imread(str(m_path))
    rgb = imread(str(r_path))
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    rgb = rgb[:, :, :3].astype(np.uint8)

    labels, region_df = filter_small_area_noise(labels, region_df)
    props_by_label = {int(prop.label): prop for prop in regionprops(labels)}
    if args.max_cells_per_region:
        region_df = region_df.head(args.max_cells_per_region).copy()

    type_by_label = assign_types_from_raw_cellvit(cellvit_root, slide, region_df)
    rows = []
    cyto_ring_px = max(1, int(round(float(args.cyto_ring_um) / float(args.mpp))))

    for _, row in region_df.iterrows():
        label = int(row["cell_label"])
        prop = props_by_label.get(label)
        if prop is None:
            continue
        minr, minc, maxr, maxc = prop.bbox
        binary = labels[minr:maxr, minc:maxc] == label
        label_crop = labels[minr:maxr, minc:maxc]
        rgb_crop = rgb[minr:maxr, minc:maxc]

        features = row.to_dict()
        morph = morphometric_features(prop, binary, int(args.fourier_harmonics))
        scale_morphometry_to_um(morph, float(args.mpp))
        features.update(morph)

        intensity = intensity_features(
            rgb_crop=rgb_crop,
            label_crop=label_crop,
            cell_label=label,
            binary=binary,
            area_px=float(prop.area),
            cyto_ring_px=cyto_ring_px,
        )
        features.update(intensity)

        if not args.skip_texture:
            h_channel = normalized_hematoxylin_channel(rgb_crop, stain_reference)
            features.update(texture_features(h_channel, binary, int(args.glcm_levels)))

        if label in type_by_label:
            features.update(type_by_label[label])

        features["apoptotic_body_proxy"] = apoptotic_body_proxy(features)
        features["_orientation_rad_for_architecture"] = float(prop.orientation)
        rows.append(features)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    points = df[["cx_roi", "cy_roi"]].to_numpy(dtype=float)
    valid_points = np.all(np.isfinite(points), axis=1)
    if np.any(valid_points):
        valid_indices = np.where(valid_points)[0]
        valid_points_arr = points[valid_points]
        radius_px = [float(r) / float(args.mpp) for r in args.radius_um]
        radius_map = {f"{r:.1f}": float(r * args.mpp) for r in radius_px}
        graph = graph_features(valid_points_arr, int(args.k_neighbors), radius_px)

        orientations = df.loc[valid_points, "_orientation_rad_for_architecture"].to_numpy(dtype=float)
        architecture = architecture_features(valid_points_arr, orientations, int(args.k_neighbors))

        type_values = (
            df.loc[valid_points, "type_name"].tolist()
            if "type_name" in df.columns
            else df.loc[valid_points, "type"].tolist()
            if "type" in df.columns
            else [None] * len(valid_points_arr)
        )
        heterotypic = heterotypic_features(valid_points_arr, type_values, int(args.k_neighbors))

        for local_i, df_i in enumerate(valid_indices):
            merged = {}
            merged.update(graph[local_i])
            merged.update(architecture[local_i])
            merged.update(heterotypic[local_i])
            add_spatial_unit_scaling(merged, float(args.mpp), radius_map)
            for key, value in merged.items():
                df.loc[df.index[df_i], key] = value

        if "equivalent_diameter_px" in df.columns and "nn_distance_min_px" in df.columns:
            radius_proxy = df["equivalent_diameter_px"] / 2.0
            df["nuclear_overlap_crowding_index"] = (
                df["nn_distance_min_px"] < (2.0 * radius_proxy)
            ).astype(float)

        if "epithelial_layer_thickness_px" in df.columns:
            df["epithelial_layer_thickness_um"] = df["epithelial_layer_thickness_px"] * float(args.mpp)
        if "distance_from_basal_line_px" in df.columns:
            df["distance_from_basal_line_um"] = df["distance_from_basal_line_px"] * float(args.mpp)

    df = df.drop(columns=["_orientation_rad_for_architecture"], errors="ignore")
    return df


def write_summary(
    summary_path: Path,
    per_slide_counts: dict[str, int],
    feature_columns: list[str],
    stain_ref_dir: Path,
    slides_using_stain_ref: list[str],
    slides_missing_stain_ref: list[str],
) -> None:
    summary = {
        "per_slide_cell_counts": per_slide_counts,
        "n_feature_columns": len(feature_columns),
        "feature_columns": feature_columns,
        "stain_ref_dir": str(stain_ref_dir),
        "slides_using_stain_ref_for_texture": sorted(slides_using_stain_ref),
        "slides_missing_stain_ref_for_texture": sorted(slides_missing_stain_ref),
        "notes": [
            "Morphology features are derived from Script02 ROI label masks.",
            "Hematoxylin/eosin features use skimage.color.rgb2hed on saved ROI RGB images.",
            "Texture features use slide-specific Macenko hematoxylin concentrations normalized by the slide's stain reference max_c.",
            "Missing stain references stop texture extraction unless --allow-missing-stain-ref is used.",
            "Cytoplasm and N:C features are fixed-width ring proxies around each nucleus.",
            "Polarity, pseudostratification, and layer thickness are PCA/kNN proxies because basement membrane annotations are not present.",
            "Mitotic figure density is left as NaN because no mitosis detector or labels are available.",
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_extraction(args: argparse.Namespace) -> list[Path]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    cellvit_root = Path(args.cellvit_root)
    stain_ref_dir = Path(args.stain_ref_dir)
    out_dir = Path(args.out_dir) if args.out_dir else Path(OUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_all_cell_tables(cellvit_root, args.slide_name)
    grouped = list(cells.groupby(["slide", "region"], sort=True))
    if args.max_regions:
        grouped = grouped[: args.max_regions]

    all_region_features = []
    per_slide_counts: dict[str, int] = defaultdict(int)
    stain_ref_cache: dict[str, StainReference | None] = {}
    slides_using_stain_ref: set[str] = set()
    slides_missing_stain_ref: set[str] = set()

    for idx, ((slide, region), region_df) in enumerate(grouped, start=1):
        print(f"[{idx}/{len(grouped)}] Extracting features: {slide} / {region}", flush=True)
        slide_key = str(slide)
        stain_reference = None
        if not args.skip_texture and slide_key not in stain_ref_cache:
            try:
                stain_ref_cache[slide_key] = load_stain_reference(stain_ref_dir, slide_key)
            except (OSError, KeyError, ValueError) as exc:
                raise ValueError(f"Could not load stain reference for {slide_key}: {exc}") from exc
            if stain_ref_cache[slide_key] is None:
                available = available_stain_reference_names(stain_ref_dir)
                available_text = ", ".join(available) if available else "none"
                message = (
                    f"Missing stain reference for {slide_key}: "
                    f"{stain_ref_dir / f'{slide_key}_stain_ref.npz'}. "
                    f"Available stain references: {available_text}"
                )
                if not args.allow_missing_stain_ref:
                    raise FileNotFoundError(message)
                print(f"  WARNING: {message}; texture will fall back to rgb2hed.", flush=True)
                slides_missing_stain_ref.add(slide_key)
            else:
                slides_using_stain_ref.add(slide_key)
        if not args.skip_texture:
            stain_reference = stain_ref_cache[slide_key]
        try:
            region_features = process_region(
                cellvit_root,
                slide,
                region,
                region_df.copy(),
                args,
                stain_reference,
            )
        except FileNotFoundError as exc:
            print(f"  Skipping region: {exc}", flush=True)
            continue
        if region_features.empty:
            print("  No cells with matching mask labels.", flush=True)
            continue
        all_region_features.append(region_features)
        per_slide_counts[str(slide)] += len(region_features)

    if not all_region_features:
        raise RuntimeError("No feature rows were produced.")

    all_features = pd.concat(all_region_features, ignore_index=True)
    id_cols = [
        col
        for col in [
            "slide",
            "region",
            "cell_label",
            "type",
            "type_name",
            "type_prob",
            "cx_roi",
            "cy_roi",
            "cx_wsi",
            "cy_wsi",
            "cx_wsi_um",
            "cy_wsi_um",
        ]
        if col in all_features.columns
    ]
    other_cols = [col for col in all_features.columns if col not in id_cols]
    all_features = all_features[id_cols + other_cols]

    output_paths = []
    for slide, slide_df in all_features.groupby("slide", sort=True):
        out_path = out_dir / f"{slide}_cell_features.csv"
        slide_df.to_csv(out_path, index=False)
        output_paths.append(out_path)
        print(f"  Wrote {len(slide_df):,} rows -> {out_path}", flush=True)

    all_path = None
    if args.slide_name:
        print("  Slide mode enabled; skipping shared all_cell_features.csv.", flush=True)
    else:
        all_path = out_dir / "all_cell_features.csv"
        all_features.to_csv(all_path, index=False)

    summary_path = (
        out_dir / f"{args.slide_name}_feature_extraction_summary.json"
        if args.slide_name
        else out_dir / "feature_extraction_summary.json"
    )
    write_summary(
        summary_path,
        dict(per_slide_counts),
        other_cols,
        stain_ref_dir,
        list(slides_using_stain_ref),
        list(slides_missing_stain_ref),
    )

    print("\nStep 03 feature extraction complete.", flush=True)
    if all_path:
        print(f"   All features -> {all_path}", flush=True)
    print(f"   Summary      -> {summary_path}", flush=True)
    return output_paths


# Variable cleaning and feature selection (formerly variable_cleaning.py).

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


def read_feature_folder(
    input_dir: Path, csv_paths: list[Path] | None = None,
) -> tuple[pd.DataFrame, dict[Path, pd.Index]]:
    if csv_paths is None:
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


def run_cleaning(
    input_dir: Path,
    output_dir: Path,
    max_features: int = DEFAULT_MAX_FEATURES,
    feature_groups: int = DEFAULT_FEATURE_GROUPS,
    features_per_group: int = DEFAULT_FEATURES_PER_GROUP,
    csv_paths: list[Path] | None = None,
) -> None:
    data, row_indices_by_path = read_feature_folder(input_dir, csv_paths)

    cleaned, summary = clean_variables(
        data,
        max_features,
        feature_groups,
        features_per_group,
    )
    write_cleaned_outputs(cleaned, row_indices_by_path, output_dir, summary)
    write_feature_matrices(data, summary, output_dir)

    print(f"Saved cleaned files to: {output_dir}")
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


def main() -> None:
    args = parse_args()
    output_paths = None
    if args.stage in ("all", "extract"):
        output_paths = run_extraction(args)
    if args.stage in ("all", "clean"):
        run_cleaning(
            Path(args.out_dir), args.cleaned_out_dir, args.max_features,
            args.feature_groups, args.features_per_group, csv_paths=output_paths,
        )


if __name__ == "__main__":
    main()
