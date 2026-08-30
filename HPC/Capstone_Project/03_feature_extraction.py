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

For HPC array jobs, pass --slide-name so each task processes one slide and
writes only that slide's feature CSV plus a slide-specific summary.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
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
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-cell features from Script02 CellViT ROI masks/RGB images."
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
    parser.add_argument("--out-dir", default=None)
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
    return parser.parse_args()


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
        "orientation_rad": float(prop.orientation),
        "orientation_deg": float(np.degrees(prop.orientation)),
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
        features["mitotic_figure_density_local"] = np.nan
        features["tufting_budding_index_proxy"] = np.nan
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

        orientations = df.loc[valid_points, "orientation_rad"].to_numpy(dtype=float)
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


def main() -> None:
    args = parse_args()
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

    for slide, slide_df in all_features.groupby("slide", sort=True):
        out_path = out_dir / f"{slide}_cell_features.csv"
        slide_df.to_csv(out_path, index=False)
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


if __name__ == "__main__":
    main()
